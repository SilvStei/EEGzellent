import os
import numpy as np
import joblib
from typing import List, Dict, Any
import argparse
from wettbewerb import EEGDataset
from scipy.signal import welch
import psutil
import time
import pywt  # Wavelet-Transformation für adaptive Fourier-Dekomposition

# Laden des Modells und des Scalers
# Basisverzeichnis ermitteln (Verzeichnis des aktuellen Skripts)
base_dir = os.path.dirname(__file__)

# Pfade relativ zum Basisverzeichnis
model_path = os.path.join(base_dir, 'ensemble_model.joblib')
scaler_path = os.path.join(base_dir, 'scaler.joblib')
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Funktion zum Berechnen der Bandpower
def bandpower(data, sf, band, window_sec=4, relative=False):
    band = np.asarray(band)
    low, high = band

    freqs, psd = welch(data, sf, nperseg=window_sec * sf)

    idx_band = np.logical_and(freqs >= low, freqs <= high)
    bp = np.trapz(psd[idx_band], freqs[idx_band])

    if relative:
        bp /= np.trapz(psd, freqs)
    return bp

# Funktion zum Berechnen der Zero-Crossing-Rate
def zero_crossing_rate(data):
    zero_crossings = np.sum(np.diff(data > 0))
    return zero_crossings

# Funktion für adaptive Fourier-Dekomposition
def adaptive_fourier_decomposition(data):
    coeffs = pywt.wavedec(data, 'db4', level=4)
    power = np.sum([np.sum(c ** 2) for c in coeffs])
    return power

# Funktion zum Berechnen der Hjorth-Mobilität
def hjorth_mobility(data):
    diff_signal = np.diff(data)
    return np.sqrt(np.var(diff_signal) / np.var(data))

# Funktion zum Extrahieren von Features aus rohen EEG-Daten
def extract_features(data, fs):
    features = []
    for channel_data in data:
        line_length = np.sum(np.abs(np.diff(channel_data)))

        theta_power = bandpower(channel_data, fs, [4, 8])
        alpha_power = bandpower(channel_data, fs, [8, 13])
        fourier_power = adaptive_fourier_decomposition(channel_data)
        zero_crossings = zero_crossing_rate(channel_data)

        features.extend([theta_power, alpha_power, fourier_power, zero_crossings, line_length])
    
    return np.array(features)

# Funktion zur Berechnung von Onset und Offset basierend auf Hjorth-Mobilität
def calculate_onset_offset(data, fs):
    window_size = fs
    step_size = fs
    hjorth_values = []

    for start in range(0, len(data[0]) - window_size + 1, step_size):
        window = data[:, start:start + window_size]
        mobility = np.mean([hjorth_mobility(channel) for channel in window])
        hjorth_values.append(mobility)

    hjorth_values = np.array(hjorth_values)
    high_mobility_indices = np.where(hjorth_values > np.percentile(hjorth_values, 95))[0]

    if len(high_mobility_indices) == 0:
        return 0.0, data.shape[1] / fs

    onset_index = high_mobility_indices[0]
    offset_index = high_mobility_indices[-1]

    onset = onset_index * step_size / fs
    offset = (offset_index + window_size) * step_size / fs

    return onset, offset

def predict_labels(channels: List[str], data: np.ndarray, fs: float, reference_system: str, model_name: str='model.json') -> Dict[str, Any]:
    try:
        # Extrahieren der Merkmale aus den Daten
        features = extract_features(data, fs)
        features = scaler.transform([features])
        
        # Vorhersage mit dem geladenen Modell
        seizure_present = model.predict(features)[0]
        seizure_confidence = model.predict_proba(features)[0, seizure_present]
        
        if seizure_present:
            onset, offset = calculate_onset_offset(data, fs)
        else:
            onset = 0.0
            offset = data.shape[1] / fs

        onset_confidence = 1.0
        offset_confidence = 1.0
        
        prediction = {
            "seizure_present": seizure_present,
            "seizure_confidence": seizure_confidence,
            "onset": onset,
            "onset_confidence": onset_confidence,
            "offset": offset,
            "offset_confidence": offset_confidence
        }
        
        return prediction
    except Exception as e:
        print(f"Fehler bei der Verarbeitung der Datei: {e}")
        return {"seizure_present": 0, "seizure_confidence": 0.0, "onset": 0.0, "onset_confidence": 0.0, "offset": 0.0, "offset_confidence": 0.0}

def main(test_dir: str):
    # Startzeit und Prozessinformationen erfassen
    start_time = time.time()
    process = psutil.Process(os.getpid())

    # Laden des Testdatensatzes
    dataset = EEGDataset(test_dir)
    predictions = []
    
    for i in range(len(dataset)):
        id, channels, data, fs, ref_system, label = dataset[i]
        prediction = predict_labels(channels, data, fs, ref_system)
        prediction['id'] = id
        predictions.append(prediction)
    
    # Speichern der Vorhersagen in einer CSV-Datei
    prediction_file = os.path.join(test_dir, "predictions.csv")
    with open(prediction_file, 'w') as f:
        f.write("id,seizure_present,seizure_confidence,onset,onset_confidence,offset,offset_confidence\n")
        for pred in predictions:
            f.write(f"{pred['id']},{pred['seizure_present']},{pred['seizure_confidence']},{pred['onset']},{pred['onset_confidence']},{pred['offset']},{pred['offset_confidence']}\n")
    
    print(f"Vorhersagen wurden in {prediction_file} gespeichert.")
    
    # Endzeit und Prozessinformationen erfassen
    end_time = time.time()
    memory_usage = process.memory_info().rss / (1024 ** 2)  # In MB
    cpu_usage = psutil.cpu_percent(interval=None)
    
    print(f"RAM-Verbrauch: {memory_usage:.2f} MB")
    print(f"CPU-Auslastung: {cpu_usage:.2f}%")
    print(f"Laufzeit: {end_time - start_time:.2f} Sekunden")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make predictions on EEG test data")
    parser.add_argument("--test_dir", type=str, required=True, help="Directory containing the test EEG data")
    args = parser.parse_args()
    
    main(args.test_dir)
