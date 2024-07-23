import os
import numpy as np
import joblib
from typing import List, Dict, Any
import argparse
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, welch
from wettbewerb import EEGDataset

# Laden des Modells und des Scalers
# Basisverzeichnis ermitteln (Verzeichnis des aktuellen Skripts)
base_dir = os.path.dirname(__file__)

# Pfade relativ zum Basisverzeichnis
model_path = os.path.join(base_dir, 'ensemble_model.joblib')
scaler_path = os.path.join(base_dir, 'scaler.joblib')
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Funktion zum Berechnen der Hjorth-Parameter
def hjorth_parameters(data):
    first_deriv = np.diff(data)
    second_deriv = np.diff(first_deriv)
    var_zero = np.var(data)
    var_d1 = np.var(first_deriv)
    var_d2 = np.var(second_deriv)
    
    if var_zero == 0:
        var_zero = 1e-10
    if var_d1 == 0:
        var_d1 = 1e-10
    
    mobility = np.sqrt(var_d1 / var_zero)
    
    if var_d2 == 0:
        var_d2 = 1e-10
    
    complexity = np.sqrt(var_d2 / var_d1) / mobility
    return mobility, complexity

# Funktion zur Berechnung von Onset und Offset basierend auf Hjorth-Mobilität und Energie
def calculate_onset_offset(data, fs):
    window_size = fs
    step_size = fs // 2  # Halb so große Schritte für feinere Analyse
    hjorth_values = []
    energy_values = []

    for start in range(0, len(data[0]) - window_size + 1, step_size):
        window = data[:, start:start + window_size]
        mobility = np.mean([hjorth_parameters(channel)[0] for channel in window])
        energy = np.sum(window ** 2)  # Berechnung der Energie des Fensters
        hjorth_values.append(mobility)
        energy_values.append(energy)

    hjorth_values = np.array(hjorth_values)
    energy_values = np.array(energy_values)
    
    # Glätten der Energiezeitreihe
    smoothed_energy = uniform_filter1d(energy_values, size=5)

    # Identifiziere die Changepoints der maximalen Energie
    energy_diffs = np.diff(smoothed_energy)
    peaks, _ = find_peaks(energy_diffs, height=np.max(energy_diffs) * 0.5)

    if len(peaks) == 0:
        peaks = [np.argmax(energy_diffs)]

    changepoint_index = peaks[0]

    # Bestimme das erste Fenster mit hoher Hjorth-Mobilität, das dem Changepoint am nächsten ist
    high_mobility_indices = np.where(hjorth_values > np.percentile(hjorth_values, 95))[0]
    
    if len(high_mobility_indices) == 0:
        return 0.0, data.shape[1] / fs

    onset_index = high_mobility_indices[np.argmin(np.abs(high_mobility_indices - changepoint_index))]
    offset_index = high_mobility_indices[-1]

    onset = onset_index * step_size / fs
    offset = (offset_index + window_size) * step_size / fs

    return onset, offset

# Funktion zum Berechnen der Bandpower
def bandpower(data, sf, band, window_sec=4, relative=False):
    band = np.asarray(band)
    low, high = band

    freqs, psd = welch(data, sf, nperseg=window_sec*sf)

    idx_band = np.logical_and(freqs >= low, freqs <= high)
    bp = np.trapz(psd[idx_band], freqs[idx_band])

    if relative:
        bp /= np.trapz(psd, freqs)
    return bp

# Funktion zum Extrahieren von Features aus rohen EEG-Daten
def extract_features(data, fs):
    features = []
    for channel_data in data:
        mean = np.mean(channel_data)
        std = np.std(channel_data)
        line_length = np.sum(np.abs(np.diff(channel_data)))
        change_rate = np.sum(np.abs(np.diff(channel_data)))
        
        delta_power = bandpower(channel_data, fs, [0.5, 4])
        theta_power = bandpower(channel_data, fs, [4, 8])
        alpha_power = bandpower(channel_data, fs, [8, 13])
        beta_power = bandpower(channel_data, fs, [13, 30])
        
        features.extend([mean, std, line_length, change_rate, delta_power, theta_power, alpha_power, beta_power])
    max_feature_length = 152  # Ensure the same number of features as during training (19 channels * 8 features)
    if len(features) < max_feature_length:
        features = np.pad(features, (0, max_feature_length - len(features)), 'constant')
    return np.array(features)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make predictions on EEG test data")
    parser.add_argument("--test_dir", type=str, required=True, help="Directory containing the test EEG data")
    args = parser.parse_args()
    
    main(args.test_dir)