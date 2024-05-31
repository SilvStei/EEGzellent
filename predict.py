import numpy as np
import os
from typing import List, Dict, Any
from wettbewerb import get_3montages
import mne
from scipy import signal as sig
import ruptures as rpt
from joblib import load
from tqdm import tqdm
import json

# Funktion zur Berechnung von Features
def calculate_features(signal, fs):
    features = {}
    signal_notch = mne.filter.notch_filter(x=signal, Fs=fs, freqs=np.array([50., 100.]), n_jobs=2, verbose=False)
    signal_filter = mne.filter.filter_data(data=signal_notch, sfreq=fs, l_freq=0.5, h_freq=70.0, n_jobs=2, verbose=False)
    features['energy'] = np.sum(signal_filter ** 2)
    features['kurtosis'] = np.mean((signal_filter - np.mean(signal_filter))**4) / np.var(signal_filter)**2
    features['line_length'] = np.sum(np.abs(np.diff(signal_filter)))
    features['entropy'] = -np.sum(signal_filter * np.log(np.abs(signal_filter) + 1e-10))
    features['skewness'] = np.mean((signal_filter - np.mean(signal_filter))**3) / np.var(signal_filter)**1.5
    features['max'] = np.max(signal_filter)
    features['std'] = np.std(signal_filter)
    features['min'] = np.min(signal_filter)
    return features

def predict_labels(channels: List[str], data: np.ndarray, fs: float, reference_system: str, model_name: str) -> Dict[str, Any]:
    # Überprüfe, ob das korrekte Modell geladen wird
    if model_name == 'model.json':
        model_name = 'model.joblib'  # Setze auf den korrekten Modellnamen um, wenn model.json übergeben wird
    
    model_path = os.path.join('./', model_name)
    try:
        clf = load(model_path)
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return {}

    scaler_path = os.path.join('./', 'scaler.pkl')
    try:
        scaler = load(scaler_path)
        print(f"Scaler loaded from {scaler_path}")
    except Exception as e:
        print(f"Error loading scaler from {scaler_path}: {e}")
        return {}

    _montage, _montage_data, _is_missing = get_3montages(channels, data)
    features_list = [calculate_features(_montage_data[j], fs) for j in range(len(_montage))]

    if not features_list:
        print("No features extracted.")
        return {}

    X = np.array([list(f.values()) for f in features_list])
    if not X.size:
        print("No features to predict.")
        return {}

    X = scaler.transform(X)
    top_n = 10
    with open('best_params.json', 'r') as f:
        best_params = json.load(f)
    importances = clf.named_estimators_['rf'].feature_importances_
    indices = np.argsort(importances)[::-1]
    top_indices = indices[:top_n]
    X_top = X[:, top_indices]
    
    try:
        seizure_present = clf.predict(X_top)[0]
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {}

    prediction = {
        "seizure_present": seizure_present,
        "seizure_confidence": 0.5,
        "onset": 0.0,
        "onset_confidence": 0.99,
        "offset": 999999,
        "offset_confidence": 0
    }

    return prediction
