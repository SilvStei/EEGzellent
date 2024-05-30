import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import gc
from wettbewerb import EEGDataset, get_3montages
import mne
from scipy import signal as sig
import ruptures as rpt
import json
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib  # Joblib zum Speichern und Laden des Modells
from joblib import Parallel, delayed  # Joblib zur Parallelisierung
from bayes_opt import BayesianOptimization

# Setze das Trainingsdatenverzeichnis
training_folder = "/home/jupyter-wki_team_2/Silvan/training/training"

# Funktion zur Berechnung von Features
def calculate_features(signal, fs):
    features = {}
    
    # Notch-Filter um Netzfrequenz zu dämpfen
    signal_notch = mne.filter.notch_filter(x=signal, Fs=fs, freqs=np.array([50., 100.]), n_jobs=2, verbose=False)
    # Bandpassfilter zwischen 0.5Hz und 70Hz um Rauschen aus dem Signal zu filtern
    signal_filter = mne.filter.filter_data(data=signal_notch, sfreq=fs, l_freq=0.5, h_freq=70.0, filter_length='auto', n_jobs=2, verbose=False)
    
    # Berechne statistische Features
    features['energy'] = np.sum(signal_filter ** 2)
    features['kurtosis'] = np.mean((signal_filter - np.mean(signal_filter))**4) / np.var(signal_filter)**2
    features['line_length'] = np.sum(np.abs(np.diff(signal_filter)))
    features['entropy'] = -np.sum(signal_filter * np.log(np.abs(signal_filter) + 1e-10))
    features['skewness'] = np.mean((signal_filter - np.mean(signal_filter))**3) / np.var(signal_filter)**1.5
    features['max'] = np.max(signal_filter)
    features['std'] = np.std(signal_filter)
    features['min'] = np.min(signal_filter)
    
    return features

# Funktion zum Verarbeiten eines Batches
def process_batch(dataset, indices):
    feature_list = []
    label_list = []
    for i in indices:
        _id, channels, data, _fs, reference_system, eeg_label = dataset[i]
        _montage, _montage_data, _is_missing = get_3montages(channels, data)
        for signal in _montage_data:
            features = calculate_features(signal, _fs)
            feature_list.append(features)
            label_list.append(eeg_label[0])
    
    return feature_list, label_list

# Funktion zur Extraktion von Features aus allen Daten in Batches
def extract_features_and_labels(dataset, indices):
    all_features = []
    all_labels = []
    
    results = Parallel(n_jobs=-1, verbose=10)(delayed(process_batch)(dataset, indices[start_idx:end_idx])
                                             for start_idx in range(0, len(indices), batch_size)
                                             for end_idx in [min(start_idx + batch_size, len(indices))])
    
    for features, labels in results:
        all_features.extend(features)
        all_labels.extend(labels)
        gc.collect()  # Manuelles Aufrufen des Garbage Collectors
    
    return all_features, all_labels

# Funktion zur Durchführung der Bayesian Optimization
def rf_cv(n_estimators, max_depth, min_samples_split, min_samples_leaf, bootstrap):
    n_estimators = int(n_estimators)
    max_depth = int(max_depth)
    min_samples_split = int(min_samples_split)
    min_samples_leaf = int(min_samples_leaf)
    bootstrap = bool(bootstrap)

    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    f1_scores = []

    for train_index, test_index in kf.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        clf = RandomForestClassifier(n_estimators=n_estimators,
                                     max_depth=max_depth,
                                     min_samples_split=min_samples_split,
                                     min_samples_leaf=min_samples_leaf,
                                     bootstrap=bootstrap,
                                     random_state=42,
                                     n_jobs=-1)

        clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_test)

        f1 = f1_score(Y_test, Y_pred)
        f1_scores.append(f1)

    return np.mean(f1_scores)

# Funktion zum Durchführen der Kreuzvalidierung und Modellierung
def k_fold_cross_validation(dataset, batch_size=100):
    global X, Y
    labels = np.array([label[0] for _, _, _, _, _, label in dataset])
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    f1_scores = []
    mae_scores = []
    best_f1 = 0  # Variable zum Speichern der besten F1-Score

    all_features, all_labels = extract_features_and_labels(dataset, list(range(len(dataset))))
    X = np.array([list(f.values()) for f in all_features])
    Y = np.array(all_labels)

    # Standardisierung der Daten
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Definiere den Parameterbereich für die Bayesian Optimization
    pbounds = {
        'n_estimators': (50, 200),
        'max_depth': (10, 50),
        'min_samples_split': (2, 10),
        'min_samples_leaf': (1, 4),
        'bootstrap': (0, 1)
    }

    rf_bo = BayesianOptimization(
        f=rf_cv,
        pbounds=pbounds,
        random_state=42,
    )

    # Durchführung der Bayesian Optimization
    rf_bo.maximize(n_iter=50, init_points=10)

    # Extrahieren der besten Hyperparameter
    best_params = rf_bo.max['params']
    best_params['n_estimators'] = int(best_params['n_estimators'])
    best_params['max_depth'] = int(best_params['max_depth'])
    best_params['min_samples_split'] = int(best_params['min_samples_split'])
    best_params['min_samples_leaf'] = int(best_params['min_samples_leaf'])
    best_params['bootstrap'] = bool(best_params['bootstrap'])

    # Trainieren des besten Modells auf den gesamten Daten
    best_rf = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    best_rf.fit(X, Y)

    # Feature Importance bestimmen
    importances = best_rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Auswahl der wichtigsten Features (zum Beispiel Top 10)
    top_n = 10
    top_indices = indices[:top_n]
    X_top = X[:, top_indices]

    # AdaBoost Modellierung
    ada_boost = AdaBoostClassifier(base_estimator=best_rf, n_estimators=50, random_state=42)

    # Ensemble Modell mit VotingClassifier
    ensemble = VotingClassifier(estimators=[
        ('rf', best_rf),
        ('ada', ada_boost)
    ], voting='soft')

    # Stratified K-Fold Cross-Validation für das Ensemble Modell
    for train_index, test_index in kf.split(X_top, Y):
        X_train, X_test = X_top[train_index], X_top[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        ensemble.fit(X_train, Y_train)
        Y_pred = ensemble.predict(X_test)

        f1 = f1_score(Y_test, Y_pred)
        mae = mean_absolute_error(Y_test, Y_pred)

        f1_scores.append(f1)
        mae_scores.append(mae)

        if f1 > best_f1:
            best_f1 = f1
            best_model = ensemble

    print(f"Durchschnittliche F1-Score: {np.mean(f1_scores)}")
    print(f"Durchschnittlicher MAE: {np.mean(mae_scores)}")

    joblib.dump(best_model, 'model.joblib')
    print('Bestes Seizure Detektionsmodell wurde gespeichert!')

    with open('best_params.json', 'w') as f:
        json.dump(best_params, f)
    print('Beste Hyperparameter wurden gespeichert!')

    with open('scaler.pkl', 'wb') as f:
        joblib.dump(scaler, f)
    print('Scaler wurde gespeichert!')

# Setze Batchgröße
batch_size = 100

if __name__ == '__main__':
    dataset = EEGDataset(training_folder)
    k_fold_cross_validation(dataset, batch_size=batch_size)
