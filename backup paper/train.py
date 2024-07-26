import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, log_loss
from wettbewerb import EEGDataset
from bayes_opt import BayesianOptimization
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold, train_test_split
import pandas as pd
from scipy.signal import welch
import psutil
import time

# Funktion zum Berechnen der Signaländerungsrate
def calculate_change_rate(data):
    change_rate = np.sum(np.abs(np.diff(data)))
    return change_rate

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
        change_rate = calculate_change_rate(channel_data)
        
        delta_power = bandpower(channel_data, fs, [0.5, 4])
        theta_power = bandpower(channel_data, fs, [4, 8])
        alpha_power = bandpower(channel_data, fs, [8, 13])
        beta_power = bandpower(channel_data, fs, [13, 30])
        
        features.extend([mean, std, line_length, change_rate, delta_power, theta_power, alpha_power, beta_power])
    return np.array(features)

# Funktion zum Verarbeiten eines Batches
def process_batch(dataset, start_idx, end_idx):
    batch_features = []
    batch_labels = []
    for i in range(start_idx, end_idx):
        id, channels, data, fs, ref_system, label = dataset[i]
        features = extract_features(data, fs)
        batch_features.append(features)
        batch_labels.append(label[0])  # Only use the first element of the label
    return batch_features, batch_labels

# Funktion zur Extraktion von Features aus allen Daten in Batches
def extract_all_features(data_folder, batch_size):
    dataset = EEGDataset(data_folder)
    dataset_size = len(dataset)
    print(f"Gesamtzahl der Dateien im Datensatz: {dataset_size}")

    all_features = []
    all_labels = []
    num_batches = (dataset_size + batch_size - 1) // batch_size

    results = Parallel(n_jobs=-1, verbose=10)(delayed(process_batch)(dataset, start_idx, min(start_idx + batch_size, dataset_size))
                                             for start_idx in range(0, dataset_size, batch_size))

    for batch_idx, (batch_features, batch_labels) in enumerate(results):
        max_feature_length = max(len(f) for f in batch_features)
        corrected_batch_features = [
            np.pad(f, (0, max_feature_length - len(f)), 'constant') if len(f) < max_feature_length else f
            for f in batch_features
        ]
        all_features.append(np.array(corrected_batch_features))
        all_labels.extend(batch_labels)
        print(f"Batch {batch_idx + 1}/{num_batches} geladen: {len(batch_features)} Dateien")

    all_features = np.vstack(all_features)
    all_labels = np.array(all_labels).astype(int)
    print(f"Gesamtzahl der geladenen Dateien: {len(all_labels)}")

    if len(all_labels) != dataset_size:
        print(f"Warnung: Es wurden nicht alle Dateien geladen! Erwartet: {dataset_size}, Geladen: {len(all_labels)}")

    return all_features, all_labels

# Zusätzliche Überprüfung der geladenen IDs
def verify_loaded_files(data_folder):
    dataset = EEGDataset(data_folder)
    loaded_ids = [dataset[i][0] for i in range(len(dataset))]
    print(f"Anzahl der geladenen IDs: {len(loaded_ids)}")

    reference_file = os.path.join(data_folder, "REFERENCE.csv")
    reference_data = pd.read_csv(reference_file, header=None)
    reference_ids = reference_data[0].tolist()
    print(f"Anzahl der IDs in REFERENCE.csv: {len(reference_ids)}")

    missing_ids = set(reference_ids) - set(loaded_ids)
    extra_ids = set(loaded_ids) - set(reference_ids)

    if missing_ids:
        print(f"Fehlende IDs: {missing_ids}")
    if extra_ids:
        print(f"Zusätzliche IDs: {extra_ids}")

    all_files_correctly_loaded = len(missing_ids) == 0 and len(extra_ids) == 0
    print(f"Alle Dateien korrekt geladen: {all_files_correctly_loaded}")

    return all_files_correctly_loaded

# Optimierungsfunktion für RandomForestClassifier
def rf_cv(n_estimators, max_depth, min_samples_split, min_samples_leaf, bootstrap):
    n_estimators = int(n_estimators)
    max_depth = int(max_depth)
    min_samples_split = int(min_samples_split)
    min_samples_leaf = int(min_samples_leaf)
    bootstrap = bool(bootstrap)

    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    f1_scores = []

    for train_index, test_index in kf.split(features, labels):
        X_train, X_test = features[train_index], features[test_index]
        Y_train, Y_test = labels[train_index], labels[test_index]

        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                     min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                     bootstrap=bootstrap, random_state=42, n_jobs=-1)

        clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_test)
        
        f1_scores.append(f1_score(Y_test, Y_pred, average='weighted'))

    return np.mean(f1_scores)

# Optimierungsfunktion für KNeighborsClassifier
def knn_cv(n_neighbors, weights, algorithm):
    n_neighbors = int(n_neighbors)
    weights = 'distance' if weights > 0.5 else 'uniform'
    algorithm = 'auto' if algorithm < 0.33 else 'ball_tree' if algorithm < 0.66 else 'kd_tree'

    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    f1_scores = []

    for train_index, test_index in kf.split(features, labels):
        X_train, X_test = features[train_index], features[test_index]
        Y_train, Y_test = labels[train_index], labels[test_index]

        clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, n_jobs=-1)

        clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_test)

        f1_scores.append(f1_score(Y_test, Y_pred, average='weighted'))

    return np.mean(f1_scores)

# Laden der Trainingsdaten
data_folder = '/home/jupyter-wki_team_2/Silvan/training/training_70'
batch_size = 1000  # Erhöhte Anzahl der Samples pro Batch

# Überprüfen, ob alle Dateien geladen werden
all_files_loaded = verify_loaded_files(data_folder)
if not all_files_loaded:
    print("Nicht alle Dateien wurden korrekt geladen!")
else:
    start_time = time.time()
    process = psutil.Process(os.getpid())
    
    features, labels = extract_all_features(data_folder, batch_size)

    # Sicherstellen, dass die Labels als Integers für Multiclass vorliegen
    unique_labels = np.unique(labels)
    print(f"Einzigartige Labels: {unique_labels}")

    # Daten normalisieren
    scaler = StandardScaler().fit(features)
    features = scaler.transform(features)

    # Daten in Trainings- und Validierungsset aufteilen
    features_train, features_val, labels_train, labels_val = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Bayesian Optimization für RandomForestClassifier
    pbounds_rf = {
        'n_estimators': (50, 200),
        'max_depth': (5, 50),
        'min_samples_split': (2, 20),
        'min_samples_leaf': (1, 10),
        'bootstrap': (0, 1)
    }

    optimizer_rf = BayesianOptimization(f=rf_cv, pbounds=pbounds_rf, random_state=42, verbose=2)
    optimizer_rf.maximize(init_points=10, n_iter=30)

    # Beste Parameter für RandomForestClassifier extrahieren
    best_params_rf = optimizer_rf.max['params']
    best_params_rf['n_estimators'] = int(best_params_rf['n_estimators'])
    best_params_rf['max_depth'] = int(best_params_rf['max_depth'])
    best_params_rf['min_samples_split'] = int(best_params_rf['min_samples_split'])
    best_params_rf['min_samples_leaf'] = int(best_params_rf['min_samples_leaf'])
    best_params_rf['bootstrap'] = bool(best_params_rf['bootstrap'])

    # Bayesian Optimization für KNeighborsClassifier
    pbounds_knn = {
        'n_neighbors': (3, 15),
        'weights': (0, 1),
        'algorithm': (0, 1)
    }

    optimizer_knn = BayesianOptimization(f=knn_cv, pbounds=pbounds_knn, random_state=42, verbose=2)
    optimizer_knn.maximize(init_points=10, n_iter=30)

    # Beste Parameter für KNeighborsClassifier extrahieren
    best_params_knn = optimizer_knn.max['params']
    best_params_knn['n_neighbors'] = int(best_params_knn['n_neighbors'])
    best_params_knn['weights'] = 'distance' if best_params_knn['weights'] > 0.5 else 'uniform'
    best_params_knn['algorithm'] = 'auto' if best_params_knn['algorithm'] < 0.33 else 'ball_tree' if best_params_knn['algorithm'] < 0.66 else 'kd_tree'

    # Trainieren der finalen Modelle mit den besten Parametern
    final_rf = RandomForestClassifier(**best_params_rf, random_state=42)
    final_rf.fit(features_train, labels_train)

    final_knn = KNeighborsClassifier(n_neighbors=best_params_knn['n_neighbors'], weights=best_params_knn['weights'], algorithm=best_params_knn['algorithm'], n_jobs=-1)
    final_knn.fit(features_train, labels_train)

    # Ensemble mit VotingClassifier
    ensemble_model = VotingClassifier(estimators=[('rf', final_rf), ('knn', final_knn)], voting='soft', n_jobs=-1)
    ensemble_model.fit(features_train, labels_train)

    # Performance auf dem Validierungsset evaluieren
    val_predictions = ensemble_model.predict(features_val)
    val_prob_predictions = ensemble_model.predict_proba(features_val)

    val_f1_score = f1_score(labels_val, val_predictions, average='weighted')
    val_precision = precision_score(labels_val, val_predictions, average='weighted')
    val_recall = recall_score(labels_val, val_predictions, average='weighted')
    val_accuracy = accuracy_score(labels_val, val_predictions)
    val_loss = log_loss(labels_val, val_prob_predictions)

    print(f"Validation F1 Score: {val_f1_score:.4f}")
    print(f"Validation Precision: {val_precision:.4f}")
    print(f"Validation Recall: {val_recall:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation Log Loss: {val_loss:.4f}")

    # Feature Importance für RandomForestClassifier extrahieren und anzeigen
    feature_importances = final_rf.feature_importances_
    num_features_per_channel = 8  # Anzahl der Features pro Kanal
    num_channels = len(feature_importances) // num_features_per_channel

    average_importances = np.zeros(num_features_per_channel)
    for i in range(num_channels):
        start_idx = i * num_features_per_channel
        end_idx = start_idx + num_features_per_channel
        average_importances += feature_importances[start_idx:end_idx]
    average_importances /= num_channels

    # Normalisieren der Feature Importances
    total_importance = np.sum(average_importances)
    normalized_importances = average_importances / total_importance

    feature_names = ['mean', 'std', 'line_length', 'change_rate', 'delta_power', 'theta_power', 'alpha_power', 'beta_power']
    for i, importance in enumerate(normalized_importances):
        print(f"Feature: {feature_names[i]}, Normalized Average Importance: {importance}")

    # Speichern des Ensemble-Modells und des Scalers
    model_path = os.path.join(os.path.dirname(__file__), 'ensemble_model.joblib')
    scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.joblib')
    joblib.dump(ensemble_model, model_path)
    joblib.dump(scaler, scaler_path)
    print("Ensemble-Modell und Scaler gespeichert.")
    
    end_time = time.time()
    memory_usage = process.memory_info().rss / (1024 ** 2)  # In MB
    cpu_usage = psutil.cpu_percent(interval=None)
    
    print(f"RAM-Verbrauch: {memory_usage:.2f} MB")
    print(f"CPU-Auslastung: {cpu_usage:.2f}%")
    print(f"Laufzeit: {end_time - start_time:.2f} Sekunden")
