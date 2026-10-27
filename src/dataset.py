import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(data_dir="data", window_size=300):
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    print(f"[DATASET] {len(csv_files)} Signal-Dateien gefunden. Starte Verarbeitung...")
    
    all_ppg = []
    all_noisy_ecg = []
    all_clean_ecg = []
    
    for file_path in csv_files:
        df = pd.read_csv(file_path)
        cols = df.columns.tolist()
        
        # Spalten für PPG und ECG identifizieren
        ppg_col = next((c for c in cols if 'PLETH' in c or 'PPG' in c), None)
        ecg_col = next((c for c in cols if 'II' in c or 'ECG' in c), None)
        
        if ppg_col is None or ecg_col is None:
            continue
            
        ppg_signal = df[ppg_col].values
        ecg_signal = df[ecg_col].values
        
        # Z-Score Normalisierung
        ppg_scaler = StandardScaler()
        ecg_scaler = StandardScaler()
        
        ppg_signal = ppg_scaler.fit_transform(ppg_signal.reshape(-1, 1)).flatten()
        ecg_signal = ecg_scaler.fit_transform(ecg_signal.reshape(-1, 1)).flatten()
        
        # 75% Überlappung (Schrittweite = 75 bei Fenstergröße 300)
        step_size = window_size // 4 
        
        for i in range(0, len(ppg_signal) - window_size, step_size):
            window_ppg = ppg_signal[i : i + window_size]
            window_ecg = ecg_signal[i : i + window_size]
            
            # Synthetischer verrauschter ECG-Kontextkanal für Modell 2
            noise = np.random.normal(0, 0.1, window_size)
            window_noisy_ecg = window_ecg + noise
            
            if len(window_ppg) == window_size:
                all_ppg.append(window_ppg)
                all_noisy_ecg.append(window_noisy_ecg)
                all_clean_ecg.append(window_ecg)
                
    X_ppg = np.array(all_ppg)[..., np.newaxis]
    X_noisy_ecg = np.array(all_noisy_ecg)[..., np.newaxis]
    Y_ecg = np.array(all_clean_ecg)[..., np.newaxis]
    
    print(f"[DATASET] Erfolgreich {X_ppg.shape[0]} Samples mit 75% Überlappung erzeugt!")
    return X_ppg, X_noisy_ecg, Y_ecg