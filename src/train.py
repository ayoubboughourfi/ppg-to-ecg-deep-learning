import os
import numpy as np
import tensorflow as tf
from architecture import build_model_1_ppg_only, build_model_2_multimodal
from dataset import load_and_preprocess_data
from evaluate import generate_comparison_image, plot_training_history

def main():
    print("--- SCHRITT 1: Datensatz laden ---")
    X_ppg, X_noisy_ecg, Y_ecg = load_and_preprocess_data(data_dir="data", window_size=300)
    
    num_samples = X_ppg.shape[0]
    if num_samples == 0:
        print("[FEHLER] Keine Daten geladen.")
        return
        
    split = int(0.8 * num_samples)
    
    X_train_m1, X_test_m1 = X_ppg[:split], X_ppg[split:]
    X_multimodal = np.concatenate([X_ppg, X_noisy_ecg], axis=-1)
    X_train_m2, X_test_m2 = X_multimodal[:split], X_multimodal[split:]
    y_train, y_test = Y_ecg[:split], Y_ecg[split:]
    
    print(f"Training Set: {split} Samples | Test Set: {num_samples - split} Samples")
    
    # Optimales Learning Rate von Optuna
    optimal_lr = 0.0011748264222019513
    
    print("\\n--- SCHRITT 2: Modell 1 trainieren (50 Epochen) ---")
    model_1 = build_model_1_ppg_only(window_size=300)
    model_1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=optimal_lr), loss="mse", metrics=["mae"])
    history_m1 = model_1.fit(X_train_m1, y_train, epochs=50, batch_size=32, validation_split=0.2)
    
    print("\\n--- SCHRITT 3: Modell 2 trainieren (50 Epochen) ---")
    model_2 = build_model_2_multimodal(window_size=300)
    model_2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=optimal_lr), loss="mse", metrics=["mae"])
    history_m2 = model_2.fit(X_train_m2, y_train, epochs=50, batch_size=32, validation_split=0.2)
    
    print("\\n--- SCHRITT 4: Evaluation & Generierung der Auswertungs-Plots ---")
    os.makedirs("outputs", exist_ok=True)
    
    plot_training_history(history_m1, history_m2, filename="outputs/training_history.png")
    
    test_idx = 0
    pred_m1 = model_1.predict(X_test_m1[test_idx : test_idx + 1])[0]
    pred_m2 = model_2.predict(X_test_m2[test_idx : test_idx + 1])[0]
    true_sample = y_test[test_idx]
    
    generate_comparison_image(
        true_ecg=true_sample.flatten(),
        pred_m1=pred_m1.flatten(),
        pred_m2=pred_m2.flatten(),
        sample_idx=test_idx,
        filename="outputs/final_ecg_model_comparison.png"
    )
    
    print("[FERTIG] Training beendet! Ergebnisse wurden im Ordner 'outputs/' gespeichert.")

if __name__ == "__main__":
    main()