import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import optuna
from dataset import load_and_preprocess_data

def objective(trial):
    # Parameter-Suchraum
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    filters_1 = trial.suggest_categorical("filters_1", [16, 32, 64])
    filters_2 = trial.suggest_categorical("filters_2", [32, 64, 128])
    
    X_ppg, _, Y_ecg = load_and_preprocess_data(data_dir="data", window_size=300)
    split = int(0.8 * X_ppg.shape[0])
    X_train, X_val = X_ppg[:split], X_ppg[split:]
    y_train, y_val = Y_ecg[:split], Y_ecg[split:]
    
    inputs = layers.Input(shape=(300, 1), name="ppg_input")
    x = layers.Conv1D(filters_1, kernel_size=5, activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(filters_2, kernel_size=5, activation="relu", padding="same")(x)
    x = layers.Conv1D(filters_2, kernel_size=5, activation="relu", padding="same")(x)
    x = layers.Conv1D(filters_1, kernel_size=5, activation="relu", padding="same")(x)
    outputs = layers.Conv1D(1, kernel_size=5, activation="linear", padding="same", name="ecg_output")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse", metrics=["mae"])
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=5,
        batch_size=32,
        verbose=0
    )
    
    return min(history.history["val_loss"])

def main():
    print("--- STARTE OPTUNA HYPERPARAMETER-OPTIMIERUNG ---")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)
    
    print("\\n[ERGEBNISSE] Beste Hyperparameter gefunden:")
    print(study.best_params)
    print(f"Bester Validation-Loss: {study.best_value:.4f}")

if __name__ == "__main__":
    main()