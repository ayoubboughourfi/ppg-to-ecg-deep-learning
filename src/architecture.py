import tensorflow as tf
from tensorflow.keras import layers, models

def build_model_1_ppg_only(window_size=300):
    """Unimodales Modell: PPG -> Clean ECG"""
    inputs = layers.Input(shape=(window_size, 1), name="ppg_input")
    
    # Encoder (64 -> 128 Filter basierend auf Optuna Best-Values)
    x = layers.Conv1D(64, kernel_size=5, activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, kernel_size=5, activation="relu", padding="same")(x)
    
    # Decoder / Reconstruction
    x = layers.Conv1D(128, kernel_size=5, activation="relu", padding="same")(x)
    x = layers.Conv1D(64, kernel_size=5, activation="relu", padding="same")(x)
    
    outputs = layers.Conv1D(1, kernel_size=5, activation="linear", padding="same", name="ecg_output")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="Model_1_PPG_Only")
    return model

def build_model_2_multimodal(window_size=300):
    """Multimodales Modell: [PPG, Noisy ECG Context] -> Clean ECG"""
    inputs = layers.Input(shape=(window_size, 2), name="multimodal_input")
    
    # Encoder with multi-channel features
    x = layers.Conv1D(64, kernel_size=5, activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, kernel_size=5, activation="relu", padding="same")(x)
    
    # Decoder / Reconstruction
    x = layers.Conv1D(128, kernel_size=5, activation="relu", padding="same")(x)
    x = layers.Conv1D(64, kernel_size=5, activation="relu", padding="same")(x)
    
    outputs = layers.Conv1D(1, kernel_size=5, activation="linear", padding="same", name="clean_ecg_output")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="Model_2_MultiModal")
    return model