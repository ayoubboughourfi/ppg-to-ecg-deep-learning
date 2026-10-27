import matplotlib.pyplot as plt
import numpy as np

def generate_comparison_image(true_ecg, pred_m1, pred_m2, sample_idx=0, filename="outputs/comparison.png"):
    plt.figure(figsize=(12, 6))
    plt.plot(true_ecg, label="True ECG", color="black", linewidth=2)
    plt.plot(pred_m1, label="Model 1 (PPG -> ECG)", color="blue", linestyle="--")
    plt.plot(pred_m2, label="Model 2 (Multi-Modal -> Clean ECG)", color="red", linestyle="-.")
    plt.title(f"ECG Reconstruction Comparison (Sample {sample_idx})")
    plt.xlabel("Time Steps")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"[EVALUATE] Vergleichsbild gespeichert unter: {filename}")

def plot_training_history(history_m1, history_m2, filename="outputs/training_history.png"):
    plt.figure(figsize=(14, 5))
    
    # MSE Loss
    plt.subplot(1, 2, 1)
    plt.plot(history_m1.history['loss'], label='Model 1 Train Loss', color='blue')
    plt.plot(history_m1.history['val_loss'], label='Model 1 Val Loss', color='blue', linestyle='--')
    plt.plot(history_m2.history['loss'], label='Model 2 Train Loss', color='red')
    plt.plot(history_m2.history['val_loss'], label='Model 2 Val Loss', color='red', linestyle='--')
    plt.title("Model Loss (MSE) over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(True)
    
    # MAE
    plt.subplot(1, 2, 2)
    plt.plot(history_m1.history['mae'], label='Model 1 Train MAE', color='blue')
    plt.plot(history_m1.history['val_mae'], label='Model 1 Val MAE', color='blue', linestyle='--')
    plt.plot(history_m2.history['mae'], label='Model 2 Train MAE', color='red')
    plt.plot(history_m2.history['val_mae'], label='Model 2 Val MAE', color='red', linestyle='--')
    plt.title("Model MAE over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Absolute Error (MAE)")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"[EVALUATE] Trainingshistorie gespeichert unter: {filename}")