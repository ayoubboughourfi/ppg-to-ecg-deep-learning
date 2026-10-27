# Deep Learning-Based Reconstruction of Electrocardiogram (ECG) Signals from Photoplethysmography (PPG) Data

## Abstract

This repository contains a full end-to-end deep learning framework designed to reconstruct Electrocardiogram (ECG) signals from Photoplethysmography (PPG) time-series data. The project evaluates two 1D Convolutional Neural Network (CNN) Autoencoder architectures: an **unimodal network** (PPG to ECG) and a **multimodal network** (PPG combined with noisy ECG context to reconstruct clean ECG).

Through systematic experimentation—incorporating **Optuna hyperparameter optimization**, **z-score signal normalization**, and **75% sliding window data augmentation**—the framework achieves near-perfect morphological and phase reconstruction for multimodal inputs, resolving data starvation and alignment challenges inherent to biological signal processing.

---

## 1. Project Overview & System Architecture

The primary goal is converting optical pulse wave signals (PPG) into electrophysiological activity maps (ECG). Reconstructing the precise morphology of ECG signals—specifically the R-peak location, QRS complex amplitude, and T-wave intervals—is challenging due to temporal lag and non-linear arterial compliance.

```
       +-------------------+       +-----------------------+       +-------------------+
       |  PPG Signal Input | ----> | 1D CNN Encoder-Decoder| ----> | Reconstructed ECG |
       +-------------------+       +-----------------------+       +-------------------+
                                               ^
                                               | (Optional Context Channel)
                                   +-----------------------+
                                   | Noisy ECG Context     |
                                   +-----------------------+
```

---

## 2. Experimental Progression: From Initial Baseline to Optimal Performance

### Phase 1: Initial Baseline & Severe Underfitting (5 Epochs)

In the initial baseline run trained for only 5 epochs on raw, unnormalized samples (~10,000 non-overlapping windows), both models produced near-flat mean predictions. The MSE loss function penalized large errors near peaks, causing the network weights to collapse toward predicting the average baseline amplitude (~0 to 10).

![Phase 1: Initial 5-Epoch Flat Output](final_ecg_model_comparison.png)
*Figure 1: Initial baseline model output showing flat response across all 300 timesteps.*

---

## 2. Experimental Progression: From Initial Baseline to Optimal Performance

### Phase 1: Initial Baseline & Severe Underfitting (5 Epochs)

In the initial baseline run trained for only 5 epochs on raw, unnormalized samples (~10,000 non-overlapping windows), both models produced near-flat mean predictions. The MSE loss function penalized large errors near peaks, causing the network weights to collapse toward predicting the average baseline amplitude (~0 to 10).

![Phase 1: Initial 5-Epoch Flat Output](outputs/final_ecg_model_comparison.png)
*Figure 1: Initial baseline model output showing flat response across all 300 timesteps.*

---

### Phase 2: Extended Training & Data Starvation (50 Epochs, Unaugmented)

Increasing training to 50 epochs allowed the networks to begin learning active waveform dynamics. However, because the dataset remained restricted to ~10,000 non-overlapping samples, the models suffered from **data starvation**:
1. **Model 1 (Unimodal):** Produced sharp spikes but exhibited phase shifts (lag relative to true R-peaks).
2. **Model 2 (Multimodal):** Followed signal activity but suffered from severe amplitude variance.

![Phase 2: Reconstruction Comparison with Phase Lags](outputs/final_ecg_model_comparison_v1.png)
*Figure 2: 50-epoch reconstruction attempt on unaugmented data exhibiting phase and amplitude misalignments.*

Validation metrics during this phase showed severe instability (spikes in validation loss/MAE), confirming that the small validation batch variations caused overfitting and poor generalization.

![Phase 2: Validation Spikes and Loss Instability](outputs/training_history.png)
*Figure 3: Training history on unaugmented data showing volatile validation spikes.*

---

### Phase 3: Hyperparameter Optimization & Refinement

To address architecture parameters, **Optuna** was executed across 10 trial iterations. The optimization targeted learning rate and layer capacity:

- **Optimal Learning Rate:** `0.0011748`
- **Filter Configurations:** `filters_1 = 64`, `filters_2 = 128`

Applying these hyperparameter adjustments improved peak definition, but the models still required significantly more spatial context to lock onto the correct phase.

![Phase 3: Hyperparameter-Tuned Signal Output](outputs/final_ecg_model_comparison_v2.png)
*Figure 4: Tuned model output displaying enhanced peak definition prior to window augmentation.*

---

### Phase 4: Final Pipeline & 75% Sliding Window Augmentation (Optimal State)

To resolve data starvation, a **sliding window strategy with 75% overlap** (`step_size = window_size // 4 = 75`) was implemented. This increased the sample count from ~10,600 to **>40,000 samples**.

Combining this augmented dataset with Z-score standardization produced convergence:
- **Model 2 (Multimodal):** Achieved near-perfect alignment with the true ECG curve, tracking the R-peaks, QRS complexes, P-waves, and T-waves across all timesteps.
- **Model 1 (Unimodal):** Retained baseline periodicity while illustrating the inherent limits of optical-only inputs.

![Phase 4: Final Reconstruction Comparison](outputs/final_ecg_model_comparison_v3.png)
*Figure 5: Final output comparing Model 1 and Model 2 against the ground truth ECG signal.*

The final training logs confirm smooth convergence across both MSE Loss and MAE, eliminating validation spikes.

![Phase 4: Final Training Loss and MAE Metrics](outputs/training_history_v1.png)
*Figure 6: Smooth loss and MAE convergence curves achieved with 75% overlapping windows.*

---

## 3. Quantitative Results & Summary

| Model Variant | Training Loss (MSE) | Validation Loss (MSE) | Test MAE | Reconstruction Quality |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 1 (5 Epochen)** | > 600.0 | > 650.0 | ~ 15.2 | Flat / Unusable |
| **Phase 2 (50 Epochen, Raw)** | ~ 280.0 | ~ 250.0 (Volatile) | ~ 7.5 | Phase-shifted |
| **Phase 4 (Final + 75% Overlap)** | **< 0.01** | **< 0.01** | **< 0.05** | **Near-Perfect Alignment** |

---

## 4. Repository Structure

```text
ppg-to-ecg-deep-learning/
│
├── data/                            # Raw BIDMC Dataset CSVs
├── src/
│   ├── architecture.py              # CNN Autoencoder architectures (Unimodal & Multimodal)
│   ├── dataset.py                   # Data preprocessing & 75% overlapping sliding window
│   ├── evaluate.py                  # Plotting utilities for comparison & training history
│   ├── optimize.py                  # Optuna hyperparameter optimization script
│   └── train.py                     # Execution pipeline (50 Epochs)
├── outputs/                         # Exported loss plots and evaluation figures
└── README.md                        # Academic documentation
```

---

## 5. Execution Instructions

To execute the entire pipeline from data ingestion to model training and plot generation:

```bash
# Clone repository and navigate to working directory
cd ppg-to-ecg-deep-learning

# Run Optuna hyperparameter search (Optional)
python src/optimize.py

# Execute main training pipeline
python src/train.py
```