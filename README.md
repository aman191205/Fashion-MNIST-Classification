# Deep Learning Assignment 01 — Fashion-MNIST Neural Networks
 
**Authors:** 23F0834, 23F0738
 
## What this project is
 
This project implements and studies a fully-connected neural network classifier on the **Fashion-MNIST** dataset (28×28 grayscale clothing images, 10 classes), moving from a from-scratch NumPy implementation up through a tuned, regularised Keras model. It is organised into seven parts:
 
1. **Data prep** — load `fashiontrain.csv` / `fashiontest.csv`, normalise pixel values to [0, 1], and split into train/validation/test sets (stratified 80/20 split of the training data).
2. **Backpropagation from scratch** — a 2-layer network (784 → 64 → 10) built and trained manually in NumPy, with gradients verified against PyTorch's autograd (differences of 0.0000000000 to 10 decimal places).
3. **Activation function study** — compares sigmoid, tanh, ReLU, and Leaky ReLU on the same architecture, examining vanishing gradients and dead units.
4. **Loss function study** — compares cross-entropy vs. MSE loss.
5. **Optimiser comparison** — SGD, SGD+momentum, RMSProp, and Adam, both at a shared learning rate and with per-optimiser tuned learning rates.
6. **Forced overfitting & regularisation** — a large network is overfit on 2,000 samples, then L2, L1, dropout, batch normalisation, early stopping, data augmentation, and extra training data are compared for how much they close the train/validation gap.
7. **Hyperparameter tuning** — a random search over learning rate, hidden width, and dropout rate, evaluated with 5-fold stratified cross-validation, with the best configuration retrained on the full training set and scored on the held-out test set.
## Key results
 
| Stage | Metric | Value |
|---|---|---|
| From-scratch NumPy NN | Validation accuracy | 0.0719 (untrained, sanity check only) |
| Baseline (cross-entropy) | Test accuracy | 0.8202 |
| Baseline (MSE loss) | Test accuracy | 0.7446 |
| Best regularisation (efficiency) | Dropout 0.2 | Gap 0.162 → 0.125 (train acc 1.00 → 0.966) |
| Best regularisation (smallest gap) | Dropout 0.6 | Gap 0.014 (train acc 1.00 → 0.834) |
| Tuned model (5-fold CV) | Mean CV accuracy | 0.8836 |
| **Final tuned model** | **Test accuracy** | **0.9017** |
| Improvement over Part 2 baseline | — | +8.15 percentage points |
 
## How it was done (briefly)
 
- Data was normalised and split before anything else, so every later stage trains/evaluates on the same splits.
- The backprop derivation was validated numerically against PyTorch rather than trusted by inspection, to catch sign/shape errors early.
- Every model-comparison stage (activations, losses, optimisers, regularisers, hyperparameters) fixes a random seed (`SEED = 42`) and holds everything except the variable under test constant, so the comparisons isolate one factor at a time.
- Regularisation methods were ranked by the *trade-off* between generalisation-gap reduction and training-accuracy cost, not just by the smallest gap alone.
- The final model's hyperparameters were chosen purely from cross-validation on the training split — the validation set and test set were never used for selection, only for final reporting.
## Requirements
 
- Python 3.9+
- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `torch`
- `tensorflow` (Keras)
Install with:
 
```bash
pip install pandas numpy matplotlib scikit-learn torch tensorflow
```
 
## Data
 
Place the following two files in the same directory as the notebook:
 
- `fashiontrain.csv`
- `fashiontest.csv`
Each is a Fashion-MNIST CSV with a `label` column followed by 784 `pixelN` columns (0–255 grayscale values).
 
## How to reproduce the results
 
1. Clone/download this repository and ensure `fashiontrain.csv` and `fashiontest.csv` are present alongside the notebook.
2. Install the dependencies listed above.
3. Open the notebook (`.ipynb`) in Jupyter or Google Colab.
4. Run all cells **top to bottom, in order** — later parts depend on variables and helper functions (e.g. `one_hot_encode`, `X_train`, `XValNp`, `SEED`) defined earlier.
5. All random operations are seeded (`SEED = 42` / `np.random.seed(42)`), so re-running should reproduce results very close to those reported above. Minor differences (±0.1–0.2%) can occur between runs due to hardware/backend non-determinism in TensorFlow.
6. Expect a full run to take several minutes, dominated by Part 6 (regularisation sweep) and Part 7 (5-fold cross-validation over 12 hyperparameter configurations).
## Notes
 
- Part 3 also includes a short regression side-study on the UCI Wine Quality dataset (fetched directly via URL) to test the same MLP architecture on a regression task.
- The final model in Part 7 is retrained on the combined train + validation data before being scored once on the untouched test set.
 
