# MLP From Scratch - NumPy Neural Network on MNIST & Fashion-MNIST

A fully-connected feedforward neural network (MLP) implemented **entirely from scratch in NumPy**; no PyTorch, TensorFlow, or Keras for the model itself. Forward passes, backpropagation, and every optimizer update rule are hand-derived and hand-coded. Keras is used only to load the raw MNIST & Fashion-MNIST datasets.

**📊 [Interactive W&B report](https://forge.coreweave.com/wandb/prasid-indian-institute-of-technology-madras/assignment_1/reports/DA6401-Assignment-1-PH21B007-PRASID--VmlldzoxNjEyODA5Ng?accessToken=7lf6abidol3880zy7aiflc38domgwf0gtrwlsqz0fhboc9dumm1bdqjfn0fs1042)**

---

## Highlights

- **Zero-framework model implementation** : backprop, weight updates, and 4 optimizers written manually.
- **Bayesian hyperparameter sweep** via Weights & Biases over a 10-dimensional search space (architecture, optimizer, regularization, init scheme).
- **Diagnostic tooling built in** : dead-ReLU-neuron tracking, gradient-norm logging, and a "most confident failures" analysis that surfaces the model's worst high-confidence mistakes.
- Supports both **MNIST** and **Fashion-MNIST** out of the box via a single CLI flag.

---

## Repository Structure

```
MLP-from-scratch/
├── train.py               # Main training script with CLI argument parsing
├── sweep.py                # W&B Bayesian hyperparameter sweep setup
├── inference.py             # Evaluation on test set + confusion matrix + failure analysis
├── best_config.json         # Best hyperparameter configuration found via sweep
├── best_model.npy           # Saved weights of the best-trained model
├── __init__.py
├── ann/
│   ├── activations.py       # ReLU, Sigmoid, Tanh, Softmax
│   ├── neural_layer.py      # Base Layer class + Dense (fully connected) layer
│   ├── neural_network.py    # NeuralNetwork: forward, backward, weight update, save/load
│   ├── optimizers.py        # SGD, Momentum, NAG, RMSprop, Adam & Nadam implemented
│   └── __init__.py
└── utils/
    ├── data_loader.py       # Dataset loading, preprocessing, one-hot encoding, W&B image logging
    └── __init__.py
```

---

## Implementation Details

### Neural Network (`ann/neural_network.py`)

The `NeuralNetwork` class manages a list of layers and provides:
- `forward(X)` : forward pass returning logits (pre-softmax)
- `forward_with_softmax(X)` : full forward pass including the final Softmax
- `backward(Y_pred, Y_true, loss_type)` : backpropagation; returns `(grad_W_list, grad_b_list)`
- `update_weights(optimizer)` : calls each optimizer's `update()` on every Dense layer
- `compute_loss(Y_pred, Y_true, loss_type)` : cross-entropy or MSE loss with L2 regularization
- `save_model(filepath)` / `set_weights(weights)` : model checkpointing via `.npy` files

### Layers (`ann/neural_layer.py`)

- **`Dense`** : fully connected layer with configurable weight initialization (`zero`, `xavier`, `random`) and optional L2 regularization (`l2_lambda`). Stores input `X` during the forward pass for use in backprop. Computes `grad_W`, `grad_b` during the backward pass.

### Activations (`ann/activations.py`)

| Class     | Forward                                     | Backward                                        |
| --------- | -------------------------------------------- | ------------------------------------------------ |
| `ReLU`    | `max(0, Z)`                                  | `dZ = dA * (Z > 0)`                               |
| `Sigmoid` | `1 / (1 + exp(-Z))` (numerically stable)     | `dZ = dA * A * (1 - A)`                           |
| `Tanh`    | `np.tanh(Z)`                                 | `dZ = dA * (1 - A^2)`                             |
| `Softmax` | `exp(Z) / sum(exp(Z))` (numerically stable)  | handles both cross-entropy & MSE via chain rule   |

### Optimizers (`ann/optimizers.py`)

| Class      | Description                                                        | Key Hyperparameters                                     |
| ---------- | ------------------------------------------------------------------- | -------------------------------------------------------- |
| `SGD`      | mini-batch gradient descent                                         | `learning_rate`                                           |
| `Momentum` | SGD with momentum                                                    | `learning_rate`, `gamma=0.9`                              |
| `Nesterov` | Nesterov Accelerated Gradient (NAG)                                  | `learning_rate`, `gamma=0.9`                              |
| `RMSprop`  | Adaptive learning rate via moving average of squared gradients      | `learning_rate`, `decay_rate=0.9`, `epsilon=1e-8`          |
| `Adam`   | Adaptive moment estimation, combines momentum with per-parameter adaptive learning rates, bias-corrected | `learning_rate`, `beta1=0.9`, `beta2=0.999`, `epsilon=1e-8` |
| `Nadam` | Adam with Nesterov-style lookahead on the first-moment estimate     | `learning_rate`, `beta1=0.9`, `beta2=0.999`, `epsilon=1e-8` |

> Adam and Nadam were implemented but excluded from this project's scope, only SGD, Momentum, NAG, and RMSprop were used for training and the sweep.

### Data Pipeline (`utils/data_loader.py`)

- Loads MNIST or Fashion-MNIST via Keras
- Splits the 60k training set into **80% train / 20% validation** (stratified, `random_state=42`)
- Flattens images to 784-dimensional vectors and normalizes pixel values to `[0, 1]`
- One-hot encodes labels into 10-class vectors
- Logs 5 sample images per class to a W&B table for visualization

---

## Getting Started

### Installation

```bash
pip install numpy scikit-learn wandb keras tensorflow matplotlib
```

### Train

```bash
python train.py [options]
```

#### Arguments

| Argument            | Short  | Default          | Choices / Type                      | Description                    |
| -------------------- | ------ | ----------------- | -------------------------------------- | --------------------------------- |
| `--dataset`         | `-d`   | `mnist`          | `mnist`, `fashion_mnist`            | Dataset to use                 |
| `--epochs`          | `-e`   | `20`             | int                                  | Number of training epochs      |
| `--batch_size`      | `-b`   | `128`            | int                                  | Mini-batch size                |
| `--loss`            | `-l`   | `cross_entropy`  | `cross_entropy`, `mse`              | Loss function                  |
| `--optimizer`       | `-o`   | `rmsprop`        | `sgd`, `momentum`, `nag`, `rmsprop`, `adam`, `nadam`  | Optimizer    |
| `--learning_rate`   | `-lr`  | `0.001818`       | float                                | Learning rate                  |
| `--weight_decay`    | `-wd`  | `0.00050751`     | float                                | L2 regularization coefficient  |
| `--num_layers`      | `-nhl` | `2`              | int                                  | Number of hidden layers        |
| `--hidden_size`     | `-sz`  | `[128, 128]`     | int (one or more)                   | Neurons per hidden layer       |
| `--activation`      | `-a`   | `relu`           | `sigmoid`, `tanh`, `relu`           | Hidden layer activation        |
| `--weight_init`     | `-w_i` | `xavier`         | `random`, `xavier`, `zero`          | Weight initialization scheme   |
| `--wandb_project`   | `-w_p` | `assignment_1`   | str                                   | W&B project name                |
| `--model_save_path` | `-msp` | `best_model.npy` | str                                   | Path to save the best model     |

#### Example

```bash
# Train a 3-layer network with RMSprop on Fashion-MNIST
python train.py -d fashion_mnist -e 20 -b 128 -o rmsprop -lr 0.001 \
                -nhl 3 -sz 128 128 128 -a relu -w_i xavier -wd 0.0005
```

If `--hidden_size` receives a single value while `--num_layers > 1`, that size is replicated for all hidden layers automatically.

### Training Loop Behaviour

- Data is shuffled every epoch.
- Dead neurons (ReLU units that never activated across an entire epoch) are tracked per layer and logged to W&B.
- Gradient norms for the first layer are logged for the first 50 steps.
- The model is checkpointed (weights + config) whenever a new best **validation macro F1** is achieved.
- NaN loss detection stops a run early and reports `val_f1_score = 0.0` to the W&B sweep optimizer.

---

## Hyperparameter Sweep

```bash
python sweep.py          # Initialises the sweep on W&B and prints the agent command
wandb agent <sweep_id>   # Run in one or more terminals to start sweep agents
```

Uses **Bayesian optimization** to maximize `val_f1_score` over:

| Hyperparameter  | Values / Distribution        |
| ----------------- | ------------------------------- |
| `epochs`        | 10, 15, 20                   |
| `batch_size`    | 32, 64, 128                  |
| `learning_rate` | log-uniform in `[1e-5, 0.1]` |
| `optimizer`     | sgd, momentum, nag, rmsprop  |
| `activation`    | relu, sigmoid, tanh          |
| `weight_init`   | random, xavier               |
| `num_layers`    | 2, 3, 4, 5                   |
| `hidden_size`   | 16, 32, 64, 128              |
| `loss`          | cross_entropy, mse           |
| `weight_decay`  | log-uniform in `[1e-5, 0.1]` |

---

## Inference

```bash
python inference.py
```

This script:
1. Loads `best_config.json` and `best_model.npy` (paths resolve relative to `inference.py` automatically).
2. Rebuilds the network architecture and loads the saved weights.
3. Evaluates on the full test set and prints **Accuracy, Precision, Recall, and macro F1-score**.
4. Saves a **confusion matrix** plot as `confusion_matrix.png`.
5. Generates a **"Most Confident Failures"** analysis, finds the 5 test samples the model was most confident about yet predicted incorrectly, plots each image alongside its full softmax probability distribution, and saves it as `confident_failures_analysis.png`.

Override paths:

```bash
python inference.py -m path/to/model.npy -c path/to/config.json
```

---

## Best Configuration Found

```json
{
    "dataset":        "mnist",
    "epochs":         25,
    "batch_size":     128,
    "loss":           "cross_entropy",
    "optimizer":      "rmsprop",
    "learning_rate":  0.001818,
    "weight_decay":   0.00050751,
    "num_layers":     2,
    "hidden_size":    [128, 128],
    "activation":     "relu",
    "weight_init":    "xavier"
}
```

**Test accuracy: 97.35% | Macro F1: 0.97753** 

---

## W&B Logging

Per-epoch metrics tracked:
- `train_loss`, `val_loss`
- `train_accuracy`, `val_accuracy`
- `val_f1_score` (macro, used as sweep objective)
- `total_dead_neurons`, `layer_<i>_dead_count` - dead ReLU neuron counts
- `grad_norm_L1` - L1 gradient norm of the first layer
- `neuron_<n>_grad` - per-neuron gradient values for the first 5 neurons, logged for the first 50 steps
- Sample images table (5 images per class) at the start of each run

---

## Findings

The W&B report goes beyond training logs, it documents a structured investigation into *why* the model behaves the way it does. Summary of what was explored (full plots and derivations in the linked report & `Findings.md`). One thing to note here that Adam & Nadam optimizers aren't used while doing following things.:

| Investigation | Setup | Key Finding |
|---|---|---|
| **Class Similarity & Visual Overlap** | 5 sample images per class logged to a W&B table | Digit pairs 4/9, 3/5, and 2/8 show significant pixel-space overlap due to shared curves/strokes; since the MLP operates on raw flattened pixels with no spatial feature extraction, this overlap directly causes confident misclassifications. |
| **Hyperparameter Sweep** | 100-run Bayesian sweep (optimizer, loss, architecture, learning rate, init, weight decay) maximizing `val_f1_score` | Best val F1 = **0.9775** (RMSprop, 2×128 ReLU, lr=0.001818, Xavier init, cross-entropy, wd=0.00050751). Optimizer choice had the single largest impact, RMSprop runs consistently outperformed others, followed by loss function (cross-entropy > MSE). Architecture size (depth/width/batch size) had near-zero importance. |
| **Optimizer Comparison** | SGD, Momentum, NAG, RMSprop on identical 3×128 ReLU architecture, 10 epochs | RMSprop converged from epoch 1, reaching 97.3% val accuracy and 0.032 train loss by epoch 10; SGD, Momentum, and NAG all stayed stuck above 2.0 loss. RMSprop's per-parameter adaptive scaling handles the highly variable gradient magnitudes of sparse pixel inputs far better than a fixed learning rate. |
| **Vanishing Gradients: Sigmoid v/s ReLU** | RMSprop + Xavier, ReLU vs. Sigmoid at 2 and 4 hidden layers, gradient norms logged for layer 1 | Confirmed vanishing gradients with Sigmoid: 4-layer Sigmoid gradient norms collapsed to ~0.00004–0.002 (val accuracy stuck near random chance, ~10%, for several epochs), while ReLU held healthy norms (0.004–0.008) at both depths and converged to ~97%. |
| **Dead Neuron Investigation** | ReLU at a high learning rate (0.1) vs. ReLU at normal lr vs. Tanh, same architecture | At lr=0.1, ReLU lost ~90% of neurons (348/384) as permanently dead by epoch 2, flatlining val accuracy at ~10%. Tanh at the same high lr showed zero dead neurons (its derivative is always positive) but still failed to converge - its failure mode is gradient-killing *saturation*, not neuron death. ReLU at normal lr (0.001818) showed zero dead neurons and converged to ~97%. |
| **Loss Function Comparison** | Cross-entropy vs. MSE, identical architecture/optimizer/learning rate | Cross-entropy converged faster and higher: ~97.3% vs. ~96.5% val accuracy by epoch 10. Cross-entropy's gradient with softmax simplifies to `(predicted − true)`, a clean, strong error signal, whereas MSE's gradient involves the full softmax Jacobian and penalizes confident wrong predictions far less aggressively. |
| **Global Performance Analysis** | Train vs. test accuracy scatter across all 100 sweep runs, colored by learning rate | No severe overfitting observed, high-performing runs cluster tightly along the diagonal (train ≈ val ≈ 90%+). A middle cluster (60–80% accuracy) reflects underfitting/slow convergence, not overfitting, given MNIST's size (48k samples) relative to the capped 128-neuron width. |
| **Error Analysis** | Confusion matrix + "most confident failures" on the best model, MNIST test set (10,000 samples) | 9,793/10,000 correct. Top confusions: 9→4 (12 cases), 5→3 (10), 2→7 (8) ; consistent with the visual-similarity findings above. The 5 most confident wrong predictions all exceeded 99% confidence (e.g. an unusually cursive "8" predicted as "4" at 99.82%), showing the model is confidently misled by atypical handwriting rather than simply uncertain. |
| **Weight Initialization & Symmetry Breaking** | Zero init vs. Xavier init, per-neuron gradients for 5 neurons tracked over 50 iterations | Zero init produced perfectly overlapping gradient lines for all 5 neurons (mathematically equivalent to a single neuron per layer) and never escaped ~10% accuracy. Xavier init broke symmetry immediately, 5 clearly distinct gradient trajectories, and reached 85% accuracy within 5 epochs, confirming symmetry breaking is a mathematical prerequisite for an MLP to function. |
| **Fashion-MNIST Transfer** | 3 configs chosen from MNIST learnings, budget-constrained | The best MNIST config (RMSprop, ReLU, 2×128, Xavier, cross-entropy) transferred best to Fashion-MNIST too (~90% val accuracy), confirming the core optimizer/loss/init choices generalize across datasets; though all configs showed more oscillation and a lower ceiling (~88–90% vs. MNIST's 97%+), reflecting Fashion-MNIST's greater inter-class visual overlap (sleeves, collars, textures) that a spatially-blind MLP struggles to separate. |

Full plots, equations, and detailed reasoning for each investigation are in the [W&B report](https://wandb.ai/prasid-indian-institute-of-technology-madras/assignment_1/reports/DA6401-Assignment-1-PH21B007-PRASID--VmlldzoxNjEyODA5Ng?accessToken=7lf6abidol3880zy7aiflc38domgwf0gtrwlsqz0fhboc9dumm1bdqjfn0fs1042) (and `Findings.md` in this repo).

---

## Background

Built using concepts taught in the course *DA6401: Introduction to Deep Learning* (IIT Madras).

Part of a deep learning project series:
[MLP from Scratch](https://github.com/4prasid/MLP-from-scratch) · [Multi-task Vision](https://github.com/4prasid/multitask-vision-vgg11) · [Transformer NMT](https://github.com/4prasid/transformer-nmt-from-scratch)

---

## License

[MIT](LICENSE)
