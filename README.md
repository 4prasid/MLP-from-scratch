# MLP From Scratch — NumPy Neural Network on MNIST & Fashion-MNIST

A fully-connected feedforward neural network (MLP) implemented **entirely from scratch in NumPy**; no PyTorch, TensorFlow, or Keras for the model itself. Forward passes, backpropagation, and every optimizer update rule are hand-derived and hand-coded. Keras is used only to load the raw MNIST & Fashion-MNIST datasets.

🔗 [W&B Report](https://forge.coreweave.com/wandb/prasid-indian-institute-of-technology-madras/assignment_1/reports/DA6401-Assignment-1-PH21B007-PRASID--VmlldzoxNjEyODA5Ng?accessToken=7lf6abidol3880zy7aiflc38domgwf0gtrwlsqz0fhboc9dumm1bdqjfn0fs1042) &nbsp;|&nbsp; 🔗 [GitHub Repo](https://github.com/4prasid/MLP-from-scratch)


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
pip install -r requirements.txt
```

Or manually:

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
- `total_dead_neurons`, `layer_<i>_dead_count` — dead ReLU neuron counts
- `grad_norm_L1` — L1 gradient norm of the first layer
- `neuron_<n>_grad` — per-neuron gradient values for the first 5 neurons, logged for the first 50 steps
- Sample images table (5 images per class) at the start of each run

---

## Background

Built using concepts taught in the course *DA6401: Introduction to Deep Learning* (IIT Madras).

Part of a deep learning project series:
[MLP from Scratch](https://github.com/4prasid/MLP-from-scratch) · Multi-task Vision (VGG11) · Transformer NMT

## License

[MIT](LICENSE)
