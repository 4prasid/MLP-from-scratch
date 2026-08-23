import argparse
import os

# Safe wandb wrapper - falls back to no-op if wandb unavailable or no internet
try:
    import wandb as _wandb
    # _wandb.init(mode="disabled")  # test if wandb works at all, comment out during wandb submission
    _wandb_available = True
except Exception:
    _wandb_available = False

class _DummyRun:
    def log(self, *a, **kw): pass
    def finish(self, *a, **kw): pass

class _DummyWandb:
    def init(self, **kw): return _DummyRun()
    def log(self, *a, **kw): pass
    def finish(self, *a, **kw): pass

if _wandb_available:
    import wandb
else:
    wandb = _DummyWandb()
import numpy as np
import json
from sklearn.metrics import f1_score
from utils.data_loader import *
from ann.neural_network import *
from ann.neural_layer import *
from ann.optimizers import *
from ann.activations import *

# Helper functions to get activation and optimizer instances based on user i/p
def get_activation(name):
    if name == 'relu': return ReLU()
    elif name == 'sigmoid': return Sigmoid()
    elif name == 'tanh': return Tanh()
    elif name == 'softmax': return Softmax()
    raise ValueError("Invalid activation function choose from 'relu', 'sigmoid', 'tanh', or 'softmax'")

# Helper function to get optimizer instance based on user i/p
def get_optimizer(name, lr):
    if name == 'sgd': return SGD(learning_rate=lr)
    elif name == 'momentum': return Momentum(learning_rate=lr)
    elif name == 'nag': return Nesterov(learning_rate=lr)
    elif name == 'rmsprop': return RMSprop(learning_rate=lr)
    # elif name == 'adam': return Adam(learning_rate=lr)
    # elif name == 'nadam': return Nadam(learning_rate=lr)
    raise ValueError("Invalid optimizer choose from 'sgd', 'momentum', 'nag' or 'rmsprop'")

def run_name(args):
    '''Generates a descriptive run name based on the hyperparameters for better organization in W&B'''

    lr_str = f"lr{args.learning_rate:.0e}".replace('+0', '').replace('-0', '-')
    loss_abbr = 'ce' if args.loss == 'cross_entropy' else 'mse'

    # Use the first hidden size value as representative (they're often all the same)
    sz = args.hidden_size[0] if isinstance(args.hidden_size, list) else args.hidden_size
    name = (
        f"{args.optimizer}_"
        f"{args.activation}_"
        f"{lr_str}_"
        f"hl{args.num_layers}_"
        f"sz{sz}_"
        f"bs{args.batch_size}_"
        f"{loss_abbr}"
    )

    return name

def evaluate_in_batches(model, X, Y, batch_size, loss_type):
    ''' Evaluates loss and accuracy in batches '''
    total_loss = 0
    correct_preds = 0
    all_preds = []
    
    for i in range(0, X.shape[0], batch_size):
        x_batch = X[i:i+batch_size]
        y_batch = Y[i:i+batch_size]
        
        y_pred_probs = model.forward_with_softmax(x_batch)
        total_loss += model.compute_loss(y_pred_probs, y_batch, loss_type) * x_batch.shape[0]
        
        preds = np.argmax(y_pred_probs, axis=1)
        trues = np.argmax(y_batch, axis=1)
        correct_preds += np.sum(preds == trues)
        all_preds.extend(preds)
        
    avg_loss = total_loss / X.shape[0]
    accuracy = correct_preds / X.shape[0]
    
    return avg_loss, accuracy, np.array(all_preds)


# Training loop
def train(model, optimizer, x_train, y_train, x_val, y_val, args):
    ''' Train the model for a specified number of epochs, logging metrics to Weights & Biases'''

    best_val_f1 = 0.0 # To keep track of best validation F1 score for model checkpointing
    gs = 0 # Global step counter for logging gradients

    # Loop over epochs
    for epoch in range(args.epochs):
        perm = np.random.permutation(x_train.shape[0]) # Shuffle training data each epoch
        x_train_shuffled = x_train[perm]
        y_train_shuffled = y_train[perm] 
        dt = {} # dead neuron tracker for ReLU layers

        # Loop over mini-batches
        for i in range(0, x_train.shape[0], args.batch_size):
            x_batch = x_train_shuffled[i:i+args.batch_size]
            y_batch = y_train_shuffled[i:i+args.batch_size]
            
            # Forward pass
            y_pred = model.forward_with_softmax(x_batch)
            
            # Count dead neurons in ReLU layers
            for la_idx, layer in enumerate(model.layers):
                if type(layer).__name__ == 'ReLU':
                    # dt[la_idx] = None
                    batch_active = np.any(layer.A > 0, axis=0)
                    if la_idx not in dt:
                        # first batch initialise tracker
                        dt[la_idx] = batch_active.copy()
                    else:
                        # accumulate activity across batches
                        dt[la_idx] |= batch_active 

                    # n_dead = np.sum(np.all(layer.A == 0, axis=0))
                    # dn += n_dead
                    # wandb.log({f"layer_{la_idx}_dead_count": n_dead}, commit=False)
            
            # Compute loss
            loss = model.compute_loss(y_pred, y_batch, args.loss)

            # Backward pass 
            model.backward(y_pred, y_batch, args.loss)

            if gs < 50: # Log gradient norms for the first 50 steps to avoid excessive logging
                first_layer = model.layers[0]
                ng = {}
                n_neurons = min(5, first_layer.grad_W.shape[1])  # guard against small layers
                for n in range(n_neurons):
                    grad_val = np.linalg.norm(first_layer.grad_W[:, n])  # L2 norm of gradients for neuron n
                    ng[f"neuron_{n}_grad"] = grad_val
                    
                # Log to W&B alongside the iteration step 
                wandb.log({"iteration": gs, **ng}, commit=True) 
                gs += 1

            # Update weights 
            model.update_weights(optimizer)

        # Coompute train loss, train accuracy in batches
        train_loss, train_acc, _ = evaluate_in_batches(model, x_train, y_train, args.batch_size, args.loss)

        # Compute validation loss, validation accuracy, and validation predictions in batches
        val_loss, val_acc, val_preds = evaluate_in_batches(model, x_val, y_val, args.batch_size, args.loss)
        
        # calculate true validation labels from one-hot vectors
        val_true = np.argmax(y_val, axis=1)

        # Calculate Validation F1 Score
        val_f1 = f1_score(val_true, val_preds, average='macro')

        # Guard against nan metrics that break Bayesian sweep optimizer
        if np.isnan(train_loss) or np.isnan(val_loss):
            print("NaN loss detected, stopping run early")
            wandb.log({"val_f1_score": 0.0})
            break

        # Log dead neuron counts for all ReLU layers and total dead neurons across the network
        tdn = 0

        for la_idx, tr in dt.items():
            if tr is None:
                continue

            n_dead = np.sum(~tr)   # neurons never activated
            tdn += n_dead

            wandb.log({f"layer_{la_idx}_dead_count": n_dead}, commit=False)
        
        # Log loss and accuracy to Weights & Biases
        wandb.log({
            "epoch": epoch + 1, 
            "train_loss": train_loss, 
            "val_loss": val_loss,
            "train_accuracy": train_acc, 
            "val_accuracy": val_acc, 
            "val_f1_score": val_f1,
            "total_dead_neurons": tdn, 
            "grad_norm_L1": np.linalg.norm(model.layers[0].grad_W)
        })

        print(f"Epoch {epoch + 1}/{args.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

        # Save best config and weights based on validation F1 score
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            print(f"New best Validation F1 Score ({best_val_f1:.4f}) found, so saving model")
            model.save_model(args.model_save_path)
            with open("best_config.json", "w") as f:
                json.dump(vars(args), f, indent=4)

# main function to parse arguments, load data, build model, and start training
def parse_arguments(args=None):
    parser = argparse.ArgumentParser(description="Train a Neural Network from scratch")
    parser.add_argument('-d', '--dataset', type=str, default='mnist', choices=['mnist', 'fashion_mnist'])
    parser.add_argument('-e', '--epochs', type=int, default=20)
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument('-l', '--loss', type=str, default='cross_entropy', choices=['cross_entropy', 'mse'])
    parser.add_argument('-o', '--optimizer', type=str, default='rmsprop', choices=['sgd', 'momentum', 'nag', 'rmsprop'])
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001818)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.00050751)
    parser.add_argument('-nhl', '--num_layers', type=int, default=2)
    parser.add_argument('-sz', '--hidden_size', type=int, nargs='+', default=[128, 128])
    parser.add_argument('-a', '--activation', type=str, default='relu', choices=['sigmoid', 'tanh', 'relu'])
    parser.add_argument('-w_i', '--weight_init', type=str, default='xavier', choices=['random', 'xavier', 'zero'])
    parser.add_argument('-w_p', '--wandb_project', type=str, default='assignment_1')
    parser.add_argument('-msp', '--model_save_path', type=str, default='best_model.npy')
    parser.add_argument('-m',   '--model_path',     type=str, default='best_model.npy')
    parser.add_argument('-c',   '--config_path',    type=str, default='best_config.json')
    return parser.parse_args(args)

def main():
    args = parse_arguments()

    # Validate hidden layer sizes
    if len(args.hidden_size) != args.num_layers:
        if len(args.hidden_size) == 1:
            args.hidden_size = args.hidden_size * args.num_layers
        else:
            raise ValueError("Length of hidden_size list must match num_layers")
        
    run_name_str = run_name(args)

    # wandb.init(project=args.wandb_project, config=vars(args), name=run_name_str, mode="offline") # Use offline mode while submitting to gradescope
    wandb.init(project=args.wandb_project, config=vars(args), name=run_name_str)

    # Load and preprocess data
    print(f"Loading {args.dataset} dataset")
    x_train, y_train, x_val, y_val, x_test, y_test, x_raw, y_raw, y_test_raw = load_and_preprocess_data(args.dataset)
    
    # Log sample images for visualization in Weights & Biases
    log_sample_images(x_raw, y_raw)

    # Dynamically build network architecture
    layers = []
    input_dim = x_train.shape[1]
    
    # Add hidden layers based on user i/p, with specified activation and weight initialization
    for size in args.hidden_size:
        layers.append(Dense(input_dim, size, weight_init=args.weight_init, l2_lambda=args.weight_decay))
        layers.append(get_activation(args.activation))
        input_dim = size
        
    # Add output layer with softmax activation for classification    
    layers.append(Dense(input_dim, 10, weight_init=args.weight_init, l2_lambda=args.weight_decay))
    layers.append(Softmax())

    # Create model and optimizer instances
    model = NeuralNetwork(layers)
    optimizer = get_optimizer(args.optimizer, args.learning_rate)

    # Train the model
    print("Starting training")
    train(model, optimizer, x_train, y_train, x_val, y_val, args)
        
    wandb.finish()

if __name__ == "__main__":
    main()
