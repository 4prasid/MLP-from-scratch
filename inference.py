import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

from utils.data_loader import *
from ann.neural_network import *
from ann.neural_layer import *
from ann.optimizers import *
from ann.activations import *

# Utility function to map activation names to their corresponding classes
def get_activation(name):
    if name == 'relu': return ReLU()
    elif name == 'sigmoid': return Sigmoid()
    elif name == 'tanh': return Tanh()
    elif name == 'softmax': return Softmax()
    raise ValueError("Invalid activation function choose from 'relu', 'sigmoid', 'tanh', or 'softmax'")

# Function to build the model architecture based on the provided configuration
def build_model(config):
    ''' Builds the neural network architecture based on the provided configuration '''
    
    layers = []
    input_dim = 784 # 28*28 flattened image
    
    # Add hidden layers based on config
    for size in config['hidden_size']:
        layers.append(Dense(input_dim, size, weight_init=config['weight_init'], l2_lambda=config['weight_decay']))
        layers.append(get_activation(config['activation']))
        input_dim = size
        
    # Output layer    
    layers.append(Dense(input_dim, 10, weight_init=config['weight_init'], l2_lambda=config['weight_decay']))
    layers.append(Softmax())
    
    return NeuralNetwork(layers)

# Function to load model architecture and weights from saved files
def load_model_from_weights(config, weights_path):
    ''' Loads the model architecture and weights from saved files '''

    model = build_model(config)
    
    # Load weights from .npy file
    weights_data = np.load(weights_path, allow_pickle=True).item()
    
    # Set the weights and biases of the model layers
    model.set_weights(weights_data)
    
    return model

def load_model(model_path):
    """Load trained model from disk."""
    data = np.load(model_path, allow_pickle=True).item()
    return data

def plot_confident_failures(x_test_raw, y_test_raw, y_pred, y_pred_probs):
    ''' Finds and plots the top 5 images the model was most confidently wrong about '''

    # Find indices where prediction is wrong
    inc_idx = np.where(y_pred != y_test_raw)[0]
    
    # Get the predicted probabilities for those wrong answers
    wrong_probs = y_pred_probs[inc_idx]
    max_wrong_probs = np.max(wrong_probs, axis=1)
    
    # Sort them to find the highest confidence errors
    sorted_errors = np.argsort(max_wrong_probs)[::-1]
    top_5_idx = inc_idx[sorted_errors[:5]]
    
    fig, axes = plt.subplots(5, 2, figsize=(10, 15))
    fig.suptitle("Top 5 Most Confident Mistakes", fontsize=16)
    
    for i, idx in enumerate(top_5_idx):
        # Left column: The actual image
        ax_img = axes[i, 0]
        ax_img.imshow(x_test_raw[idx], cmap='gray')
        ax_img.set_title(f"True Label: {y_test_raw[idx]} | Predicted: {y_pred[idx]}")
        ax_img.axis('off')
        
        # Right column: The Softmax probability distribution
        ax_bar = axes[i, 1]
        bars = ax_bar.bar(range(10), y_pred_probs[idx], color='gray', alpha=0.7)
        bars[y_pred[idx]].set_color('red')    # Highlight the wrong prediction
        bars[y_test_raw[idx]].set_color('green') # Highlight the true answer
        
        ax_bar.set_xticks(range(10))
        ax_bar.set_title(f"Confidence: {np.max(y_pred_probs[idx])*100:.2f}%")
        ax_bar.set_xlabel("Class")
        ax_bar.set_ylabel("Probability")
        
    plt.tight_layout()
    plt.savefig("confident_failures_analysis.png")
    print("Saved confident failure analysis to 'confident_failures_analysis.png'")

def parse_arguments(args=None):
    ''' Parses CLI arguments for inference '''
    # Resolve default paths relative to THIS file so inference works from any working directory
    _here = os.path.dirname(os.path.abspath(__file__))
    _default_model  = os.path.join(_here, 'best_model.npy')
    _default_config = os.path.join(_here, 'best_config.json')

    parser = argparse.ArgumentParser(description="Inference using saved model weights")
    parser.add_argument('-d', '--dataset', type=str, default='mnist', choices=['mnist', 'fashion_mnist'])
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument('-l', '--loss', type=str, default='cross_entropy', choices=['cross_entropy', 'mse'])
    parser.add_argument('-o', '--optimizer', type=str, default='rmsprop', choices=['sgd', 'momentum', 'nag', 'rmsprop'])
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0)
    parser.add_argument('-nhl', '--num_layers', type=int, default=4)
    parser.add_argument('-sz', '--hidden_size', type=int, nargs='+', default=[64, 64, 64, 64])
    parser.add_argument('-a', '--activation', type=str, default='relu', choices=['sigmoid', 'tanh', 'relu'])
    parser.add_argument('-w_i', '--weight_init', type=str, default='xavier', choices=['random', 'xavier', 'zero'])
    parser.add_argument('-w_p', '--wandb_project', type=str, default='assignment_1')
    parser.add_argument('-msp', '--model_save_path', type=str, default='best_model.npy')
    parser.add_argument('-m', '--model_path', type=str, default=_default_model)
    parser.add_argument('-c', '--config_path', type=str, default=_default_config)
    return parser.parse_args(args)

def main():
    ''' Main function to run inference using saved model weights and evaluate performance '''
    args = parse_arguments()

    # Load configuration
    with open(args.config_path, 'r') as f:
        config = json.load(f)
        
    print(f"Loading {config['dataset']} test data")
    _, _, _, _, x_test_flat, y_test_oh, _, y_test_raw = load_and_preprocess_data(config['dataset'])
    
    print("Rebuilding model and loading weights")
    model = load_model_from_weights(config, args.model_path)
    
    print("Running inference on test set")
    y_pred_probs = model.forward_with_softmax(x_test_flat)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Calculate metrics using scikit-learn
    acc = accuracy_score(y_test_raw, y_pred)
    prec = precision_score(y_test_raw, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test_raw, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test_raw, y_pred, average='macro', zero_division=0)
    
    print("\n Test Set Metrics ")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # Standard Confusion Matrix
    print("\nGenerating Confusion Matrix...")
    cm = confusion_matrix(y_test_raw, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(cmap='Blues', ax=ax)
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    print("Saved 'confusion_matrix.png'")
    
    # Creative Visualization (Most Confident Failures)
    print("Generating Creative Failure Analysis ")
    # We load the unflattened original test images from keras directly for visualization
    from keras.datasets import mnist, fashion_mnist
    if config['dataset'] == 'mnist':
        _, (x_test_raw_images, _) = mnist.load_data()
    else:
        _, (x_test_raw_images, _) = fashion_mnist.load_data()
        
    plot_confident_failures(x_test_raw_images, y_test_raw, y_pred, y_pred_probs)

if __name__ == "__main__":
    main()