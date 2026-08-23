import numpy as np
import wandb
import matplotlib.pyplot as plt
from keras.datasets import mnist, fashion_mnist
from sklearn.model_selection import train_test_split

def one_hot(labels, nc): 
    ''' 
    Converts integer labels to one-hot vector format 
    labels: array of integer labels here y,
    nc: number of unique classes 
    '''

    oh = np.zeros((labels.shape[0], nc)) 
    oh[np.arange(labels.shape[0]), labels] = 1  # 
    return oh

def load_and_preprocess_data(dataset_name="mnist"):
    '''
    Loads and preprocesses the MNIST or Fashion-MNIST dataset 
    Returns preprocessed training, validation, and test sets along 
    with raw data (x_train, y_train) for visualization 
    dataset_name:  "mnist" or "fashion_mnist" 
    '''

    # Load dataset

    # MNIST and Fashion-MNIST are already split into train and test sets, so we will further split 
    # the training set into train and validation sets
    if dataset_name == "mnist":
        (x_train_full, y_train_full), (x_test_full, y_test_full) = mnist.load_data()
    elif dataset_name == "fashion_mnist":
        (x_train_full, y_train_full), (x_test_full, y_test_full) = fashion_mnist.load_data()
    else:
        raise ValueError("Dataset must be 'mnist' or 'fashion_mnist'")

    # Split training data into training and validation sets (80% train, 20% val) with 
    # stratification to maintain class distribution
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )

    # Flatten images and normalize pixel values to [0, 1]
    x_train_flat = x_train.reshape(x_train.shape[0], -1) / 255.0
    x_val_flat = x_val.reshape(x_val.shape[0], -1) / 255.0
    x_test_flat = x_test_full.reshape(x_test_full.shape[0], -1) / 255.0

    # One-hot labels
    num_classes = len(np.unique(y_train_full))
    y_train_oh = one_hot(y_train, num_classes)
    y_val_oh = one_hot(y_val, num_classes)
    y_test_oh = one_hot(y_test_full, num_classes)

    return x_train_flat, y_train_oh, x_val_flat, y_val_oh, x_test_flat, y_test_oh, x_train, y_train

def log_sample_images(x_raw, y_raw):
    ''' 
    Logs 5 sample images per class to W&B 
    x_raw: raw image data (not flattened) for visualization (x_train before flattening)
    y_raw: corresponding integer labels for the raw images (y_train before one-hot encoding) 
    '''
    
    # Create a W&B table to log images and labels
    table = wandb.Table(columns=["Image", "Label"])
    np.random.seed(42)
    for i in range(10):
        idx = np.where(y_raw == i)[0] 
        s_idx = np.random.choice(idx, 5, replace=False) 
        for j in s_idx:
            img = x_raw[j]
            label = y_raw[j]
            table.add_data(wandb.Image(img), label)
    wandb.log({"Sample Images": table})