import wandb

# Define the hyperparameter space
sweep_config = {
    'program': 'train.py', # W&B will automatically run this file
    'method': 'bayes',  
    'metric': { 'name': 'val_f1_score',
                'goal': 'maximize'} ,
    'parameters': {
        # W&B will pass these as command-line arguments: e.g., --epochs 10 --batch_size 32
        'dataset': {'value': 'mnist'},
        'epochs': {'values': [5, 10, 15]},
        'batch_size': {'values': [16, 32, 64, 128]},
        'learning_rate': { 'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1},
        'optimizer': {'values': ['sgd', 'momentum', 'nag', 'rmsprop']},
        'activation': {'values': ['relu', 'sigmoid', 'tanh']},
        'weight_init': {'values': ['random', 'xavier']},
        'num_layers': {'values': [2, 3, 4, 5, 6]},
        'hidden_size': {'values': [16, 32, 64, 128]},
        'loss': {'values': ['cross_entropy', 'mse']} 
    }
}

# Initialize the sweep on W&B servers
sweep_id = wandb.sweep(sweep_config, project="assignment_1")
print(f"Sweep initialized. Run this command in your terminal to start the agents:")
print(f"wandb agent {sweep_id}")