import numpy as np
from ann.neural_layer import Dense
from ann.activations import ReLU, Sigmoid, Tanh, Softmax

# NeuralNetwork class that manages the layers, forward pass, backward pass, and weight updates
class NeuralNetwork:
    def __init__(self, layers_or_args):
        import argparse
        if isinstance(layers_or_args, argparse.Namespace):
            self._args = layers_or_args
            self.layers = []
        else:
            self._args = None
            self.layers = list(layers_or_args)

    def _build_from_weights(self, weights):
        """Build layers from weight shapes so architecture matches exactly."""
        
        # Handle case where weights is a numpy array containing a dict (e.g. from np.load)
        if isinstance(weights, np.ndarray):
            weights = weights.item()

        # Check if keys are in 'W0', 'b0' format or '0', '1' format and build w_list accordingly
        if any(isinstance(k, str) and k.startswith('W') for k in weights.keys()):
            indices = sorted(set(int(k[1:]) for k in weights if len(k)>1 and k[1:].isdigit()))
            w_list = [(weights[f'W{i}'], weights[f'b{i}']) for i in indices]
        else:
            sorted_keys = sorted(weights.keys(), key=lambda k: int(k))
            w_list = [(weights[k]['W'], weights[k]['b']) for k in sorted_keys]

        args = self._args

        # extract architecture and hyperparameters from args if available, else use defaults
        activation = getattr(args, 'activation', 'relu') if args else 'relu'
        w_init = getattr(args, 'weight_init', 'xavier') if args else 'xavier'
        w_decay = getattr(args, 'weight_decay', 0.0) if args else 0.0

        # Helper function to get activation layer based on name
        def get_act(name):
            if name == 'sigmoid': return Sigmoid()
            if name == 'tanh': return Tanh()
            return ReLU()

        # Build layers from w_list, inserting activations in between. Assume final layer is always softmax.
        layers = []
        for i, (W, b) in enumerate(w_list):
            W = np.array(W)
            in_dim, out_dim = W.shape
            layer = Dense(in_dim, out_dim, weight_init=w_init, l2_lambda=w_decay)
            layer.W = W.copy()
            layer.b = np.array(b).copy()
            layers.append(layer)
            if i < len(w_list) - 1:
                layers.append(get_act(activation))
        layers.append(Softmax())
        return layers

    # Forward pass through all layers — returns LOGITS (pre-softmax)
    def forward(self, X):
        '''Passes the input through all layers and returns the logits (pre-softmax output).
        Always runs the Softmax layer too (to cache .A for backward), but returns logits.'''
        logits = None
        for i, layer in enumerate(self.layers):
            if X is None:
                raise ValueError(f"Layer {i} ({type(self.layers[i-1]).__name__}) returned None!")
            is_last_layer = (i == len(self.layers) - 1)
            if is_last_layer and type(layer).__name__ == 'Softmax':
                logits = X          # save logits before softmax
                layer.forward(X)    # run softmax to cache .A (ignore return)
                break
            X = layer.forward(X)
        return logits if logits is not None else X

    def forward_with_softmax(self, X):
        '''Full forward pass including the final Softmax layer.'''
        for i, layer in enumerate(self.layers):
            if X is None:
                raise ValueError(f"Layer {i} ({type(self.layers[i-1]).__name__}) returned None!")
            X = layer.forward(X)
        return X

    # Backward pass through all layers in reverse order
    def backward(self, Y_pred_or_grad, Y_true=None, loss_type='cross_entropy'):
        '''Computes gradients.
        Returns (grad_W_list, grad_b_list) where each is a list of arrays ordered
        from last dense layer to first dense layer, so the caller can unpack: 
        grad_W, grad_b = model.backward(X, Y)
        Every Dense layer also stores gradients in .grad_W / .grad_w after this call.
        '''
        if Y_true is None:
            # The incoming gradient is w.r.t. softmax outputs (probabilities).
            # We must propagate through Softmax.backward() first so the chain
            # rule is complete, then through the remaining layers in reverse.
            dA = Y_pred_or_grad
            for layer in reversed(self.layers):
                if type(layer).__name__ == 'Softmax':
                    dA = layer.backward(dA, loss_type='cross_entropy')
                else:
                    dA = layer.backward(dA)
        else:
            first_dense = next((l for l in self.layers if hasattr(l, 'W')), None)
            softmax_layer = next(
                (l for l in reversed(self.layers) if type(l).__name__ == 'Softmax'), None
            )
            # Determine network input size from first Dense layer
            input_size = first_dense.W.shape[0] if first_dense is not None else -1
            Y_pred_2d = Y_pred_or_grad.reshape(1, -1) if Y_pred_or_grad is not None and Y_pred_or_grad.ndim == 1 else Y_pred_or_grad
            Y_true_2d = Y_true.reshape(1, -1) if Y_true.ndim == 1 else Y_true

            # Always run full forward pass to get correct probs:
            cached_layer = next((l for l in self.layers if hasattr(l, 'X')), None)
            if Y_pred_2d is not None and Y_pred_2d.shape[-1] == input_size:
                # Y_pred_or_grad is the input X -> run full forward
                x = Y_pred_2d
                for layer in self.layers:
                    x = layer.forward(x)
                probs = x
            elif cached_layer is not None:
                # Re-run from cached input
                x = cached_layer.X
                for layer in self.layers:
                    x = layer.forward(x)
                probs = x
            elif softmax_layer is not None and hasattr(softmax_layer, 'A'):
                probs = softmax_layer.A
            else:
                probs = Y_pred_2d if Y_pred_2d is not None else Y_pred_or_grad

            probs_2d = probs.reshape(1, -1) if probs.ndim == 1 else probs
            m = probs_2d.shape[0]
            if loss_type == 'cross_entropy':
                dA = (probs_2d - Y_true_2d) / m
            elif loss_type == 'mse':
                dA = 2 * (probs_2d - Y_true_2d) / m
            else:
                raise ValueError("Invalid loss type, choose 'cross_entropy' or 'mse'")
            for layer in reversed(self.layers):
                if type(layer).__name__ == 'Softmax':
                    dA = layer.backward(dA, loss_type=loss_type)
                else:
                    dA = layer.backward(dA)

        # Return (grad_W_list, grad_b_list) ordered last dense to first dense.
        dense_layers_rev = [l for l in reversed(self.layers) if hasattr(l, 'grad_W')]
        grad_W_list = [l.grad_W for l in dense_layers_rev]
        grad_b_list = [l.grad_b for l in dense_layers_rev]
        return grad_W_list, grad_b_list

    # Update weights of all layers using the provided optimizer
    def update_weights(self, optimizer):
        '''Updates the weights of all layers using the provided optimizer'''
        for layer in self.layers:
            optimizer.update(layer)

    # Compute loss with L2 regularization for all layers
    def compute_loss(self, Y_pred, Y_true, loss_type='cross_entropy'):
        '''Computes the loss with L2 regularization for all layers'''
        m = Y_true.shape[0]
        if loss_type == 'cross_entropy':
            loss = -np.mean(np.sum(Y_true * np.log(Y_pred + 1e-8), axis=1))
        elif loss_type == 'mse':
            loss = np.mean(np.sum(np.square(Y_pred - Y_true), axis=1))

        l2_loss = 0
        for layer in self.layers:
            if hasattr(layer, 'W'):
                l2_loss += (layer.l2_lambda / (2*m)) * np.sum(layer.W ** 2)
        return loss + l2_loss

    def get_weights(self):
        '''Returns a dictionary of weights and biases for each layer'''
        weights = {}
        idx = 0
        for layer in self.layers:
            if hasattr(layer, 'W'):
                weights[idx] = {'W': layer.W.copy(), 'b': layer.b.copy()}
                idx += 1
        return weights

    def set_weights(self, weights):
        '''Sets weights. If model was built from args, builds layers from weight shapes first.'''
        import numpy as np
        if isinstance(weights, np.ndarray):
            weights = weights.item()

        if not self.layers:
            self.layers = self._build_from_weights(weights)
            return

        keys = list(weights.keys())
        if any(isinstance(k, str) and k.startswith('W') for k in keys):
            layer_indices = sorted(set(int(k[1:]) for k in keys if len(k)>1 and k[1:].isdigit()))
            idx = 0
            for layer in self.layers:
                if hasattr(layer, 'W') and idx < len(layer_indices):
                    i = layer_indices[idx]
                    layer.W = np.array(weights[f'W{i}']).copy()
                    layer.b = np.array(weights[f'b{i}']).copy()
                    idx += 1
        else:
            sorted_keys = sorted(weights.keys(), key=lambda k: int(k))
            weight_list = [weights[k] for k in sorted_keys]
            idx = 0
            for layer in self.layers:
                if hasattr(layer, 'W') and idx < len(weight_list):
                    layer.W = np.array(weight_list[idx]['W']).copy()
                    layer.b = np.array(weight_list[idx]['b']).copy()
                    idx += 1

    def save_model(self, filepath="model.npy"):
        '''Saves the weights and biases of the network'''
        np.save(filepath, self.get_weights())
