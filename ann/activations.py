from ann.neural_layer import *

# Activation layers:

# ReLU    
class ReLU(Layer):
    ''' ReLU: f(z) = max(0, z) '''

    def __init__(self):
        super().__init__()

    def forward(self, Z):
        # ReLU : f(z) = max(0, z)
        self.Z = Z # Save i/p for backprop
        self.A = np.maximum(0, Z) # save for calculating the dead neurons
        return self.A

    def backward(self, dA):
        # Derivative of ReLU is 1 where Z > 0, else 0
        dZ = dA * (self.Z > 0)
        return dZ

# Sigmoid    
class Sigmoid(Layer):
    ''' Sigmoid: f(z) = 1 / (1 + exp(-z)) '''

    def __init__(self):
        super().__init__()

    def forward(self, Z):
        # Sigmoid : f(z) = 1 / (1 + exp(-z))
        self.Z = Z # Save i/p for backprop
        out = np.empty_like(Z)

        pos = Z >= 0
        neg = ~pos

        out[pos] = 1 / (1 + np.exp(-Z[pos]))
        exp_z = np.exp(Z[neg])
        out[neg] = exp_z / (1 + exp_z)

        self.A = out # save for backprop
        return self.A

    def backward(self, dA):
        # Derivative of sigmoid(s) is s*(1-s), here s is self.A
        dZ = dA * self.A * (1 - self.A)
        return dZ

# Softmax    
class Softmax(Layer):
    ''' Softmax: f(z_i) = exp(z_i) / sum(exp(z_k)) sum over all k classes '''

    def __init__(self):
        super().__init__()

    def forward(self, Z):
        # Softmax : f(z_i) = exp(z_i) / sum(exp(z_k)) sum over all k classes
        self.Z = Z # Save i/p for backprop
        Z2d = Z.reshape(1, -1) if Z.ndim == 1 else Z  # handle 1D input
        exp_Z = np.exp(Z2d - np.max(Z2d, axis=1, keepdims=True)) # for numerical stability
        A2d = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        self.A = A2d.reshape(Z.shape)  # restore original shape
        return self.A

    # For softmax, the backward pass depends on the loss function used   
    def backward(self, dA, loss_type='cross_entropy'):
        if loss_type == 'cross_entropy':
            # For softmax with cross-entropy, dA is already the gradient of the loss w.r.t. Z
            return dA
        elif loss_type == 'mse': 
            # For softmax with MSE, we need to compute the gradient of the loss w.r.t. Z using the chain rule
            s = self.A
            return s * (dA - np.sum(dA * s, axis=1, keepdims=True))
        else:
            raise ValueError("Invalid loss type choose 'mse' or 'cross_entropy'")

# Tanh    
class Tanh(Layer):
    ''' Tanh: f(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z)) '''

    def __init__(self):
        super().__init__()

    def forward(self, Z):
        # Tanh : f(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z))
        self.Z = Z  # Save i/p for backprop
        self.A = np.tanh(Z) # Save for backprop
        return self.A

    def backward(self, dA):
        # Derivative of tanh(z) is 1 - tanh^2(z), here tanh(z) is self.A
        dZ = dA * (1 - self.A ** 2)
        return dZ

