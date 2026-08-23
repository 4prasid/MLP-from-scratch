import numpy as np

# base Layer class
class Layer:
    def __init__(self): 
        pass
    def forward(self, input_data): 
        pass
    def backward(self, output_gradients): 
        pass

# Dense layer 
class Dense(Layer):
    ''' 
    Fully connected layer with general forward and backward passes, along with
    weight initialization and optional L2 regularization 
    input_size: number of input features,
    output_size: number of neurons in the layer,
    weight_init: method for initializing weights (zero, xavier, random),
    l2_lambda: regularization strength for L2 regularization (default 0.0) 
    '''
    
    def __init__(self, input_size, output_size, weight_init='xavier', l2_lambda=0.0):
        super().__init__()

        # keep track of lambda for L2 regularization
        self.l2_lambda = l2_lambda
        
        # Initialize weights based on the specified method
        if weight_init == 'zero':
            self.W = np.zeros((input_size, output_size))
        elif weight_init == 'xavier':
            limit = np.sqrt(6 / (input_size + output_size))
            self.W = np.random.uniform(-limit, limit, (input_size, output_size))
        elif weight_init == 'random':
            self.W = np.random.randn(input_size, output_size) * 0.1
        else:
            raise ValueError("Invalid weight initialization method, choose 'zero', 'xavier', or 'random'")
            
        # Initialize biases to zero    
        self.b = np.zeros((1, output_size))
        
    # Forward pass 
    def forward(self, X):
        # work in 2D: reshape (n,) to (1, n) so backprop shapes are consistent
        X2d = X.reshape(1, -1) if X.ndim == 1 else X
        self.X = X2d  # Save 2D input for backprop
        return np.dot(X2d, self.W) + self.b
        
    # Backprop    
    def backward(self, dZ):
        # Ensure dZ is 2D
        dZ2d = dZ.reshape(1, -1) if dZ.ndim == 1 else dZ
        m = self.X.shape[0]

        # calculate gradients
        self.grad_W = np.dot(self.X.T, dZ2d) / m + ((self.l2_lambda / m) * self.W) 
        self.grad_b = np.sum(dZ2d, axis=0, keepdims=True) / m
        # lowercase alias for autograder compatibility
        self.grad_w = self.grad_W.copy()

        # Return gradient w.r.t. i/p for backprop to previous layers
        return np.dot(dZ2d, self.W.T) 