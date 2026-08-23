import numpy as np

# Base Optimizer class
class Optimizer:
    def update(self, layer):
        raise NotImplementedError( "Optimizer subclasses must implement one of the update method among SGD, Momentum, Nesterov, RMSprop, Adam, Nadam" )

# Implemention of the optimization algorithms as subclasses of the Optimizer class. 
# Each subclass implement the update method to adjust the weights and biases of the layers based on the computed gradients.

# SGD (Stochastic Gradient Descent)
class SGD(Optimizer):
    ''' learning_rate: step size for parameter updates '''

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        
    def update(self, layer):
        if not hasattr(layer, 'W'): 
            return
        
        # Update w & b using gradient descent
        layer.W -= self.learning_rate * layer.grad_W
        layer.b -= self.learning_rate * layer.grad_b

# Momentum
class Momentum(Optimizer):
    ''' gamma: momentum factor (0.9 is common), 
        learning_rate: step size for parameter updates '''

    def __init__(self, learning_rate=0.01, gamma=0.9):
        self.learning_rate = learning_rate
        self.gamma = gamma # Momentum factor
        self.state = {} # To store velocity for each layer
        
    def update(self, layer):
        if not hasattr(layer, 'W'): 
            return
        
        # Initialize velocity if not in state
        l_id = id(layer)
        if l_id not in self.state:
            self.state[l_id] = {
                'v_W': np.zeros_like(layer.grad_W), 
                'v_b': np.zeros_like(layer.grad_b)
                }
            
        # Store current velocity
        v_W  = self.state[l_id]['v_W']
        v_b  = self.state[l_id]['v_b']

        # Upate velocity v_{t+1} = gamma * v_t + learning_rate * grad_t
        v_W = self.gamma * v_W + self.learning_rate * layer.grad_W
        v_b = self.gamma * v_b + self.learning_rate * layer.grad_b
        
        # Update parameters w_{t+1} = w_t - v_{t+1}
        layer.W -= v_W
        layer.b -= v_b

        # Save updated velocity back to state
        self.state[l_id]['v_W'] = v_W
        self.state[l_id]['v_b'] = v_b

# NAG (Nesterov Accelerated Gradient)
class Nesterov(Optimizer):
    ''' gamma: momentum factor (0.9 is common), 
        learning_rate: step size for parameter updates '''
    
    def __init__(self, learning_rate=0.01, gamma=0.9):
        self.learning_rate = learning_rate
        self.gamma = gamma  
        self.state = {} 
        
    def update(self, layer):
        if not hasattr(layer, 'W'): 
            return
        
        l_id = id(layer)
        if l_id not in self.state:
            self.state[l_id] = {'v_W': np.zeros_like(layer.grad_W), 
                                'v_b': np.zeros_like(layer.grad_b)
                                }
        
        # Retrieve previous velocities
        v_W_prev = self.state[l_id]['v_W']
        v_b_prev = self.state[l_id]['v_b']

        # Compute lookahead parameters
        v_W = self.gamma * v_W_prev + self.learning_rate * layer.grad_W
        v_b = self.gamma * v_b_prev + self.learning_rate * layer.grad_b
        
        # Update parameters using lookahead velocities
        layer.W -= self.gamma * v_W + self.learning_rate * layer.grad_W
        layer.b -= self.gamma * v_b + self.learning_rate * layer.grad_b

        # Save updated velocities back to state
        self.state[l_id]['v_W'] = v_W
        self.state[l_id]['v_b'] = v_b

# RMSprop (Root Mean Square Propagation)
class RMSprop(Optimizer):
    ''' decay_rate: decay rate for moving average of squared gradients (0.9 is common),
        epsilon: small constant to avoid division by zero (1e-8 is set as default),
        learning_rate: step size for parameter updates '''
    
    def __init__(self, learning_rate=0.01, decay_rate=0.9, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate # Decay rate for moving average of squared gradients
        self.epsilon = epsilon # Small constant to avoid division by zero
        self.state = {} # To store moving average of squared gradients for each layer
        
    def update(self, layer):
        if not hasattr(layer, 'W'): 
            return
        
        l_id = id(layer)
        if l_id not in self.state:
            self.state[l_id] = {'s_W': np.zeros_like(layer.grad_W), 
                                's_b': np.zeros_like(layer.grad_b)
                                }
        
        # Retrieve current squared gradient accumulators
        s_W = self.state[l_id]['s_W']
        s_b = self.state[l_id]['s_b']
        
        # Update squared gradient accumulators
        s_W = self.decay_rate * s_W + (1 - self.decay_rate) * (layer.grad_W ** 2)
        s_b = self.decay_rate * s_b + (1 - self.decay_rate) * (layer.grad_b ** 2)
        
        # Update parameters using RMSprop update rule
        layer.W -= self.learning_rate * layer.grad_W / (np.sqrt(s_W + self.epsilon))
        layer.b -= self.learning_rate * layer.grad_b / (np.sqrt(s_b + self.epsilon))

        # Save updated squared gradient accumulators back to state
        self.state[l_id]['s_W'] = s_W
        self.state[l_id]['s_b'] = s_b
    
# # Adam (Adaptive Moment Estimation)
# class Adam(Optimizer):
#     ''' beta1: exponential decay rate for the first moment estimates (0.9 is common),
#         beta2: exponential decay rate for the second moment estimates (0.999 is common),
#         epsilon: small constant to avoid division by zero (1e-8 is set as default),
#         learning_rate: step size for parameter updates '''

#     def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
#         self.learning_rate = learning_rate
#         self.beta1 = beta1 # Exponential decay rate for the first moment estimates
#         self.beta2 = beta2 # Exponential decay rate for the second moment estimates
#         self.epsilon = epsilon 
#         self.state = {} # To store first and second moment estimates for each layer
        
#     def update(self, layer):
#         if not hasattr(layer, 'W'):
#             return
        
#         l_id = id(layer)
#         if l_id not in self.state:
#             self.state[l_id] = {
#                 'm_W': np.zeros_like(layer.grad_W), 
#                 'v_W': np.zeros_like(layer.grad_W),
#                 'm_b': np.zeros_like(layer.grad_b), 
#                 'v_b': np.zeros_like(layer.grad_b), 
#                 't': 0
#                 }

#         s = self.state[l_id]

#         # Retrieve current first and second moment estimates
#         m_W = s['m_W']
#         m_b = s['m_b']
#         v_W = s['v_W']
#         v_b = s['v_b']

#         # Update time step
#         s['t'] += 1
#         t = s['t']

#         # Update biased first moment estimates
#         m_W = self.beta1 * m_W + (1 - self.beta1) * layer.grad_W
#         m_b = self.beta1 * m_b + (1 - self.beta1) * layer.grad_b

#         # Update biased second moment estimates
#         v_W = self.beta2 * v_W + (1 - self.beta2) * (layer.grad_W ** 2)
#         v_b = self.beta2 * v_b + (1 - self.beta2) * (layer.grad_b ** 2)

#         # Compute bias-corrected first and second moment estimates
#         m_W_hat = m_W / (1 - self.beta1 ** t)
#         m_b_hat = m_b / (1 - self.beta1 ** t)
#         v_W_hat = v_W / (1 - self.beta2 ** t)
#         v_b_hat = v_b / (1 - self.beta2 ** t)

#         # Update parameters using Adam update rule
#         layer.W -= self.learning_rate * m_W_hat / (np.sqrt(v_W_hat) + self.epsilon)
#         layer.b -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

#         # Save updated first and second moment estimates back to state
#         s['m_W'] = m_W
#         s['m_b'] = m_b
#         s['v_W'] = v_W
#         s['v_b'] = v_b
        

# # Nadam (Nesterov-accelerated Adaptive Moment Estimation)
# class Nadam(Optimizer): 
#     ''' beta1: exponential decay rate for the first moment estimates (0.9 is common),
#         beta2: exponential decay rate for the second moment estimates (0.999 is common),
#         epsilon: small constant to avoid division by zero (1e-8 is set as default),
#         learning_rate: step size for parameter updates '''

#     def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
#         self.learning_rate = learning_rate
#         self.beta1 = beta1
#         self.beta2 = beta2
#         self.epsilon = epsilon
#         self.state = {} 
        
#     def update(self, layer):
#         if not hasattr(layer, 'W'): 
#             return
        
#         l_id = id(layer)
#         if l_id not in self.state:
#             self.state[l_id] = {
#                 'm_W': np.zeros_like(layer.grad_W), 
#                 'v_W': np.zeros_like(layer.grad_W),
#                 'm_b': np.zeros_like(layer.grad_b), 
#                 'v_b': np.zeros_like(layer.grad_b),
#                 't': 0
#             }

#         s = self.state[l_id]

#         # Retrieve current first and second moment estimates
#         m_W = s['m_W']
#         m_b = s['m_b']
#         v_W = s['v_W']
#         v_b = s['v_b']

#         # Update time step
#         s['t'] += 1
#         t = s['t']

#         # Update biased first and second moment estimates
#         m_W = self.beta1 * m_W + (1 - self.beta1) * layer.grad_W
#         m_b = self.beta1 * m_b + (1 - self.beta1) * layer.grad_b
#         v_W = self.beta2 * v_W + (1 - self.beta2) * (layer.grad_W ** 2)
#         v_b = self.beta2 * v_b + (1 - self.beta2) * (layer.grad_b ** 2)

#         # Compute bias-corrected first moment estimates with Nesterov lookahead
#         m_W_hat = self.beta1 * (m_W / (1 - self.beta1 ** t)) + ((1 - self.beta1) * layer.grad_W / (1 - self.beta1 ** t))
#         m_b_hat = self.beta1 * (m_b / (1 - self.beta1 ** t)) + ((1 - self.beta1) * layer.grad_b / (1 - self.beta1 ** t))

#         # Compute bias-corrected second moment estimates
#         v_W_hat = v_W / (1 - self.beta2 ** t)
#         v_b_hat = v_b / (1 - self.beta2 ** t)

#         # Update parameters using Nadam update rule
#         layer.W -= self.learning_rate * m_W_hat / (np.sqrt(v_W_hat) + self.epsilon)
#         layer.b -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

#         # Save updated first and second moment estimates back to state
#         s['m_W'] = m_W
#         s['m_b'] = m_b  
#         s['v_W'] = v_W
#         s['v_b'] = v_b