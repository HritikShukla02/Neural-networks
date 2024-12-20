import numpy as np
class NeuralNetwork():
    def __init__(self, input_size, output_size, architecture):
        self.input_size = input_size
        self.output_size = output_size
        self.architecture = architecture
        self.num_layers  = len(self.architecture)
        self.parameters = self.initialize_parameters()
        

    def linear(self, W, X, b):
        z = np.sum(np.dot(W, X), b)
        return z
    

    def activation(self, Z, activation):

        if activation == 'relu':
            g = np.maximum(Z,0)

        elif activation == 'sigmoid':
            g = 1/(1 + np.exp(-Z))

        elif activation == 'softmax':
            # Subtract max for numerical stability
            Z_stable = Z - np.max(Z, axis=1, keepdims=True)
            exp = np.exp(Z_stable)
            g = exp / np.sum(exp, axis=1, keepdims=True)

        else:
            g = Z
        
        cache = (Z, g)
        return cache, g
    

    def initialize_parameters(self):
        """
        Initializes weights and biases for all layers in the network.

        - Weights are initialized with small random values (scaled by 0.01) for better convergence.
        - Biases are initialized to zeros.
        """
        parameters = {}
        # num_layers = len(self.architecture)

        for i in range(self.num_layers):
            if i == 0:
                # Initialize weights for the first layer
                parameters['W' + str(i + 1)] = np.random.randn(self.architecture[i], self.input_size) * 0.01
            else:
                # Initialize weights for subsequent layers
                parameters['W' + str(i + 1)] = np.random.randn(self.architecture[i], self.architecture[i - 1]) * 0.01
            
            # Initialize biases for all layers
            parameters['b' + str(i + 1)] = np.zeros((self.architecture[i], 1))


    def forward_propagation(self, inputs):
        """
        Perform forward propagation through all layers of the network.

        Args:
        - inputs: Input data, shape (input_size, number of examples).

        Returns:
        - caches: List of caches containing intermediate computations for backpropagation.
        - output: Final activation output from the last layer.
        """
        caches = []  # To store intermediate computations for backpropagation
        A_prev = inputs  # Input data for the first layer

        for i in range(self.num_layers):
            W = self.parameters['W' + str(i + 1)]
            b = self.parameters['b' + str(i + 1)]
            
            # Linear step
            Z = self.linear(W, A_prev, b)
            
            # Activation step
            if i == self.num_layers - 1:  # Output layer
                if self.architecture[-1] > 1:
                    activation_cache, A = self.activation(Z, 'softmax')  # Multi-class output
                else:
                    activation_cache, A = self.activation(Z, 'sigmoid')  # Binary output
            else:  # Hidden layers
                activation_cache, A = self.activation(Z, 'relu')
            # Store caches
            fw_cache = (W, b, A, A_prev, Z)
            A_prev = A
            caches.append((fw_cache, activation_cache))
        
        return caches, A


    def compute_loss(self, y_pred, Y):
        """
        Compute the loss based on the network's architecture.

        Args:
        - y_pred: Predicted probabilities/output from the model.
        - Y: True labels (integer for sparse categorical, 0/1 for binary).

        Returns:
        - loss: Computed loss value.
        """
        epsilon = 1e-15  # Small value to avoid numerical instability
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clip predictions to avoid log(0)
        
        if self.architecture[-1] > 1:  # Sparse Categorical Cross-Entropy
            # Ensure Y is integer-encoded
            loss = -np.mean(np.log(y_pred[Y, np.arange(len(Y))]))
        
        elif self.architecture[-1] == 1:  # Binary Cross-Entropy
            loss = -np.mean(Y * np.log(y_pred) + (1 - Y) * np.log(1 - y_pred))
        
        else:
            raise ValueError("Invalid architecture configuration.")
        
        return loss
    
    
    def back_propagation(self, y_pred, Y, loss, cache):
        grads = {}

        m = Y.shape[1]
        dZ  = np.subtract(y_pred, Y)    

        for i in reversed(range(len(self.num_layers))):
            fw_cache, act_cache = cache[i]
            W, b, A, A_prev, Z = fw_cache


            grads['dW'+str(i+1)] = np.dot(dZ, A_prev.T)/m
            grads['db'+str(i+1)] = np.sum(dZ, axis=1, keepdims=True)/m
            
            if i > 0:
                dZ = np.dot(W, dZ)*np.where(Z > 0, 1, 0)

        return grads

        
def optimize(self, grads, learning_rate=0.001, epsilon=0.001):
    """
    Update network parameters using gradient descent with optional weight decay.

    Args:
    - grads: Dictionary containing gradients 'dW' and 'db' for each layer.
    - learning_rate: Learning rate for gradient descent (default: 0.001).
    - epsilon: Weight decay factor (default: 0.001).

    Returns:
    - None (updates self.parameters in place).
    """
    for i in range(1, self.num_layers + 1):
        # Ensure gradients and parameters are compatible
        assert self.parameters['W' + str(i)].shape == grads['dW' + str(i)].shape, \
            f"Shape mismatch in W{i} and dW{i}"
        assert self.parameters['b' + str(i)].shape == grads['db' + str(i)].shape, \
            f"Shape mismatch in b{i} and db{i}"
        
        # Update weights with gradient descent
        W_update = self.parameters['W' + str(i)] - learning_rate * grads['dW' + str(i)]
        b_update = self.parameters['b' + str(i)] - learning_rate * grads['db' + str(i)]
        
        # Apply weight decay (optional)
        self.parameters['W' + str(i)] = self.parameters['W' + str(i)] * (1 - epsilon) + W_update
        self.parameters['b' + str(i)] = self.parameters['b' + str(i)] * (1 - epsilon) + b_update







            





            
