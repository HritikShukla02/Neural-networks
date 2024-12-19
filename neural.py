import numpy as np
class NeuralNetwork():
    def __init__(self, input_size, output_size, architecture):
        self.input_size = input_size
        self.output_size = output_size
        self.architecture = architecture
        self.num_layers  = len(self.architecture)
        

    def linear(self, W, X, b):
        z = np.sum(np.dot(W, X), b)
        return z
    

    def activation(self, Z, activation):

        if activation == 'relu':
            g = np.maximum(Z,0)

        elif activation == 'sigmoid':
            g = 1/(1 + np.exp(-1*Z))

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
        self.parameters = {}
        # num_layers = len(self.architecture)

        for i in range(self.num_layers):
            if i == 0:
                # Initialize weights for the first layer
                self.parameters['W' + str(i + 1)] = np.random.randn(self.architecture[i], self.input_size) * 0.01
            else:
                # Initialize weights for subsequent layers
                self.parameters['W' + str(i + 1)] = np.random.randn(self.architecture[i], self.architecture[i - 1]) * 0.01
            
            # Initialize biases for all layers
            self.parameters['b' + str(i + 1)] = np.zeros((self.architecture[i], 1))


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
        A = inputs  # Input data for the first layer

        for i in range(self.num_layers):
            W = self.parameters['W' + str(i + 1)]
            b = self.parameters['b' + str(i + 1)]
            
            # Linear step
            Z = self.linear(W, A, b)
            
            # Activation step
            if i == self.num_layers - 1:  # Output layer
                if self.architecture[-1] > 1:
                    activation_cache, A = self.activation(Z, 'softmax')  # Multi-class output
                else:
                    activation_cache, A = self.activation(Z, 'sigmoid')  # Binary output
            else:  # Hidden layers
                activation_cache, A = self.activation(Z, 'relu')

            # Store caches
            fw_cache = (W, b, A, Z)
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
    
    





            





            
