import numpy as np
class NeuralNetwork():
    def __init__(self, input_size, output_size, architecture, loss):
        self.input_size = input_size
        self.output_size = output_size
        self.architecture = architecture
        self.num_layers = len(self.architecture)
        self.loss = loss
        self.parameters = self.initialize_parameters()
        self.iteration = 0
        self.learning_rate = 0.01  # Increased initial learning rate
        self.weight_clip_value = 5.0
        self.grad_clip_value = 1.0

    def linear(self, W, X, b):
        # Add gradient clipping in linear transformation
        z = np.dot(W, X) + b
        return np.clip(z, -1e10, 1e10)  # Prevent extreme values
    
    def activation(self, Z, activation):
        if activation == 'relu':
            # Leaky ReLU instead of regular ReLU to prevent dying neurons
            g = np.maximum(0.01 * Z, Z)
        elif activation == 'sigmoid':
            # Clip values to prevent overflow
            Z = np.clip(Z, -500, 500)
            g = 1/(1 + np.exp(-Z))
        else:
            g = Z
        return g
    

    def initialize_parameters(self):
        parameters = {}
        for i in range(self.num_layers):
            if i == 0:
                fan_in = self.input_size
                fan_out = self.architecture[i]
            else:
                fan_in = self.architecture[i-1]
                fan_out = self.architecture[i]
            # Xavier initialization
            limit = np.sqrt(6 / (fan_in + fan_out))
            parameters['W' + str(i + 1)] = np.random.uniform(-limit, limit, (self.architecture[i], fan_in))
            parameters['b' + str(i + 1)] = np.zeros((self.architecture[i], 1))
        return parameters

    
    def clip_gradients(self, grads):
        for key in grads:
            np.clip(grads[key], -self.grad_clip_value, self.grad_clip_value, out=grads[key])
        return grads
        
    def layer_normalize(self, x, epsilon=1e-8):
        mean = np.mean(x, axis=1, keepdims=True)
        var = np.var(x, axis=1, keepdims=True)
        return (x - mean) / np.sqrt(var + epsilon)



    def forward_propagation(self, inputs):
        caches = []
        A_prev = inputs
        
        for i in range(self.num_layers):
            W = np.clip(self.parameters['W' + str(i + 1)], -self.weight_clip_value, self.weight_clip_value)
            b = self.parameters['b' + str(i + 1)]
            
            Z = np.dot(W, A_prev) + b
            
            # Only apply batch normalization to hidden layers
            if i < self.num_layers - 1:
                Z = self.batch_normalize(Z)
                A = np.maximum(0.01 * Z, Z)  # Leaky ReLU
            else:
                A = Z  # Linear activation for output layer
            
            fw_cache = (W, b, A_prev, Z)
            A_prev = A
            caches.append(fw_cache)
            self.parameters['W' + str(i + 1)] = W
        
        return caches, A


    def batch_normalize(self, x, gamma=1.0, beta=0.0, epsilon=1e-8):
        mu = np.mean(x, axis=1, keepdims=True)
        var = np.var(x, axis=1, keepdims=True)
        x_norm = (x - mu) / np.sqrt(var + epsilon)
        return gamma * x_norm + beta
    

    def back_propagation(self, y_pred, Y, caches):
        grads = {}
        m = Y.shape[1]
        
        dZ_next = np.clip((y_pred - Y) / m, -self.grad_clip_value, self.grad_clip_value)
        
        for i in reversed(range(self.num_layers)):
            W, b, A_prev, Z = caches[i]
            
            grads['dW' + str(i + 1)] = np.clip(np.dot(dZ_next, A_prev.T), -self.grad_clip_value, self.grad_clip_value)
            grads['db' + str(i + 1)] = np.clip(np.sum(dZ_next, axis=1, keepdims=True), -self.grad_clip_value, self.grad_clip_value)
            
            if i > 0:
                dA = np.dot(W.T, dZ_next)
                dZ_next = np.clip(dA * (Z > 0), -self.grad_clip_value, self.grad_clip_value)
        
        return grads

    def optimize(self, grads, learning_rate=0.001):
        for i in range(1, self.num_layers + 1):
            self.parameters['W' + str(i)] -= learning_rate * grads['dW' + str(i)]
            self.parameters['b' + str(i)] -= learning_rate * grads['db' + str(i)]
            
            # Clip weights after update
            self.parameters['W' + str(i)] = np.clip(
                self.parameters['W' + str(i)], 
                -self.weight_clip_value, 
                self.weight_clip_value
            )
   

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
        m = Y.shape[1]
        
        if self.loss == "Sparse Categorical Cross-Entropy":  # Sparse Categorical Cross-Entropy
            # Ensure Y is a 1D array of integers (true class indices)
            if Y.ndim > 1:
                Y = Y.flatten()  # Flatten if Y is a column vector or higher-dimensional array

            # Compute the sparse categorical cross-entropy loss
            loss = -np.mean(np.log(y_pred[Y, np.arange(m)]))

        
        elif self.loss == "Binary Cross-Entropy":  # Binary Cross-Entropy
            loss = -np.mean(Y * np.log(y_pred) + (1 - Y) * np.log(1 - y_pred))

        elif self.loss == "MSE":
            loss = np.mean(np.power(y_pred-Y, 2))/2     
        else:
            raise ValueError("Invalid architecture configuration.")
        
        return loss
    
    '''
    def back_propagation(self, y_pred, Y, caches):
        grads = {}
        m = Y.shape[1]
        
        # Initial dZ calculation based on loss type
        if self.loss == "MSE":
            dZ = y_pred - Y
        elif self.loss == "Sparse Categorical Cross-Entropy":
            Y_one_hot = np.zeros_like(y_pred)
            Y_one_hot[Y, np.arange(m)] = 1
            dZ = y_pred - Y_one_hot
        elif self.loss == "Binary Cross-Entropy":
            dZ = y_pred - Y
        
        for i in reversed(range(self.num_layers-1)):  # Changed from self.num_layers
            cache = caches[i]
            W, b, A_prev, Z = cache
            
            grads['dW' + str(i + 1)] = np.dot(dZ, A_prev.T) / m
            grads['db' + str(i + 1)] = np.sum(dZ, axis=1, keepdims=True) / m
            
            if i > 0:  
                dA = np.dot(W.T, dZ)
                dZ = dA * (Z > 0)
        
        return grads'''
    





    def predict(self, X, threshold=0.5):
        """
        Predict output labels for given input data.

        Args:
        - X: Input data, shape (input_size, number of examples).

        Returns:
        - predictions: Predicted labels, shape (1, number of examples) for binary,
        or (number of examples,) for multi-class.
        """
        _, y_pred = self.forward_propagation(X)
        
        if self.loss == "Binary Cross-Entropy":  # Binary classification
            # Apply thresholding to sigmoid output
            predictions = (y_pred > threshold).astype(int)
        elif self.loss == "Sparse Categorical Cross-Entropy":  # Multi-class classification
            # Select class with highest probability for each example
            predictions = np.argmax(y_pred, axis=0)
        elif self.loss == "MSE":
            predictions = y_pred
        
        return predictions


    def evaluate(self, X, Y):
        """
        Evaluate the model's performance on given input data and labels.

        Args:
        - X: Input data, shape (input_size, number of examples).
        - Y: True labels, shape (1, number of examples) for binary,
            or (number of examples,) for multi-class.

        Returns:
        - loss: Computed loss on the dataset.
        - predictions: Predicted labels (binary or multi-class).
        """
        caches, y_pred = self.forward_propagation(X)  # Get raw probabilities
        loss = self.compute_loss(y_pred, Y)  # Compute loss using probabilities
        predictions = self.predict(X)  # Get discrete predictions
        return loss, predictions


    def normalize_gradients(self, grads):
        total_norm = 0
        for key in grads:
            total_norm += np.sum(np.square(grads[key]))
        total_norm = np.sqrt(total_norm)
        
        if total_norm > 1:
            for key in grads:
                grads[key] = grads[key] / total_norm
        return grads


    def decay_learning_rate(self):
        self.learning_rate = self.learning_rate / (1 + 0.0001 * self.iteration)


    def clip_gradients(self, grads):
        for key in grads:
            np.clip(grads[key], -self.grad_clip_value, self.grad_clip_value, out=grads[key])
        return grads

    def debug_check(self, tensor, name):
        if np.any(np.isnan(tensor)) or np.any(np.abs(tensor) > 1e10):
            print(f"Warning: {name} contains NaN or extreme values")
            print(f"Max value: {np.max(np.abs(tensor))}")
            return False
        return True
    

    def layer_normalize(self, x, epsilon=1e-8):
        mean = np.mean(x, axis=1, keepdims=True)
        var = np.var(x, axis=1, keepdims=True)
        return (x - mean) / np.sqrt(var + epsilon)
    

    def model_train(self, X_batch, Y_batch):
        self.iteration += 1
        caches, y_pred = self.forward_propagation(X_batch)
        loss = self.compute_loss(y_pred, Y_batch)
        
        if np.isnan(loss):
            return loss
            
        grads = self.back_propagation(y_pred, Y_batch, caches)
        grads = self.clip_gradients(grads)
        
        # Modified learning rate decay
        self.learning_rate = 0.01 * (0.95 ** (self.iteration // 200))
        self.optimize(grads, learning_rate=self.learning_rate)
        
        return loss
