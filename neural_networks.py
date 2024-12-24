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


    def initialize_parameters(self):
        parameters = {}
        for i in range(self.num_layers):
            if i == 0:
                fan_in = self.input_size
                fan_out = self.architecture[i]
            else:
                fan_in = self.architecture[i-1]
                fan_out = self.architecture[i]

            parameters['W' + str(i + 1)] = np.random.randn(fan_out, fan_in)*0.01
            parameters['b' + str(i + 1)] = np.zeros((fan_out, 1))*0.01
        return parameters
    

    def linear(self, W, X, b):
        # Add gradient clipping in linear transformation
        z = np.dot(W, X) + b
        return np.clip(z, -1e10, 1e10)  # Prevent extreme values
    
    def activation(self, Z, activation):
        if activation == 'relu':
            g = np.maximum(0, Z)

        elif activation == "linear":
            g = Z
        return g

    def forward_propagation(self, inputs):
        caches = []
        A_prev = inputs
        
        for i in range(self.num_layers):
            W = self.parameters['W' + str(i + 1)]
            b = self.parameters['b' + str(i + 1)]
            
            Z = self.linear(W, A_prev, b)

            if i == self.num_layers-1:
                A = self.activation(Z=Z, activation='linear')
            else:
                A = self.activation(Z=Z, activation='relu')
                        
            fw_cache = (W, A_prev, Z)
            A_prev = A
            caches.append(fw_cache)
        
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
        m = Y.shape[1]
        loss = np.mean(np.power(y_pred - Y, 2), axis=1)/2

        return loss

   
    def back_propagation(self, y_pred, Y, caches):
        grads = {}
        m = Y.shape[1]
        
        
        
        for i in reversed(range(self.num_layers)):
            W, A_prev, Z = caches[i]

            if i == self.num_layers-1:
                dZ = (y_pred - Y) / m
            else:
                dZ = dA * (Z > 0)


            grads['dW' + str(i + 1)] = np.dot(dZ, A_prev.T)
            grads['db' + str(i + 1)] = np.mean(dZ, axis=1, keepdims=True)
            
            if i > 0:
                dA = np.dot(W.T, dZ)
                
        
        return grads

    def optimize(self, grads, learning_rate=0.001, tau= 0.001):
        for i in range(1, self.num_layers + 1):
            W_temp = self.parameters['W' + str(i)] - learning_rate * grads['dW' + str(i)]
            b_temp = self.parameters['b' + str(i)] - learning_rate * grads['db' + str(i)]
            self.parameters['W' + str(i)] = (1-tau)*self.parameters['W' + str(i)] + tau*W_temp
            self.parameters['b' + str(i)] = (1-tau)*self.parameters['b' + str(i)] + tau*b_temp


            # Clip weights after update
            # self.parameters['W' + str(i)] = np.clip(
            #     self.parameters['W' + str(i)], 
            #     -self.weight_clip_value, 
            #     self.weight_clip_value
            # )
   
    '''
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
        
        return loss'''
    
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


    def model_train(self, X_batch, Y_batch, learning_rate):
        self.iteration += 1
        caches, y_pred = self.forward_propagation(X_batch)
        loss = self.compute_loss(y_pred, Y_batch)
        
        # if np.isnan(loss):
        #     return loss
            
        grads = self.back_propagation(y_pred, Y_batch, caches)
        # grads = self.clip_gradients(grads)
        
        # Modified learning rate decay
        self.learning_rate = 0.01 * (0.95 ** (self.iteration // 200))
        self.optimize(grads, learning_rate=learning_rate)
        
        return loss
