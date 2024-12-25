import numpy as np
class NeuralNetwork():
    def __init__(self, input_size, architecture, activations, loss, class_weights=None):
        self.input_size = input_size
        self.architecture = architecture
        self.output_size = self.architecture[-1]
        self.activations = activations
        self.num_layers = len(self.architecture)
        self.loss = loss
        self.class_weights = class_weights
        self.iteration = 0
        self.parameters = self.initialize_parameters()
        self.grads = {}



    def initialize_parameters(self):
        parameters = {}
        for i in range(self.num_layers):
            if i == 0:
                fan_in = self.input_size
                fan_out = self.architecture[i]
            else:
                fan_in = self.architecture[i-1]
                fan_out = self.architecture[i]

            parameters['W' + str(i + 1)] = np.random.randn(fan_out, fan_in)*np.sqrt(2/fan_in)
            parameters['b' + str(i + 1)] = np.zeros((fan_out, 1))
        return parameters
    

    def linear(self, W, X, b):
        # Add gradient clipping in linear transformation
        z = np.dot(W, X) + b
        return np.clip(z, -1e10, 1e10)  # Prevent extreme values
    
    def activation(self, Z, activation):
        """
        Function Description: activation
        The activation function applies the specified activation function to the input Z, commonly used in neural networks.

        Parameters:
        Z (numpy array): Input array, typically the pre-activation output of a neural network layer.
        activation (string): The activation function to apply. Supported options:
        - 'ReLU': Rectified Linear Unit (max(0, Z)).
        - 'Leaky_ReLU': Allows small gradients for negative values (Z > 0 ? Z : 0.01 * Z).
        - 'ELU': Smooths negative values (Z > 0 ? Z : 0.1 * (exp(Z) - 1)).
        - 'Linear': Linear activation (Z).
        - 'Sigmoid': Squeezes values to [0, 1] (1 / (1 + exp(-Z))).
        - 'Softmax': Converts logits to probabilities, with numerical stability (exp(Z - max(Z)) / sum(exp(Z))).
        Returns:
        - a (numpy array): Output after applying the activation function.
        Notes:
        Raises ValueError for unsupported activation names.
        Softmax implementation ensures numerical stability.
        """

        if activation == 'ReLU':
            a = np.maximum(0, Z)
            grad = (Z > 0).astype(float)  # Derivative: 1 for Z > 0, 0 otherwise
        
        elif activation == "Leaky_ReLU":
            a = np.where(Z > 0, Z, 0.01 * Z)
            grad = np.where(Z > 0, 1, 0.01)  # Derivative: 1 for Z > 0, 0.01 otherwise
        
        elif activation == "ELU":
            alpha = 0.1
            a = np.where(Z > 0, Z, alpha * (np.exp(Z) - 1))
            grad = np.where(Z > 0, 1, alpha * np.exp(Z))  # Derivative: exp(Z) * alpha for Z <= 0

        elif activation == "Linear":
            a = Z
            grad = np.ones_like(Z)  # Derivative: 1 everywhere

        elif activation == 'Sigmoid':
            a = 1/(1+ np.exp(-Z))
            grad = a * (1 - a)  # Derivative: sigmoid(Z) * (1 - sigmoid(Z))

        elif activation == 'Softmax':
            exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            a = exp_Z/np.sum(exp_Z, axis=0, keepdims=True)
            grad = a * (1 - a)  # Jacobian diagonal approximation for simplicity

        else:
            raise ValueError(f"Unknown activation function: {activation}")
        activation_cache = (a, grad)
        return a, activation_cache
    

    def clip_gradients(self, grads, threshold=1.0):
        for key in grads:
            grads[key] = np.clip(grads[key], -threshold, threshold)
        return grads

    def forward_propagation(self, inputs):
        caches = []
        A_prev = inputs
        
        for i in range(self.num_layers):
            W = self.parameters['W' + str(i + 1)]
            b = self.parameters['b' + str(i + 1)]
            
            Z = self.linear(W, A_prev, b)

            
            A, activation_cache = self.activation(Z=Z, activation=self.activations[i])
            
                        
            fw_cache = (W, A_prev, Z)
            A_prev = A
            cache = (fw_cache, activation_cache)
            caches.append(cache)
        
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
        if self.loss == "MSE":
            loss = np.mean((y_pred - Y) ** 2, axis=1) / 2
        elif self.loss == "BCE":
            loss = -np.mean(Y * np.log(y_pred) + (1 - Y) * np.log(1 - y_pred), axis=1)
        elif self.loss == "CCE":
            # loss = -np.mean(np.sum(Y * np.log(y_pred), axis=1))
            if self.class_weights is not None:
                weighted_loss = -np.sum(self.class_weights * Y * np.log(y_pred + 1e-15), axis=1)
                loss = np.mean(weighted_loss)  # Average loss over batch
            else:
                loss = -np.mean(np.sum(Y * np.log(y_pred + 1e-15), axis=1))  # divide by batch size
        # loss = np.mean(np.power(y_pred - Y, 2), axis=1)/2

        return loss

   
    def back_propagation(self, y_pred, Y, caches):
        # grads = {}
        m = Y.shape[1]
        
        
        
        for i in reversed(range(self.num_layers)):
            forward_cache, activation_cache = caches[i]
            W, A_prev, Z = forward_cache
            A, activation_grad = activation_cache
            if i == self.num_layers-1:
                if self.loss == "MSE":
                    dZ = (y_pred - Y) / m
                elif self.loss == "BCE":
                    dZ = y_pred - Y
                elif self.loss == "CCE":
                    dZ = (y_pred - Y)/m
            else:
                # dZ = dA * (Z > 0) #for relu
               
                dZ = dA * activation_grad


            self.grads['dW' + str(i + 1)] = np.dot(dZ, A_prev.T)
            self.grads['db' + str(i + 1)] = np.sum(dZ, axis=1, keepdims=True)
            
            
            if i > 0:
                dA = np.dot(W.T, dZ)
                
        


    def optimize(self, grads, learning_rate=0.001, tau= 0.1):
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
        
        if self.loss == "BCE":  # Binary classification
            # Apply thresholding to sigmoid output
            predictions = (y_pred > threshold).astype(int)
        elif self.loss == "CCE":  # Multi-class classification
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
        # assert y_pred.shape == Y.shape

        loss = self.compute_loss(y_pred, Y)  # Compute loss using probabilities
        predictions = self.predict(X)  # Get discrete predictions
        return loss, predictions


    def grad_check(self):
        for key, val in self.parameters:
            print(key)
            print(val)


    def model_train(self, X_batch, Y_batch, learning_rate, decay_rate=0.01):
        self.iteration += 1
        caches, y_pred = self.forward_propagation(X_batch)
        loss = self.compute_loss(y_pred, Y_batch)
        
            
        self.back_propagation(y_pred, Y_batch, caches)
    
        learning_rate = learning_rate / (1 + decay_rate * self.iteration)

        self.grads = self.clip_gradients(self.grads, threshold=1.0)

        self.optimize(self.grads, learning_rate=learning_rate)
        
        return loss
    

