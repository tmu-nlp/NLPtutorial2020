import numpy as np

class Linear():
    def __init__(self, input_dim, output_dim, weights_init="xavier"):
        self.input_dim = input_dim
        self.output_dim = output_dim
        if weights_init=="xavier":
            self.weights = np.random.randn(self.output_dim, self.input_dim) / np.sqrt(self.input_dim)
            self.bias = np.random.randn(self.output_dim) / np.sqrt(self.input_dim)
        if weights_init=="random":
            self.weights = (np.random.rand(self.output_dim, self.input_dim) - 0.5) / 5
            self.bias = (np.random.rand(self.output_dim) - 0.5) / 5
    
    def __call__(self, inputs):
        return self.forward(inputs)
    
    def forward(self, inputs):
        self.inputs = inputs
        self.score = np.tanh(np.dot(self.weights, self.inputs) + self.bias)
        return self.score
    
    def backward(self, delta):
        self.delta_prime = delta * (1 - self.score**2)
        delta_back = np.dot(self.delta_prime, self.weights)
        return delta_back
    
    def update_weights(self, delta_prime, lr=0.001):
        self.lr = lr
        self.weights += self.lr * np.outer(self.delta_prime, self.inputs)
        self.bias += self.lr * self.delta_prime


'''
class nn():
    def __init__(self, input_dim, output_dim, weights_init="xavier"):
        self.input_dim = input_dim
        self.output_dim = output_dim
        if weights_init=="xavier":
            self.weights = np.random.randn(self.output_dim, self.input_dim) / np.sqrt(self.input_dim)
        if weights_init=="random":
            self.weights = np.random.rand(self.output_dim, self.input_dim) – 0.5
        self.bias = -np.ones(self.output_dim, 1)
'''