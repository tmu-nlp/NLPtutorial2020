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

class RNN():
    def __init__(self, input_dim, hidden_dim, output_dim, weights_init="xavier"):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        if weights_init=="xavier":
            np.random.seed(1)
            self.weights_rx = np.random.randn(self.hidden_dim, self.input_dim) / np.sqrt(self.input_dim)
            np.random.seed(1)
            self.weights_rh = np.random.randn(self.hidden_dim, self.hidden_dim) / np.sqrt(self.hidden_dim)
            np.random.seed(1)
            self.bias_r = np.random.randn(self.hidden_dim) / np.sqrt(self.input_dim)
            np.random.seed(1)
            self.weights_oh = np.random.randn(self.output_dim, self.hidden_dim) / np.sqrt(self.hidden_dim)
            np.random.seed(1)
            self.bias_o = np.random.randn(self.output_dim) / np.sqrt(self.hidden_dim)
        if weights_init=="random":
            self.weights_rx = (np.random.rand(self.hidden_dim, self.input_dim) - 0.5) / 5
            self.weights_rh = (np.random.rand(self.hidden_dim, self.hidden_dim) - 0.5) / 5
            self.bias_r = (np.random.rand(self.hidden_dim) - 0.5) / 5
            self.weights_oh = (np.random.rand(self.output_dim, self.hidden_dim) - 0.5) / 5
            self.bias_o = (np.random.rand(self.output_dim) - 0.5) / 5
    
    def __call__(self, inputs):
        return self.forward(inputs)
    
    def forward(self, inputs):
        self.inputs = inputs
        h, p, y = [], [], []
        for t in range(len(self.inputs)):
            if t > 0:
                h.append(np.tanh(np.dot(self.weights_rx, self.inputs[t]) \
                    + np.dot(self.weights_rh, h[t-1]) + self.bias_r))
            else:
                h.append(np.tanh(np.dot(self.weights_rx, self.inputs[t]) + self.bias_r))
            p.append(np.tanh(np.dot(self.weights_oh, h[t]) + self.bias_o))
            y.append(np.argmax(p[t]))
        self.h, self.p, self.y = h, p, y
        return h, p, y
    
    def backward(self, labels):
        self.delta_w_rx = np.zeros((self.hidden_dim, self.input_dim))
        self.delta_w_rh = np.zeros((self.hidden_dim, self.hidden_dim))
        self.delta_b_r = np.zeros(self.hidden_dim)
        self.delta_w_oh = np.zeros((self.output_dim, self.hidden_dim))
        self.delta_b_o = np.zeros(self.output_dim)
        delta_r_prime = np.zeros(len(self.bias_r))
        for t in list(range(len(self.inputs)))[::-1]:
            delta_o_prime = labels[t] - self.p[t]
            #self.delta_w_oh += np.outer(self.h[t], delta_o_prime)
            self.delta_w_oh += np.outer(delta_o_prime, self.h[t])
            self.delta_b_o += delta_o_prime
            delta_r = np.dot(delta_r_prime, self.weights_rh) + np.dot(delta_o_prime, self.weights_oh)
            delta_r_prime = delta_r * (1 - self.h[t]**2)
            #self.delta_w_rx += np.outer(self.inputs[t], delta_r_prime)
            self.delta_w_rx += np.outer(delta_r_prime, self.inputs[t])
            self.delta_b_r += delta_r_prime
            if t != 0:
                #self.delta_w_rh += np.outer(self.h[t-1], delta_r_prime)
                self.delta_w_rh += np.outer(delta_r_prime, self.h[t-1])
    
    def update_weights(self, lr=0.001):
        self.weights_rx += lr * self.delta_w_rx
        self.weights_rh += lr * self.delta_w_rh
        self.bias_r += lr * self.delta_b_r
        self.weights_oh += lr * self.delta_w_oh
        self.bias_o += lr * self.delta_b_o
