import sys
sys.path.append("../")
import numpy as np
from common.functions import cross_entropy_error

class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None

    def forward(self, x):
        W, = self.params
        out = np.dot(x, W)
        self.x = x
        return out

    def backward(self, dout):
        W, = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        self.grads[0][...] = dW
        return dx


class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        self.x = x
        return out

    def backward(self, dout):
        W, b = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0)
        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx

class Sigmoid:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx

class SigmoidWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = 1 / (1 + np.exp(-x))
        self.loss = cross_entropy_error(np.c_[1 - self.y, self.y], self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) * dout / batch_size
        return dx

# class Sign:
#     def __init__(self):
#         self.params, self.grads = [], []
#         self.out = None

#     def forward(self, x):
#         out = 1 if x >= 0 else -1
#         self.out = out
#         return out
        
#     def backward(self, dout):
#         return dout

# class SignWithLoss:
#     def __init__(self):
#         self.params, self.grads = [], []
#         self.loss = None
#         self.y = None
#         self.t = None

#     def forward(self, x, t):
#         self.t = t
#         self.y = 1 if x >= 0 else -1
#         self.loss = cross_entropy_error(np.c[1-self.y, self.y], self.t)
#         return self.loss
        
#     def backward(self, dout=1):
#         batch_size = self.t.shape[0]
#         dx = -self.t * dout
#         return dx