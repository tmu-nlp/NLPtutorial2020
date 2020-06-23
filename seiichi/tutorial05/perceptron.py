"""
single-layer perceptron
input: sentence from article of wikipedia
output: whether articles topic is about a person or not
"""

import os, sys
sys.path.append("../")
import math
import dill
import copy
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from common.layers import Affine, Sigmoid, SigmoidWithLoss
from common.optimizer import SGD

class Perceptron(object):
    def __init__(self, input_size, output_size):
        W1 = np.random.randn(input_size, output_size)
        b1 = np.random.randn(output_size)
        self.loss_layer = SigmoidWithLoss()
        self.layers = [Affine(W1, b1), Sigmoid()]
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads
        return 

    def predict(self, X):
        for layer in self.layers:
             X = layer.forward(X)
        return X

    def forward(self, X, y):
        score = self.predict(X)
        loss = self.loss_layer.forward(score, y)
        return loss
    
    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout


def binarization(y):
    return 1 if y == 1 else 0

def load_data(path, labeled=True):
    with open(path) as f:
        tmp = f.readlines()
    X, y = [], []
    vocab = set()
    for line in tmp:
        l = line.split()
        if labeled:
            label, sent = [binarization(int(l[0]))], l[1:]
        else:
            label, sent = None, l
        X.append(sent)
        y.append(label)
        for w in sent:
            vocab.add(w)
    return X, y, vocab

def remove_duplicate(params, grads):
    params, grads = params[:], grads[:]
    while True:
        find_flg = False
        L = len(params)
        for i in range(0, L - 1):
            for j in range(i + 1, L):
                if params[i] is params[j]:
                    grads[i] += grads[j] 
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)
                elif params[i].ndim == 2 and params[j].ndim == 2 and \
                     params[i].T.shape == params[j].shape and np.all(params[i].T == params[j]):
                    grads[i] += grads[j].T
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)
                if find_flg: break
            if find_flg: break
        if not find_flg: break
    return params, grads

def train_model(model, optimizer, X, y, batch_size=32, max_epoch=10):
    batch_size = batch_size
    data_size = len(X)
    max_epoch = max_epoch
    max_iters = data_size // batch_size
    total_loss = 0
    loss_count = 0
    for epoch in range(max_epoch):
        idx = np.random.permutation(np.arange(data_size))
        X = X[idx]
        y = y[idx]
        for iters in range(max_iters):
            batch_X = X[iters*batch_size:(iters+1)*batch_size]
            batch_y = y[iters*batch_size:(iters+1)*batch_size]
            loss = model.forward(batch_X, batch_y)
            model.backward()
            params, grads = remove_duplicate(model.params, model.grads)
            optimizer.update(params, grads)
            total_loss += loss
            loss_count += 1

        avg_loss = total_loss / loss_count    
        print("epoch {}, loss {:.3f}, ".format(epoch, avg_loss))
        total_loss, loss_count = 0, 0

if __name__ == '__main__':
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    # train = "../../test/03-train-input.txt"
    train = "../../data/titles-en-train.labeled"
    test = "../../data/titles-en-test.word"
    train_X, train_y, train_v = load_data(train)
    # test_X, test_y, test_v = load_data(test, labeled=False)
    vectorizer = TfidfVectorizer(max_df=0.9)
    # print(train_X)
    X = vectorizer.fit_transform(list(map(lambda x: " ".join(x), train_X)))
    words = vectorizer.get_feature_names()
    train_X = X.toarray()
    train_X, train_y = np.array(train_X).astype(np.float32), np.array(train_y).astype(np.int32)
    model = Perceptron(len(train_X[0]), 1)
    optimizer = SGD(lr=0.001)
    batch_size = 32
    max_epoch = 30
    train_model(model, optimizer, train_X, train_y, batch_size, max_epoch)
    pred_y = model.predict(train_X)
    print(accuracy_score(train_y, np.int32(pred_y >= 0.5)))
    print(confusion_matrix(train_y, np.int32(pred_y >= 0.5)))
