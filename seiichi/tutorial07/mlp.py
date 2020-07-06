"""
multi-layer perceptron
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
from common.layers import Affine, Tanh, TanhWithLoss
from common.optimizer import SGD, Adam


class MLP(object):
    def __init__(self, input_size, hidden_size, output_size):
        W1 = 0.01 * np.random.randn(input_size, hidden_size)
        b1 = np.zeros(hidden_size)
        W2 = 0.01 * np.random.randn(hidden_size, output_size)
        b2 = np.zeros(output_size)
        self.layers = [Affine(W1, b1), Tanh(), Affine(W2, b2)]
        self.loss_layer = TanhWithLoss()
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


def load_data(path, labeled=True):
    with open(path) as f:
        tmp = f.readlines()
    X, y = [], []
    vocab = set()
    for line in tmp:
        l = line.split()
        if labeled:
            label, sent = [int(l[0])], l[1:]
        else:
            label, sent = None, l
        X.append(sent)
        y.append(label)
        for w in sent:
            vocab.add(w)
    return X, y, vocab

def train_model(model, optimizer, X, y, batch_size=32, max_epoch=10):
    batch_size = batch_size
    data_size = len(X)
    max_epoch = max_epoch
    max_iters = data_size // batch_size
    for epoch in range(max_epoch):
        idx = np.random.permutation(np.arange(data_size))
        X = X[idx]
        y = y[idx]
        total_loss = 0
        loss_count = 0
        for iters in range(max_iters):
            batch_X = X[iters*batch_size:(iters+1)*batch_size]
            batch_y = y[iters*batch_size:(iters+1)*batch_size]
            loss = model.forward(batch_X, batch_y)
            # pred_y = model.predict(batch_X)
            model.backward()
            optimizer.update(model.params, model.grads)
            total_loss += loss
            loss_count += 1
            # from sklearn.metrics import accuracy_score
            # print(accuracy_score(batch_y, np.int32(sigmoid(pred_y)>=0)))
        avg_loss = total_loss / loss_count    
        print("epoch {}, loss {:.3f}, ".format(epoch, avg_loss))

def get_count(vectorizer, train_X):
    X = vectorizer.transform(list(map(lambda x: " ".join(x), train_X)))

def get_tdidf(vectorizer, train_X):
    X = vectorizer.transform(list(map(lambda x: " ".join(x), train_X)))
    return X.toarray()

def get_reduced(svd, train_X):
    X = svd.transform(train_X)
    return X

def activate(y):
    tmp = np.int32(y >= 0)
    return np.where(tmp == 0, -1, tmp)


if __name__ == '__main__':
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.decomposition import TruncatedSVD
    from scipy.linalg import svd
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    # train = "../../test/03-train-input.txt"
    train = "../../data/titles-en-train.labeled"
    test = "../../data/titles-en-test.labeled"
    train_X, train_y, train_v = load_data(train)
    test_X, test_y, test_v = load_data(test)
    v = TfidfVectorizer(max_df=0.8)
    v.fit(list(map(lambda x: " ".join(x), train_X)))
    train_X, test_X = get_tdidf(v, train_X), get_tdidf(v, test_X)
    svd = TruncatedSVD(n_components=200, random_state=3939)
    svd.fit(train_X)
    train_X, test_X = get_reduced(svd, train_X), get_reduced(svd, test_X)
    train_X, train_y = np.array(train_X).astype(np.float32), np.array(train_y).astype(np.int32)
    test_X, test_y = np.array(test_X).astype(np.float32), np.array(test_y).astype(np.int32)
    # print(train_X, train_y)
    model = MLP(len(train_X[0]), len(train_X[0])//2, 1)
    # optimizer = SGD(lr=0.01)
    optimizer = Adam(lr=0.01)
    batch_size = 32
    max_epoch = 50
    train_model(model, optimizer, train_X, train_y, batch_size, max_epoch)
    pred_y = model.predict(test_X)
    pred_y = np.tanh(pred_y)
    print(accuracy_score(test_y, activate(pred_y)))
    print(confusion_matrix(test_y, activate(pred_y)))
    print(classification_report(test_y, activate(pred_y)))

"""result
#1 SGD(lr=0.5)
0.9341126461211477
[[1406   71]
 [ 115 1231]]
              precision    recall  f1-score   support

          -1       0.92      0.95      0.94      1477
           1       0.95      0.91      0.93      1346

    accuracy                           0.93      2823
   macro avg       0.93      0.93      0.93      2823
weighted avg       0.93      0.93      0.93      2823

#2 SGD(lr=0.01)
0.9284449167552249
[[1399   78]
 [ 124 1222]]
              precision    recall  f1-score   support

          -1       0.92      0.95      0.93      1477
           1       0.94      0.91      0.92      1346

    accuracy                           0.93      2823
   macro avg       0.93      0.93      0.93      2823
weighted avg       0.93      0.93      0.93      2823

#3 Adam(lr=0.01)
0.9404888416578109
[[1389   88]
 [  80 1266]]
              precision    recall  f1-score   support

          -1       0.95      0.94      0.94      1477
           1       0.94      0.94      0.94      1346

    accuracy                           0.94      2823
   macro avg       0.94      0.94      0.94      2823
weighted avg       0.94      0.94      0.94      2823
"""