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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


class SVM:
    def __init__(self, vocab, output_size=1, m=0.0005, c=0.0001, lr=1):
        self.M = m
        self.C = c
        self.lr = lr
        self.unk = "<UNK>"
        # self.Phi = dict()
        self.W = dict()
        for v in vocab:
            # self.Phi[v] = 0
            self.W[v] = 0
        # self.Phi[self.unk] = 0
        self.W[self.unk] = 0
        input_size = len(vocab) + 1
        return 

    def forward(self, X, y):
        phi = self.create_feat(X)
        y_hat = self.predict(X)
        for i in range(len(y)):
            self.normalize(X[i])
            if y_hat[i] * y[i] <= self.M:
                self.update(phi[i], y[i])
        return self

    def create_feat(self, X):
        ret = []
        for X_i in X:
            phi = defaultdict(lambda: 0)
            for w in X_i:
                phi[w] += 1
            ret.append(phi)
        return ret

    def normalize(self, need):
        for word in need:
            if word not in self.W.keys():
                continue
            val = self.W[word]
            c = 0 if abs(val) < self.C else self._sign(val) * self.C
            self.W[word] -= c
        return self

    def _sign(self, val):
        if val == 0: return val
        return 1 if val > 0 else -1

    def _get_margin(self):
        return 1 / np.sqrt(np.linalg.norm(list(self.W.values())))

    def update(self, phi_i, y_i):
        for word, val in phi_i.items():
            if word in self.W.keys():
                self.W[word] += val * y_i * self.lr
        return self
        
    def train(self, X, y, itr=10):
        for i in tqdm(range(itr)):
            self.forward(X, y)
        return self
    
    def predict(self, X):
        phi = self.create_feat(X)
        pred = []
        for i in range(len(phi)):
            score = 0
            for word, val in phi[i].items():
                if word in self.W.keys():
                    score += val * self.W[word]
            pred.append(1 if score>=0 else -1)
        return pred

    def report_acc(self, y, pred_y):
        return accuracy_score(y, pred_y)
    
    def save(self, path="./model/svm.model"):
        with open(path, "wb") as f:
            dill.dump(self.W, f)
    
    def load(self, path="./model/svm.model"):
        with open(path, "rb") as f:
            self.W = dill.load(f)


def load_data(path, labeled=True):
    with open(path) as f:
        tmp = f.readlines()
    X, y = [], []
    vocab = set()
    for line in tmp:
        l = line.split()
        if labeled:
            label, sent = int(l[0]), l[1:]
        else:
            label, sent = None, l
        X.append(sent)
        y.append(label)
        for w in sent:
            vocab.add(w)
    return X, y, vocab

if __name__ == '__main__':
    # from sklearn.feature_extraction.text import TfidfVectorizer
    # train = "../../test/03-train-input.txt"
    train = "../../data/titles-en-train.labeled"
    test = "../../data/titles-en-test.labeled"
    train_X, train_y, train_v = load_data(train)
    test_X, test_y, test_v = load_data(test)
    # vectorizer = TfidfVectorizer(max_df=0.9)
    # X = vectorizer.fit_transform(np.array(train_X).flatten())
    # words = vectorizer.get_feature_names()
    model = SVM(train_v)
    model.train(train_X, train_y, itr=30).save()
    model.load()
    pred_y = model.predict(train_X)
    pred_test_y = model.predict(test_X)
    print(accuracy_score(test_y, pred_test_y))
    print(confusion_matrix(test_y, pred_test_y))
    print(classification_report(test_y, pred_test_y))
    with open("./result/wikien.result", "w") as f:
        for i, p in enumerate(pred_test_y):
            f.write(str(p)+" "+" ".join(test_X[i])+"\n")


"""result
0.8738930216082182
[[1267  210]
 [ 146 1200]]
              precision    recall  f1-score   support

          -1       0.90      0.86      0.88      1477
           1       0.85      0.89      0.87      1346

    accuracy                           0.87      2823
   macro avg       0.87      0.87      0.87      2823
weighted avg       0.87      0.87      0.87      2823
"""