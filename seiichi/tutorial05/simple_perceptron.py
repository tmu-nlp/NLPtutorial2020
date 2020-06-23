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


class SimplePerceptron:
    def __init__(self, vocab, output_size=1):
        self.unk = "Unk"
        self.Phi = dict()
        for v in vocab: self.Phi[v] = 0
        self.Phi[self.unk] = 0
        input_size = len(vocab) + 1
        self.W1 = np.random.randn(input_size, output_size)
        return 

    def forward(self, X, y):
        phi = self.create_feat(X)
        # y_hat = np.dot(phi, self.W1)
        y_hat = self.predict(X)
        for i in range(len(y)):
            if y[i] != y_hat[i]:
                self.update(phi[i], y[i])
        return self

    def create_feat(self, X):
        ret = []
        for X_i in X:
            phi = copy.deepcopy(self.Phi)
            for w in X_i:
                if w not in phi.keys():
                    phi[self.unk] += 1
                else:
                    phi[w] += 1
            ret.append(list(phi.values()))
        return ret

    def update(self, phi_i, y_i):
        # tmp = np.array(phi, dtype=float).T * np.array(y, dtype=float)
        self.W1 += (np.array(phi_i) * np.array(y_i)).reshape(-1, 1)
        return self
        
    def train(self, X, y, itr=10):
        for i in range(itr):
            self.forward(X, y)
        return self
    
    def predict(self, X):
        phi = self.create_feat(X)
        pred = list(map(lambda x: 1 if x >= 0 else -1, np.dot(phi, self.W1)))
        return pred
    
    def save(self, path="./model/simple.model"):
        with open(path, "wb") as f:
            dill.dump(self.W1, f)
    
    def load(self, path="./model/simple.model"):
        with open(path, "rb") as f:
            self.W1 = dill.load(f)


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
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    # train = "../../test/03-train-input.txt"
    train = "../../data/titles-en-train.labeled"
    test = "../../data/titles-en-test.word"
    train_X, train_y, train_v = load_data(train)
    test_X, test_y, test_v = load_data(test, labeled=False)
    # vectorizer = TfidfVectorizer(max_df=0.9)
    # X = vectorizer.fit_transform(np.array(train_X).flatten())
    # words = vectorizer.get_feature_names()
    model = SimplePerceptron(train_v)
    # model.train(train_X, train_y, itr=1).save()
    model.load()
    pred_y = model.predict(train_X)
    print(accuracy_score(train_y, pred_y))
    print(confusion_matrix(train_y, pred_y))
    pred_test_y = model.predict(test_X)
    with open("./result/wikien.result", "w") as f:
        for i, p in enumerate(pred_test_y):
            f.write(str(p)+" "+" ".join(test_X[i])+"\n")
    # print(train_y, y_hat)
