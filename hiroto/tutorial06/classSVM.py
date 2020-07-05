# python SVM.py ../data/titles-en-train.labeled ../data/titles-en-train.word ../data/titles-en-test.word
from collections import defaultdict
import sys
import pickle
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tqdm import tqdm

class SVM():
    def __init__(self, train_fname, test1_fname, test2_fname):
        self.train_fname = train_fname
        self.test1_fname = test1_fname
        self.test2_fname = test2_fname
        with open(train_fname) as train_f, \
             open(test1_fname) as test1_f, \
             open(test2_fname) as test2_f:
            self.train_text = train_f.readlines()
            self.test1_text = test1_f.readlines()
            self.test2_text = test2_f.readlines()
    
    def train(self, margin, c, max_iter=20):
        self.c = c
        self.max_iter = max_iter
        self.weights = defaultdict(lambda: 0)
        self.accs1, self.accs2 = [], []
        self.last = defaultdict(int)
        self.itr = 0
        for _ in tqdm(range(max_iter)):
            for line in self.train_text:
                label, sent = line.split('\t')
                y = int(label)
                #一文中で出現した各単語の頻度をまとめた辞書(key:name, value:freqency)
                phi = self.create_features(sent)
                val = self.cal_val(phi, y)
                #ラベルを予測
                label_pred = self.predict_one(phi)
                #print(type(self.weights))
                #マージンに入っているか
                if val <= margin:
                    self.itr += 1
                    self.update_weights(phi, y)
                    #print(type(self.weights))
            self.accs1.append(self.predict_all(self.test1_fname))
            self.accs2.append(self.predict_all(self.test2_fname))
    
    #一文中で出現した各単語の頻度をまとめた辞書(key:name, value:freqency)
    def create_features(self, sent):
        phi = defaultdict(lambda: 0)
        words = sent.split()
        for word in words:
            phi[f"UNI:{word}"] += 1
        return phi
    
    #ラベルを予測
    def predict_one(self, phi):
        score = 0
        for name, value in phi.items():
            weight = self.getw(name)
            score += value * self.weights[name]
        return np.sign(score)
    
    def cal_val(self, phi, y):
        val = 0
        for name, value in phi.items():
            val += self.weights[name] * value * y
        return val
    
    #重み更新
    def update_weights(self, phi, y):
        for name, value in phi.items():
            self.weights[name] += value * y
    
    def getw(self, name):
        if self.itr != self.last[name]:
            c_size = self.c * (self.itr - self.last[name])
            if abs(self.weights[name]) <= c_size:
                self.weights[name] = 0
            else:
                self.weights[name] -= np.sign(self.weights[name]) * c_size
            self.last[name] = self.itr
        return self.weights[name]
    
    def predict_all(self, input_fname):
        labels_true = []
        labels_pred = []
        labeled_fname = re.sub(r".word", r".labeled", input_fname)
        with open(input_fname) as in_f \
            , open(labeled_fname) as labeled_file:
            for line in labeled_file:
                label, sent = line.split('\t')
                labels_true.append(int(label))
            for line in in_f:
                phi = self.create_features(line)
                pred = self.predict_one(phi)
                labels_pred.append(pred)
        return accuracy_score(labels_true, labels_pred)

    def draw(self, margin, c):
        plt.xlabel("epoch")
        plt.ylabel("Accuracy")
        plt.plot(list(range(self.max_iter)), self.accs1)
        plt.plot(list(range(self.max_iter)), self.accs2)
        plt.title(f"c={c}, margin={margin}")
        plt.savefig(f"./picture/c{c}_margin{margin}.png")
        plt.clf()

    def param_search(self, margins, cs, max_iter=20):
        for margin in margins:
            for c in cs:
                self.train(margin, c, max_iter)
                print(f"c : {c}, margin : {margin}")
                print(f"train acc : {self.accs1[-1]}")
                print(f"test acc : {self.accs2[-1]}")
                self.draw(margin, c)
    
def main():
    train_fname = sys.argv[1]
    test1_fname = sys.argv[2]
    test2_fname = sys.argv[3]
    svm = SVM(train_fname, test1_fname, test2_fname)
    margin_params = [0.1, 1.0, 10]
    c_params = [1.0, 0.1, 0.01, 0.001, 0.0001]
    svm.param_search(margins=margin_params, cs=c_params, max_iter=10)


if __name__ == '__main__':
    main()

'''
c : 1.0, margin : 0.1
train acc : 0.3779234585400425
test acc : 0.3705278072972016
c : 0.1, margin : 0.1
train acc : 0.6253543586109143
test acc : 0.624512929507616
c : 0.01, margin : 0.1
train acc : 0.8683557760453579
test acc : 0.8650371944739639
c : 0.001, margin : 0.1
train acc : 0.9558823529411765
test acc : 0.9192348565356004
c : 0.0001, margin : 0.1
train acc : 0.9865343727852587
test acc : 0.9298618490967057
c : 1.0, margin : 1.0
train acc : 0.4591601700921332
test acc : 0.44668792065178886
c : 0.1, margin : 1.0
train acc : 0.6443125442948263
test acc : 0.6376195536663124
c : 0.01, margin : 1.0
train acc : 0.7777285613040397
test acc : 0.7718738930216082
c : 0.001, margin : 1.0
train acc : 0.9619064493267186
test acc : 0.9241941197307828
c : 0.0001, margin : 1.0
train acc : 0.9872430900070872
test acc : 0.9312787814381863
c : 1.0, margin : 10
train acc : 0.27391920623671157
test acc : 0.2685086787105916
c : 0.1, margin : 10
train acc : 0.5530652019844082
test acc : 0.5423308537017357
c : 0.01, margin : 10
train acc : 0.8486888731396173
test acc : 0.8466170740347149
c : 0.001, margin : 10
train acc : 0.9594259390503189
test acc : 0.9241941197307828
c : 0.0001, margin : 10
train acc : 0.9920269312544295
test acc : 0.9319872476089267

Perceptron:
Accuracy = 92.702798%
'''