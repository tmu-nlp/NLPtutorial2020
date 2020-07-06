# python classSVM.py ../data/titles-en-train.labeled ../data/titles-en-train.word ../data/titles-en-test.word
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
        self.flag = True
        with open(train_fname) as train_f, \
             open(test1_fname) as test1_f, \
             open(test2_fname) as test2_f:
            self.train_text = train_f.readlines()
            self.test1_text = test1_f.readlines()
            self.test2_text = test2_f.readlines()
    
    def train_mode(self):
        self.flag = True
    
    def eval_mode(self):
        self.flag = False

    def train(self, margin, c, max_iter=20):
        self.c = c
        self.max_iter = max_iter
        self.weights = defaultdict(lambda: 0)
        self.accs1, self.accs2 = [], []
        self.last = defaultdict(int)
        self.itr = 0
        for _ in tqdm(range(max_iter)):
            for line in self.train_text:
                self.train_mode()
                label, sent = line.split('\t')
                y = int(label)
                #一文中で出現した各単語の頻度をまとめた辞書(key:name, value:freqency)
                phi = self.create_features(sent)
                #val = self.cal_val(phi, y)
                #ラベルを予測
                label_pred = self.predict_one(phi)
                val = self.cal_val(phi, y)
                #print(type(self.weights))
                #マージンに入っているか
                if val <= margin:
                    self.itr += 1
                    self.update_weights(phi, y)
                    #print(type(self.weights))
            self.eval_mode()
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
            score += value * weight
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
        if self.flag == True:
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
        plt.savefig(f"./picture_sub/c{c}_margin{margin}.png")
        plt.clf()

    def param_search(self, margins, cs, max_iter=20):
        for margin in margins:
            for c in cs:
                self.train(margin, c, max_iter)
                print(f"c : {c}, margin : {margin}")
                print(f"train acc : {self.accs1[-1]}")
                print(f"test acc : {self.accs2[-1]}")
                #self.draw(margin, c)
    
def main():
    train_fname = sys.argv[1]
    test1_fname = sys.argv[2]
    test2_fname = sys.argv[3]
    svm = SVM(train_fname, test1_fname, test2_fname)
    margin_params = [0.1, 1.0, 10]
    c_params = [1.0, 0.1, 0.01, 0.001, 0.0001]
    svm.param_search(margins=margin_params, cs=c_params, max_iter=30)


if __name__ == '__main__':
    main()

'''
100%|███████████████████████████████████████████████████████████████████████████████| 30/30 [00:34<00:00,  1.15s/it]
c : 1.0, margin : 0.1
train acc : 0.6371367824238129
test acc : 0.5894438540559688
100%|███████████████████████████████████████████████████████████████████████████████| 30/30 [00:45<00:00,  1.50s/it]
c : 0.1, margin : 0.1
train acc : 0.630846917080085
test acc : 0.6255756287637265
100%|███████████████████████████████████████████████████████████████████████████████| 30/30 [00:48<00:00,  1.61s/it]
c : 0.01, margin : 0.1
train acc : 0.7144755492558469
test acc : 0.7077577045696068
100%|███████████████████████████████████████████████████████████████████████████████| 30/30 [00:47<00:00,  1.57s/it]
c : 0.001, margin : 0.1
train acc : 0.9238128986534373
test acc : 0.8901877435352462
100%|███████████████████████████████████████████████████████████████████████████████| 30/30 [00:43<00:00,  1.43s/it]
c : 0.0001, margin : 0.1
train acc : 0.9988483345145287
test acc : 0.9323414806942969
100%|███████████████████████████████████████████████████████████████████████████████| 30/30 [00:34<00:00,  1.16s/it]
c : 1.0, margin : 1.0
train acc : 0.655474840538625
test acc : 0.6007793127878144
100%|███████████████████████████████████████████████████████████████████████████████| 30/30 [00:47<00:00,  1.58s/it]
c : 0.1, margin : 1.0
train acc : 0.6424521615875266
test acc : 0.6372653205809422
100%|███████████████████████████████████████████████████████████████████████████████| 30/30 [00:46<00:00,  1.56s/it]
c : 0.01, margin : 1.0
train acc : 0.8804039688164422
test acc : 0.8660998937300743
100%|███████████████████████████████████████████████████████████████████████████████| 30/30 [00:44<00:00,  1.47s/it]
c : 0.001, margin : 1.0
train acc : 0.97537207654146
test acc : 0.9259652851576338
100%|███████████████████████████████████████████████████████████████████████████████| 30/30 [00:43<00:00,  1.45s/it]
c : 0.0001, margin : 1.0
train acc : 0.9990255138199858
test acc : 0.9355295784626284
100%|███████████████████████████████████████████████████████████████████████████████| 30/30 [00:41<00:00,  1.38s/it]
c : 1.0, margin : 10
train acc : 0.53756201275691
test acc : 0.5239107332624867
100%|███████████████████████████████████████████████████████████████████████████████| 30/30 [00:45<00:00,  1.51s/it]
c : 0.1, margin : 10
train acc : 0.5833628632175761
test acc : 0.5745660644704216
100%|███████████████████████████████████████████████████████████████████████████████| 30/30 [00:44<00:00,  1.48s/it]
c : 0.01, margin : 10
train acc : 0.8586109142452162
test acc : 0.8586609989373007
100%|███████████████████████████████████████████████████████████████████████████████| 30/30 [00:46<00:00,  1.56s/it]
c : 0.001, margin : 10
train acc : 0.9687278525868178
test acc : 0.9252568189868934
100%|███████████████████████████████████████████████████████████████████████████████| 30/30 [00:44<00:00,  1.48s/it]
c : 0.0001, margin : 10
train acc : 0.9994684620836286
test acc : 0.9344668792065179

Perceptron:
Accuracy = 92.702798%
'''