# python classNN.py ../data/titles-en-train.labeled ../data/titles-en-train.word ../data/titles-en-test.word
from collections import defaultdict
import sys
import pickle
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from nn import Linear
from pprint import pprint

def word2id(ids, train_text):
    for line in train_text:
        line = line.strip()
        _, sent = line.split('\t')
        words = sent.split()
        for word in words:
            ids[f"UNI:{word}"]
    return ids

class MyNN():
    def __init__(self, train_fname, test1_fname, test2_fname, hidden_dim=2):
        self.train_fname = train_fname
        self.test1_fname = test1_fname
        self.test2_fname = test2_fname
        self.flag = True
        self.ids = defaultdict(lambda:len(self.ids))
        with open(train_fname) as train_f, \
             open(test1_fname) as test1_f, \
             open(test2_fname) as test2_f:
            self.train_text = train_f.readlines()
            self.test1_text = test1_f.readlines()
            self.test2_text = test2_f.readlines()
        self.ids = word2id(self.ids, self.train_text)

    def __call__(self, inputs):
        return self.forward(inputs)

    def arch(self, output_dims):
        self.layers = []
        for i, o_dim in enumerate(output_dims):
            if i==0:
                self.layers.append(Linear(len(self.ids), o_dim))
            else:
                self.layers.append(Linear(in_dim, o_dim))
            in_dim = o_dim

    def forward(self, inputs):
        for layer in self.layers:
            outputs = layer(inputs)
            inputs = outputs
        return outputs
    
    def backward(self, delta):
        for layer in self.layers[::-1]:
            delta = layer.backward(delta[0])

    def update_weights(self, lr=0.01):
        for layer in self.layers:
            layer.update_weights(self, lr)


    def train(self):
        self.flag = True
    
    def eval(self):
        self.flag = False

    def fit(self, max_iter=20, lr=0.001):
        self.max_iter = max_iter
        self.accs1, self.accs2 = [], []
        for epoch in tqdm(range(self.max_iter)):
            np.random.shuffle(self.train_text)
            cnt = 0
            running_loss = 0
            for line in self.train_text:
                self.train()
                label, sent = line.strip().split('\t')
                y = int(label)
                #一文中で出現した各単語の頻度をまとめた辞書(key:name, value:freqency)
                phi = self.create_features(sent)
                score = self(phi)
                err = self.criterion(y, score)
                running_loss += err
                self.backward(y - score)
                self.update_weights(lr)
                cnt += 1
            self.eval()
            train_acc = self.predict_all(self.test1_fname)
            dev_acc = self.predict_all(self.test2_fname)
            self.accs1.append(train_acc)
            self.accs2.append(dev_acc)
            loss = running_loss / cnt
            print(f"epoch{epoch+1}")
            print(f"loss: {loss[0]}")
            print(f"train acc: {train_acc}")
            print(f"dev acc: {dev_acc}")
            
    
    #一文中で出現した各単語の頻度をまとめた辞書(key:name, value:freqency)
    def create_features(self, sent):
        phi = np.zeros(len(self.ids))
        words = sent.strip().split()
        for word in words:
            if f"UNI:{word}" in self.ids.keys():
                phi[self.ids[f"UNI:{word}"]] += 1
            else: pass
        return phi
    
    #ラベルを予測
    def predict_one(self, phi):
        score = self(phi)
        return np.sign(score)
    
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
   
    def criterion(self, true, score):
        return (true - score)**2 / 2
    
    def draw(self, lr):
        plt.xlabel("epoch")
        plt.ylabel("Accuracy")
        plt.plot(list(range(self.max_iter)), self.accs1)
        plt.plot(list(range(self.max_iter)), self.accs2)
        plt.savefig(f"./picture/lr{lr}.png")
        plt.clf()

def main():
    train_fname = sys.argv[1]
    test1_fname = sys.argv[2]
    test2_fname = sys.argv[3]
    model = MyNN(train_fname, test1_fname, test2_fname)
    #それぞれのレイヤの出力次元数
    #e.g.) model.arch([output_dim1, output_dim2, output_dim3, ...])
    model.arch([5, 2, 1])
    model.fit(10)
    model.draw(0.001)

if __name__ == '__main__':
    main()


'''
epoch1
loss: 0.3124149728298576
train acc: 0.9013111268603827
dev acc: 0.900106269925611
 10%|████████                                                                        | 1/10 [00:09<01:25,  9.53s/it]epoch2
loss: 0.13982233585889728
train acc: 0.9316087880935506
dev acc: 0.9227771873893021
 20%|████████████████                                                                | 2/10 [00:17<01:13,  9.13s/it]epoch3
loss: 0.11017418959199078
train acc: 0.9412650602409639
dev acc: 0.9280906836698548
 30%|████████████████████████                                                        | 3/10 [00:25<01:01,  8.79s/it]epoch4
loss: 0.09350968300057352
train acc: 0.9519844082211197
dev acc: 0.9341126461211477
 40%|████████████████████████████████                                                | 4/10 [00:33<00:51,  8.55s/it]epoch5
loss: 0.08016959132665105
train acc: 0.9584514528703048
dev acc: 0.9365922777187389
 50%|████████████████████████████████████████                                        | 5/10 [00:41<00:41,  8.37s/it]epoch6
loss: 0.06891697830898284
train acc: 0.9642983699503898
dev acc: 0.9397803754870705
 60%|████████████████████████████████████████████████                                | 6/10 [00:49<00:32,  8.24s/it]epoch7
loss: 0.06022621275644925
train acc: 0.9670446491849752
dev acc: 0.9376549769748495
 70%|████████████████████████████████████████████████████████                        | 7/10 [00:57<00:24,  8.27s/it]epoch8
loss: 0.05267416469541161
train acc: 0.9685506732813607
dev acc: 0.9358838115479986
 80%|████████████████████████████████████████████████████████████████                | 8/10 [01:05<00:16,  8.15s/it]epoch9
loss: 0.04717361666503433
train acc: 0.9701452870304749
dev acc: 0.934821112291888
 90%|████████████████████████████████████████████████████████████████████████        | 9/10 [01:13<00:08,  8.05s/it]epoch10
loss: 0.04118409859839187
train acc: 0.9746633593196314
dev acc: 0.9355295784626284
100%|███████████████████████████████████████████████████████████████████████████████| 10/10 [01:21<00:00,  8.14s/it]

~/Desktop/tutorial/ch07 1m 23s
(anaconda3-2019.10) anaconda3-2019.10 ❯ python classNN.py ../data/titles-en-train.labeled ../data/titles-en-train.word ../data/titles-en-test.word
  0%|                                                                                        | 0/10 [00:00<?, ?it/s]epoch1
loss: 0.19253094396317944
train acc: 0.9279766123316796
dev acc: 0.9156925256818987
 10%|████████                                                                        | 1/10 [00:07<01:04,  7.15s/it]epoch2
loss: 0.1135236357903621
train acc: 0.9437455705173635
dev acc: 0.9302160821820759
 20%|████████████████                                                                | 2/10 [00:14<00:57,  7.17s/it]epoch3
loss: 0.09248385225432373
train acc: 0.9518072289156626
dev acc: 0.9351753453772582
 30%|████████████████████████                                                        | 3/10 [00:21<00:49,  7.13s/it]epoch4
loss: 0.07892997081632665
train acc: 0.9643869596031184
dev acc: 0.940843074743181
 40%|████████████████████████████████                                                | 4/10 [00:28<00:43,  7.26s/it]epoch5
loss: 0.06589284633981599
train acc: 0.9687278525868178
dev acc: 0.9383634431455898
 50%|████████████████████████████████████████                                        | 5/10 [00:36<00:36,  7.34s/it]epoch6
loss: 0.05746498971156581
train acc: 0.9762579730687456
dev acc: 0.9436769394261424
 60%|████████████████████████████████████████████████                                | 6/10 [00:44<00:30,  7.51s/it]epoch7
loss: 0.048859297247207334
train acc: 0.9752834868887313
dev acc: 0.9411973078285512
 70%|████████████████████████████████████████████████████████                        | 7/10 [00:52<00:22,  7.63s/it]epoch8
loss: 0.04348386648290219
train acc: 0.9825478384124734
dev acc: 0.9454481048529932
 80%|████████████████████████████████████████████████████████████████                | 8/10 [01:00<00:15,  7.93s/it]epoch9
loss: 0.03837781047018687
train acc: 0.9835223245924876
dev acc: 0.9415515409139213
 90%|████████████████████████████████████████████████████████████████████████        | 9/10 [01:08<00:07,  7.94s/it]epoch10
loss: 0.033999836378485804
train acc: 0.9858256555634302
dev acc: 0.9397803754870705
100%|███████████████████████████████████████████████████████████████████████████████| 10/10 [01:17<00:00,  7.71s/it]
'''