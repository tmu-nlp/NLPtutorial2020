# python classRNN.py ../data/wiki-en-train.norm_pos ../data/wiki-en-train.norm_pos ../data/wiki-en-test.norm_pos
from collections import defaultdict
import sys
import pickle
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from nn import RNN
from pprint import pprint

def wordtag2id(ids_word, ids_tag, train_text):
    for line in train_text:
        line = line.strip()
        wordtags = line.split()
        for wordtag in wordtags:
            word, tag = wordtag.split('_')
            ids_word[word]
            ids_tag[tag]
    return ids_word, ids_tag

class MyNN():
    def __init__(self, train_fname, test1_fname, test2_fname):
        self.train_fname = train_fname
        self.test1_fname = test1_fname
        self.test2_fname = test2_fname
        self.flag = True
        self.ids_word = defaultdict(lambda:len(self.ids_word))
        self.ids_tag = defaultdict(lambda:len(self.ids_tag))
        with open(train_fname) as train_f, \
             open(test1_fname) as test1_f, \
             open(test2_fname) as test2_f:
            self.train_text = train_f.readlines()
            self.test1_text = test1_f.readlines()
            self.test2_text = test2_f.readlines()
        self.ids_word, self.ids_tag = wordtag2id(self.ids_word, self.ids_tag, self.train_text)

    def __call__(self, inputs):
        return self.forward(inputs)

    def arch(self, hidden_dim=20):
        self.rnn = RNN(input_dim=len(self.ids_word), hidden_dim=hidden_dim, output_dim=len(self.ids_tag))

    def forward(self, inputs):
        return self.rnn(inputs)
    
    def backward(self, tag_list):
        self.rnn.backward(tag_list)

    def update_weights(self, lr=0.001):
        self.rnn.update_weights(lr)

    def train(self):
        self.flag = True
    
    def eval(self):
        self.flag = False

    def segment(self, wordtags):
        #word_list, tag_list はそれぞれの単語の one-hot ベクトルを入れたもの
        #tagsはラベル( id )を入れたもの
        word_list, tag_list, tags = [], [], []
        for wordtag in wordtags:
            word, tag = wordtag.split('_')
            #word_vec, tag_vec は one-hot ベクトル
            word_vec = np.zeros(len(self.ids_word))
            tag_vec = np.zeros(len(self.ids_tag))
            if word in self.ids_word.keys():
                word_vec[self.ids_word[word]] += 1
                tag_vec[self.ids_tag[tag]] += 1
            else: pass
            tags.append(self.ids_tag[tag])
            word_list.append(word_vec)
            tag_list.append(tag_vec)
        return  word_list, tag_list, tags

    def fit(self, max_iter=20, lr=0.001):
        self.max_iter = max_iter
        self.accs1, self.accs2 = [], []
        np.random.seed(1)
        np.random.shuffle(self.train_text)
        for epoch in tqdm(range(self.max_iter)):
            cnt = 0
            running_loss = 0
            for line in self.train_text:
                self.train()
                wordtags = line.strip().split()
                word_list, tag_list, _ = self.segment(wordtags)
                _, p, _ = self(word_list)
                err = self.criterion(tag_list, p)
                running_loss += err
                self.backward(tag_list)
                self.update_weights(lr)
                cnt += 1
            self.eval()
            train_acc = self.predict_all(self.test1_text)
            test_acc = self.predict_all(self.test2_text)
            self.accs1.append(train_acc)
            self.accs2.append(test_acc)
            loss = running_loss / cnt
            print(f"epoch{epoch+1}")
            print(f"loss: {loss}")
            print(f"train acc: {train_acc}")
            print(f"test acc: {test_acc}")
            
    def predict_all(self, text):
        labels_true = []
        labels_pred = []
        for line in text:
            wordtags = line.strip().split()
            word_list, tag_list, tags = self.segment(wordtags)
            _, _, y = self(word_list)
            labels_true += tags
            labels_pred += y
        return accuracy_score(labels_true, labels_pred)
   
    def criterion(self, tag_list, p):
        loss = 0
        for true, pred in zip(tag_list, p):
            loss += np.sum((true - pred)**2 / 2)
        return loss / len(p)
    
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
    model.arch(hidden_dim=50)
    lr = 0.001
    model.fit(max_iter=30, lr=lr)
    model.draw(lr=lr)

if __name__ == '__main__':
    main()

'''
  0%|                                                                                        | 0/30 [00:00<?, ?it/s]epoch1
loss: 0.40890008437408804
train acc: 0.5300078167974291
test acc: 0.5226824457593688
  3%|██▋                                                                             | 1/30 [00:34<16:41, 34.55s/it]epoch2
loss: 0.31524243890998904
train acc: 0.6114183144668655
test acc: 0.61385053692746
  7%|█████▎                                                                          | 2/30 [01:10<16:21, 35.05s/it]epoch3
loss: 0.27444649385356507
train acc: 0.6518919544888683
test acc: 0.6568047337278107
 10%|████████                                                                        | 3/30 [01:46<15:51, 35.23s/it]epoch4
loss: 0.2500491708667721
train acc: 0.6844329926753713
test acc: 0.6920885382423844
 13%|██████████▋                                                                     | 4/30 [02:22<15:19, 35.38s/it]epoch5
loss: 0.23131176091951158
train acc: 0.7109811528328653
test acc: 0.7164146394915626
 17%|█████████████▎                                                                  | 5/30 [02:57<14:46, 35.45s/it]epoch6
loss: 0.21511618568475105
train acc: 0.7329550389392316
test acc: 0.7332895025202717
 20%|████████████████                                                                | 6/30 [03:33<14:12, 35.54s/it]epoch7
loss: 0.20074687106812403
train acc: 0.7537129787788426
test acc: 0.7519175980714442
 23%|██████████████████▋                                                             | 7/30 [04:09<13:38, 35.58s/it]epoch8
loss: 0.1880588812737217
train acc: 0.7714310529515648
test acc: 0.7646285338593031
 27%|█████████████████████▎                                                          | 8/30 [04:44<13:03, 35.62s/it]epoch9
loss: 0.17692572598956677
train acc: 0.7855302394256102
test acc: 0.7758053911900066
 30%|████████████████████████                                                        | 9/30 [05:20<12:28, 35.66s/it]epoch10
loss: 0.1671436260799254
train acc: 0.7995425725948873
test acc: 0.7861056322594784
 33%|██████████████████████████▎                                                    | 10/30 [05:58<12:04, 36.23s/it]epoch11
loss: 0.15849358329899
train acc: 0.8143076344054891
test acc: 0.8016655708963402
 37%|████████████████████████████▉                                                  | 11/30 [06:43<12:18, 38.86s/it]epoch12
loss: 0.1507885419479751
train acc: 0.8288410874033757
test acc: 0.8132807363576594
 40%|███████████████████████████████▌                                               | 12/30 [07:26<12:05, 40.31s/it]epoch13
loss: 0.14387848244403786
train acc: 0.8385397064358299
test acc: 0.8205128205128205
 43%|██████████████████████████████████▏                                            | 13/30 [08:02<11:02, 38.96s/it]epoch14
loss: 0.13764207855643726
train acc: 0.8468197214904027
test acc: 0.8279640587332895
 47%|████████████████████████████████████▊                                          | 14/30 [08:39<10:11, 38.21s/it]epoch15
loss: 0.1319804209076021
train acc: 0.8544628123100084
test acc: 0.8332237563006794
 50%|███████████████████████████████████████▌                                       | 15/30 [09:14<09:21, 37.45s/it]epoch16
loss: 0.12681313636651276
train acc: 0.8618453432153094
test acc: 0.8369493754109139
 53%|██████████████████████████████████████████▏                                    | 16/30 [09:50<08:37, 36.93s/it]epoch17
loss: 0.12207492592102483
train acc: 0.8676355635331925
test acc: 0.843962305500767
 57%|████████████████████████████████████████████▊                                  | 17/30 [10:26<07:56, 36.65s/it]epoch18
loss: 0.11771230768563785
train acc: 0.873802148171738
test acc: 0.8472496164803857
 60%|███████████████████████████████████████████████▍                               | 18/30 [11:02<07:15, 36.33s/it]epoch19
loss: 0.11368090440944878
train acc: 0.8789554442546539
test acc: 0.8511943896559281
 63%|██████████████████████████████████████████████████                             | 19/30 [11:38<06:38, 36.22s/it]epoch20
loss: 0.10994334483772285
train acc: 0.8838481804232651
test acc: 0.8531667762436993
 67%|████████████████████████████████████████████████████▋                          | 20/30 [12:13<06:01, 36.10s/it]epoch21
loss: 0.10646768344330387
train acc: 0.8896094496395588
test acc: 0.8562349331580101
 70%|███████████████████████████████████████████████████████▎                       | 21/30 [12:49<05:23, 35.97s/it]epoch22
loss: 0.10322624303442333
train acc: 0.8939231637763817
test acc: 0.8595222441376288
 73%|█████████████████████████████████████████████████████████▉                     | 22/30 [13:25<04:46, 35.86s/it]epoch23
loss: 0.1001947980748759
train acc: 0.8979763179988999
test acc: 0.8623712469866316
 77%|████████████████████████████████████████████████████████████▌                  | 23/30 [14:00<04:10, 35.81s/it]epoch24
loss: 0.09735201355587716
train acc: 0.9024347876436698
test acc: 0.8643436335744028
 80%|███████████████████████████████████████████████████████████████▏               | 24/30 [14:36<03:34, 35.82s/it]epoch25
loss: 0.09467905751481126
train acc: 0.9061694797487044
test acc: 0.8674117904887135
 83%|█████████████████████████████████████████████████████████████████▊             | 25/30 [15:12<02:58, 35.71s/it]epoch26
loss: 0.09215932000474295
train acc: 0.9106279493934745
test acc: 0.8687267148805611
 87%|████████████████████████████████████████████████████████████████████▍          | 26/30 [15:47<02:22, 35.67s/it]epoch27
loss: 0.08977818960516568
train acc: 0.9146811036159926
test acc: 0.8706991014683323
 90%|███████████████████████████████████████████████████████████████████████        | 27/30 [16:23<01:46, 35.60s/it]epoch28
loss: 0.08752285522230534
train acc: 0.918473697924206
test acc: 0.8735481043173351
 93%|█████████████████████████████████████████████████████████████████████████▋     | 28/30 [16:58<01:11, 35.58s/it]epoch29
loss: 0.08538211522154222
train acc: 0.9221794389276512
test acc: 0.8742055665132589
 97%|████████████████████████████████████████████████████████████████████████████▎  | 29/30 [17:34<00:35, 35.59s/it]epoch30
loss: 0.08334618772471768
train acc: 0.9252772067977186
test acc: 0.8755204909051063
100%|███████████████████████████████████████████████████████████████████████████████| 30/30 [18:09<00:00, 36.33s/it]

HMM:
Accuracy: 90.82% (4144/4563)
'''