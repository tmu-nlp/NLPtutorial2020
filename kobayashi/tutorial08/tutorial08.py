#python tutorial08.py ../../data/wiki-en-train.norm_pos ../../data/wiki-en-test.norm result.txt
import numpy as np
import sys
from collections import defaultdict

class RNN:
    def __init__(self):
        self.word_ids = defaultdict(lambda:len(self.word_ids))
        self.lab_ids = defaultdict(lambda:len(self.lab_ids))
        self.word_with_lab = []
        self.word_lab_vec = []
        self.nodes = 5

    def count_features(self, input_file):
        with open (input_file, "r", encoding = "utf-8") as input_file:
            for line in input_file:
                for word_with_lab in line.strip().split(" "):
                    word, lab = word_with_lab.split("_")
                    self.word_ids[word]
                    self.lab_ids[lab]

    def create_net(self):
        self.net = []
        w_rx = np.random.rand(self.nodes, len(self.word_ids))-0.5
        self.net.append(w_rx)
        w_rh = np.random.rand(self.nodes, self.nodes)-0.5
        self.net.append(w_rh)
        b_r = np.random.rand(self.nodes)-0.5
        self.net.append(b_r)
        w_oh = np.random.rand(len(self.lab_ids), self.nodes)-0.5 
        self.net.append(w_oh)
        b_o = np.random.rand(len(self.lab_ids))-0.5
        self.net.append(b_o)

    #w_rx = net[0]
    #w_rh = net[1]
    #w_oh = net[3]
    #b_r = net[2]
    #b_o = net[4]
    def forward(self, x):
        h = [0] * len(x)
        p = [0] * len(x)
        y = [0] * len(x)
        for t in range(len(x)):
            if t > 0:
                h[t] = np.tanh(np.dot(self.net[0], x[t]) + np.dot(self.net[1], h[t-1]) + self.net[2])
            else:
                h[t] = np.tanh(np.dot(self.net[0], x[t]) + self.net[2])
            p[t] = self.softmax(np.dot(self.net[3], h[t]) + self.net[4])
            y[t] = np.argmax(p[t])
        return h, p, y
    
    def softmax(self, x):
        a= np.sum(np.exp(x))
        return np.exp(x)/a

    def create_onehot(self, id, size):
        vec = np.zeros(size)
        vec[id] = 1
        return vec
    
    def find_best(self,p):
        y=0
        for i in range(len(p)):
            if p[i] > p[y]:
                y = i
        return y

    def gradient(self, x, h, p, y_d):
        Δw_rx = np.zeros_like(self.net[0])
        Δw_rh = np.zeros_like(self.net[1])
        Δb_r = np.zeros_like(self.net[2])
        Δw_oh = np.zeros_like(self.net[3])
        Δb_o = np.zeros_like(self.net[4])
        deltar_d = np.zeros(len(self.net[2]))

        for t in range(len(x))[::-1]:
            p_d = y_d
            deltao_d = p_d[t] - p[t]
            Δw_oh += np.outer(deltao_d, h[t])
            Δb_o += deltao_d
            deltar = np.dot(deltar_d, self.net[1])+np.dot(deltao_d,self.net[3])
            deltar_d = deltar * (1 - h[t] ** 2)
            Δw_rx += np.outer( deltar_d, x[t])
            Δb_r += deltar_d
            if t != 0:
                Δw_rh += np.outer(deltar_d, h[t-1])
        return [Δw_rx, Δw_rh, Δb_r, Δw_oh, Δb_o]

    def create_features(self, x):
        word_vec = []
        lab_vec = []
        words_labs = x.split(" ")
        for word_lab in words_labs:
            word, lab = word_lab.split("_")
            word_vec.append(self.create_onehot(self.word_ids[word], len(self.word_ids)))
            lab_vec.append(self.create_onehot(self.lab_ids[lab], len(self.lab_ids)))
        return [word_vec, lab_vec]
    
    def create_features_test(self, x):
        word_vec = []
        words = x.split(" ")
        for word in words:
            if word in self.word_ids:
                word_vec.append(self.create_onehot(self.word_ids[word], len(self.word_ids)))
        return word_vec

    def train(self, input_file1,lr,iter):
        self.count_features(input_file1)
        self.create_net()
        
        for i in range(iter):
            input_file = open (input_file1, "r", encoding = "utf-8")
            print(i) 
            for line in input_file:
                word_vec, lab_vec = self.create_features(line.strip())
                h, p, _ = self.forward(word_vec)
                Δ = self.gradient(word_vec, h, p, lab_vec)
                self.update_weights(Δ, lr)
        input_file.close()                
    
    def update_weights(self, Δ, rate):
        Δw_rx, Δw_rh, Δb_r, Δw_oh, Δb_o = Δ
        self.net[0] += rate * Δw_rx
        self.net[1] += rate * Δw_rh
        self.net[2] += rate * Δb_r
        self.net[3] += rate * Δw_oh
        self.net[4] += rate * Δb_o
    
    def test(self, input_file, output_file):
        with open(input_file, "r", encoding = "utf-8") as input_file,\
            open(output_file, "w", encoding = "utf-8") as output_file:
            for line in input_file:
                a = []
                word_vec = self.create_features_test(line.strip())
                _, _, y = self.forward(word_vec)
                for k in range(len(y)):
                    for i,j in self.lab_ids.items():
                        if j == y[k]:
                            a.append(str(i))
                            output_file.write(f"{i} ")
                output_file.write("\n")
                
if __name__ == "__main__":
    rnn = RNN()
    rnn.train(sys.argv[1],0.01,3)
    rnn.test(sys.argv[2], sys.argv[3])