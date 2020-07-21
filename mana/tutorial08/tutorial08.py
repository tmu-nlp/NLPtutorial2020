import numpy as np
from collections import defaultdict
from tqdm import tqdm

class rnn():
    def __init__(self, node = 10):
        self.feat_lab = []
        self.net = []
        self.word_ids = defaultdict(lambda: len(self.word_ids))
        self.tag_ids = defaultdict(lambda: len(self.tag_ids))
        self.node = node

    def init_network(self):
        # Networkの初期化（ランダム）
        # np.random.rand(n, m): -0.5以上, 0.5未満のｍ要素のリストｘnのリスト
    
        # 入力に対する重み
        w_r_x = np.random.rand(self.node, len(self.word_ids)) / 5 - 0.1
        # 次の時刻に伝播する重み
        w_r_h = np.random.rand(self.node, self.node) / 5 - 0.1
        # 出力に対する重み
        w_o_h = np.random.rand(len(self.tag_ids), self.node) / 5 - 0.1

        b_r = np.random.rand(self.node) / 5 - 0.1
        b_o = np.random.rand(len(self.tag_ids)) / 5 - 0.1

        self.net = [w_r_x, w_r_h, w_o_h, b_r, b_o]
        #print(self.net)


    def gradient_rnn(self, sent, h, p):
        w_r_x, w_r_h, w_o_h, b_r, b_o = self.net
        delta_w_r_x = np.zeros((self.node, len(self.word_ids)))
        delta_w_r_h = np.zeros((self.node, self.node))
        delta_b_r = np.zeros(self.node)
        delta_w_o_h = np.zeros((len(self.tag_ids), self.node))
        delta_b_o = np.zeros(len(self.tag_ids))
        delta_r_prime = np.zeros(self.node)
        for t in range(len(sent)-1, -1, -1):
            word, tag = sent[t]
            p_prime = create_one_hot(self.tag_ids[tag], len(self.tag_ids))
            x = create_one_hot(self.word_ids[word], len(self.word_ids))
            
            delta_o_prime = p_prime - p[t]
            delta_w_o_h += np.outer(h[t], delta_o_prime).T
            delta_b_o += delta_o_prime
            delta_r = np.dot(delta_r_prime, w_r_h) + np.dot(delta_o_prime, w_o_h)
            delta_r_prime = delta_r * (1-h[t]**2)
            delta_w_r_x += np.outer(x, delta_r_prime).T
            delta_b_r += delta_r_prime
            if t!=0:
                delta_w_r_h += np.outer(h[t-1], delta_r_prime).T
        return [delta_w_r_x, delta_w_r_h, delta_w_o_h, delta_b_r, delta_b_o]

    def update_weights(self, delta, lr):
        w_r_x, w_r_h, w_o_h, b_r, b_o = self.net
        delta_w_r_x, delta_w_r_h, delta_w_o_h, delta_b_r, delta_b_o = delta

        w_r_x += lr*delta_w_r_x
        w_r_h += lr*delta_w_r_h
        b_r += lr*delta_b_r
        w_o_h += lr*delta_w_o_h
        b_o += lr*delta_b_o

        self.net = [w_r_x, w_r_h, w_o_h, b_r, b_o]


    def forward_rnn(self, sent):
        w_r_x, w_r_h, w_o_h, b_r, b_o = self.net
        h = []
        p = []
        y = []
        for t in range(len(sent)):
            x, _ = sent[t]
            x = create_one_hot(self.word_ids[x], len(self.word_ids))
            if t > 0:
                h.append(np.tanh(np.dot(w_r_x, x) + np.dot(w_r_h, h[t-1]) + b_r))
            else:
                h.append(np.tanh(np.dot(w_r_x, x) + b_r))

            p.append(np.tanh(np.dot(w_o_h, h[t]) + b_o))
            y.append(find_best(p[t]))

        return h, p, y


    def preparation(self, input_file):
        with open(input_file) as f:
            inputFile = f.readlines()

        for line in inputFile:
            word_tags = line.strip().split()
            sent = []
            for elem in word_tags:
                word, tag = elem.split("_")
                self.word_ids[word]
                self.tag_ids[tag]
                sent.append((word, tag))
            self.feat_lab.append(sent)


    def train_rnn(self, train_path, epoch):
        self.preparation(train_path)
        self.init_network()
        for _ in tqdm(range(epoch)):
            for sent in self.feat_lab:
                h, p, y = self.forward_rnn(sent)
                delta = self.gradient_rnn(sent, h, p)
                self.update_weights(delta, 0.01)

    def forward_rnn_test(self, sent):
        w_r_x, w_r_h, w_o_h, b_r, b_o = self.net
        h = []
        p = []
        y = []
        for t in range(len(sent)):
            x = sent[t]
            if x in self.word_ids:
                x = create_one_hot(self.word_ids[x], len(self.word_ids))
            else:
                x = np.zeros(len(self.word_ids))
            if t > 0:
                h.append(np.tanh(np.dot(w_r_x, x) + np.dot(w_r_h, h[t-1]) + b_r))
            else:
                h.append(np.tanh(np.dot(w_r_x, x) + b_r))

            p.append(np.tanh(np.dot(w_o_h, h[t]) + b_o))
            y.append(find_best(p[t]))

        return h, p, y

    def test_rnn(self, test_file, ans_file):
        with open(test_file) as f:
            testFile = f.readlines()

        ans = open(ans_file, "w")

        for line in testFile:
            pos = []
            line = line.strip().split()

            h, p, y = self.forward_rnn_test(line)
            for elem in y:
                for key, value in self.tag_ids.items():
                    if elem == value:
                        pos.append(key)

            ans.write(" ".join(pos) + "\n")

  
def create_one_hot(id, size):
    vec = np.zeros(size)
    vec[id] = 1
    return vec

def find_best(p):
    y = 0
    for i in range(len(p)):
        if p[i] > p[y]:
            y = i
    return y


rnn_model = rnn()
#rnn_model.train_rnn("/work/test/05-train-input.txt", 20)
#rnn_model.test_rnn("/work/test/05-test-input.txt")


rnn_model.train_rnn("/work/data/wiki-en-train.norm_pos", 20)
rnn_model.test_rnn("/work/data/wiki-en-test.norm", "my_answer.txt")

"""
iter: 50, node: 10, lr: 0.01
Accuracy: 71.60% (3267/4563)

Most common mistakes:
VBN --> RB      116
VB --> RB       89
VBP --> RB      79
NNS --> NN      65
JJ --> NN       64
-RRB- --> RB    47
-LRB- --> RB    46
PRP --> RB      43
MD --> RB       43
VBG --> RB      43

iter: 20, node: 10, lr: 0.01
Accuracy: 69.87% (3188/4563)

Most common mistakes:
NN --> NNS      122
VBN --> RB      85
VB --> RB       80
JJ --> NNS      76
NNP --> NNS     62
VBP --> RB      57
VBN --> NNS     56
-RRB- --> RB    46
-LRB- --> RB    40
RB --> NNS      36

For a reference, HMM model

Accuracy: 90.82% (4144/4563)

Most common mistakes:
NNS --> NN      45
NN --> JJ       27
NNP --> NN      22
JJ --> DT       22
VBN --> NN      12
JJ --> NN       12
NN --> IN       11
NN --> DT       10
NNP --> JJ      8
JJ --> VBN      7
"""


