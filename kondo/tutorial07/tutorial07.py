import numpy as np
from collections import defaultdict

class NeuralNetwork():
    def __init__(self, layer=1, node=2):
        self.layer = layer
        self.node = node
        self.ids = defaultdict(lambda: len(self.ids))
        self.feat_lab = []
        self.net = []
        np.random.seed(7)

    def init_net(self, data):
        samples=[]
        with open(data, encoding="utf-8") as f:
            for line in f:
                sample = line.split("\t")
                samples.append(sample)
                #idsの更新
                self.create_features(sample[1])
        for y, x in samples:
            self.feat_lab.append((self.create_features(x), int(y)))

        #初期値は[-0.1, 0.1]
        #入力層
        w_in = np.random.rand(self.node, len(self.ids))/5 - 0.1
        b_in = np.random.rand(self.node)/5 - 0.1
        self.net.append((w_in, b_in))

        #隠れ層
        for i in range(self.layer-1):
            w_h = np.random.rand(self.node, self.node)/5 - 0.1
            b_h = np.random.rand(self.node)/5 - 0.1
            self.net.append((w_h, b_h))

        #出力層
        w_out = np.random.rand(1, self.node)/5 - 0.1
        b_out = np.random.rand(1)/5 - 0.1
        self.net.append((w_out, b_out))

    def create_features(self, x):
        phi = [0]*len(self.ids)
        words = x.strip().split()
        for word in words:
            if self.ids["UNI:"+word] < len(phi):
                phi[self.ids["UNI:"+word]] += 1
            else:
                phi.append(1)
        return phi

    def forward_nn(self, phi0):
        #各層への入力ベクトル
        phi = [np.array(phi0)]
        for i in range(len(self.net)):
            w, b = self.net[i]
            phi.append(np.tanh(np.dot(w, phi[i].T)) + b)
        return phi

    def backward_nn(self, phi, y_prime):
        J = len(self.net)
        #伝播されるベクトル
        #J+1個の配列
        delta = [0]*J
        delta.append(np.array([y_prime - phi[J][0]]))
        delta_prime = [0]*(J+1)
        for i in range(J)[::-1]:
            delta_prime[i+1] = delta[i+1]*(1-phi[i+1]**2)
            w, b = self.net[i]
            delta[i] = np.dot(delta_prime[i+1], w)
        return delta_prime

    def update_weights(self, phi, delta_prime, lamb):
        for i in range(len(self.net)):
            w, b = self.net[i]
            w += lamb*np.outer(delta_prime[i+1], phi[i])
            b += lamb*delta_prime[i+1]

    def fit(self, data, lr=0.1, iter=10):
        self.init_net(data)
        for i in range(iter):
            for (phi0, y) in self.feat_lab:
                phi = self.forward_nn(phi0)
                delta_prime = self.backward_nn(phi, y)
                self.update_weights(phi, delta_prime, lr)

        with open("weight_file", "w", encoding="utf-8") as wf, \
                open("id_file", "w", encoding="utf-8") as idf:
            for i, x in enumerate(self.net):
                wf.write(f"{i+1}\n{x}\n")
            for key, value in self.ids.items():
                idf.write(f"{value}\t{key}\n")

    #推論

    def test_create_features(self, x):
        phi = [0]*len(self.ids)
        words=x.strip().split()
        for word in words:
            if "UNI:"+word in self.ids:
                phi[self.ids["UNI:"+word]] += 1
        #未知語は放置？
        return phi

    def predict(self, data, ans):
        with open(data, encoding="utf-8") as f,\
                open(ans, "w", encoding="utf-8") as of:
            for line in f:
                phi = self.test_create_features(line)
                score = self.forward_nn(phi)[-1]
                if score >= 0:
                    y = 1
                else:
                    y = -1
                of.write(f"{y}\n")



if __name__ == "__main__":
    #training_file = "../../test/03-train-input.txt"
    training_file = "../../data/titles-en-train.labeled"
    nn = NeuralNetwork(layer=2, node=3)
    nn.fit(training_file, lr = 0.01 ,iter=1)
    test_file = "../../data/titles-en-test.word"
    ans_file = "my_answer"
    nn.predict(test_file, ans_file)

"""
../../script/grade-prediction.py ../../data/titles-en-test.labeled my_answer
layer: 1, node: 2, 学習率: 0.01, イテレーション: 1, シード: 1
Accuracy = 91.852639%
"""

"""
../../script/grade-prediction.py ../../data/titles-en-test.labeled my_answer
layer: 2, node: 2, 学習率: 0.01, イテレーション: 1, シード: 1
Accuracy = 90.896210%
"""

"""
../../script/grade-prediction.py ../../data/titles-en-test.labeled my_answer
layer: 1, node: 3, 学習率: 0.01, イテレーション: 1, シード: 1
error
"""

"""
../../script/grade-prediction.py ../../data/titles-en-test.labeled my_answer
layer: 1, node: 2, 学習率: 0.001, イテレーション: 2, シード: 1
Accuracy = 90.329437%
"""

"""
../../script/grade-prediction.py ../../data/titles-en-test.labeled my_answer
layer: 2, node: 3, 学習率: 0.01, イテレーション: 1, シード: 1
Accuracy = Accuracy = 92.171449%
"""

"""
../../script/grade-prediction.py ../../data/titles-en-test.labeled my_answer
layer: 2, node: 3, 学習率: 0.01, イテレーション: 1, シード: 7
Accuracy = 92.206872%
"""