import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import accuracy_score

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
        for i in tqdm(range(iter)):
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

            check_score("../../data/titles-en-test.labeled", self.predict("../../data/titles-en-test.word", f"my_answer_ep{i:02}"))

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
        my_ans = []
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
                my_ans.append(y)
        return my_ans

def check_score(ans_file, my_ans):
    ans = []
    with open(ans_file, encoding="utf-8") as f:
        for line in f:
            label=int(line.split("\t")[0])
            ans.append(label)
    ans = np.array(ans)
    my_ans = np.array(my_ans)
    print(f"accuracy: {accuracy_score(ans, my_ans)}")

if __name__ == "__main__":
    #training_file = "../../test/03-train-input.txt"
    training_file = "../../data/titles-en-train.labeled"
    nn = NeuralNetwork(layer=1, node=2)
    nn.fit(training_file, lr = 0.001 ,iter=10)
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

"""
layer: 1, node: 2, 学習率: 0.001, イテレーション: 10, シード: 7
  0%|                                                                                                                                | 0/10 [00:00<?, ?it/s]
  accuracy: 0.889833510449876
 10%|████████████                                                                                                            | 1/10 [00:44<06:42, 44.67s/it]
 accuracy: 0.9025859015232023
 20%|████████████████████████                                                                                                | 2/10 [01:28<05:55, 44.46s/it]
 accuracy: 0.9139213602550478
 30%|████████████████████████████████████                                                                                    | 3/10 [02:12<05:09, 44.23s/it]
 accuracy: 0.9210060219624513
 40%|████████████████████████████████████████████████                                                                        | 4/10 [02:57<04:27, 44.60s/it]
 accuracy: 0.9256110520722636
 50%|████████████████████████████████████████████████████████████                                                            | 5/10 [03:43<03:44, 44.95s/it]
 accuracy: 0.9284449167552249
 60%|████████████████████████████████████████████████████████████████████████                                                | 6/10 [04:27<02:58, 44.63s/it]
 accuracy: 0.9305703152674459
 70%|████████████████████████████████████████████████████████████████████████████████████                                    | 7/10 [05:15<02:16, 45.65s/it]
 accuracy: 0.932695713779667
 80%|████████████████████████████████████████████████████████████████████████████████████████████████                        | 8/10 [05:59<01:30, 45.10s/it]
 accuracy: 0.9337584130357776
 90%|████████████████████████████████████████████████████████████████████████████████████████████████████████████            | 9/10 [06:42<00:44, 44.60s/it]
 accuracy: 0.9337584130357776
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [07:26<00:00, 44.64s/it]
"""