r"""tutorial07.py
- train-nn: NN を学習するプログラム
- test-nn: NN を用いて予測するプログラム

[Ref]
- https://github.com/oreilly-japan/deep-learning-from-scratch-2/blob/master/common/optimizer.py

[Result]
Accuracy = 90.967056% (#05 Perceptron)
Accuracy = 93.765498% (#06 SVM)
#07 NN(optimizer).fit(corpus=train_path, dimensions=[2, 1], epochs=10)
Accuracy = 94.899044%; SGD(lr=0.1)
Accuracy = 94.757350%; SGD(lr=0.01)
Accuracy = 94.226001%; Momentum(lr=0.01)
Accuracy = 94.332271%; Momentum(lr=0.001)
Accuracy = 92.950762%; Adam(lr=0.001)
Accuracy = 94.544810%; Adam(lr=0.0001)
"""
import logging
import subprocess
import tempfile
from collections import defaultdict

import numpy as np
import tqdm

seed = 2020
np.random.seed(seed)
logging.basicConfig(level=logging.DEBUG)


class Dataset:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.idxs = np.arange(len(self))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        idx = self.idxs[idx]
        return self.X[idx].reshape(1, -1), self.y[idx]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def shuffle(self):
        np.random.shuffle(self.idxs)


class Affine:
    def __init__(self, d_in, d_out):
        self.W = np.random.rand(d_in, d_out) / 5 - 0.1
        self.b = np.random.rand(1, d_out) / 5 - 0.1
        self.params = [self.W, self.b]
        self.grads = [np.zeros_like(self.W), np.zeros_like(self.b)]
        self.x = None

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        dx = dout @ self.W.T
        dW = self.x.T @ dout
        db = np.sum(dout, axis=0)
        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx


class Tanh:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        self.out = np.tanh(x)
        return self.out

    def backward(self, dout):
        return dout * (1 - self.out ** 2)


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def __call__(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]


class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def __call__(self, params, grads):
        if self.v is None:
            self.v = [np.zeros_like(param) for param in params]
        for i in range(len(params)):
            self.v[i] = self.momentum * self.v[i] - self.lr * grads[i]
            params[i] += self.v[i]


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def __call__(self, params, grads):
        if self.m is None:
            self.m, self.v = [], []
            for param in params:
                self.m.append(np.zeros_like(param))
                self.v.append(np.zeros_like(param))
        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)
        for i in range(len(params)):
            self.m[i] += (1 - self.beta1) * (grads[i] - self.m[i])
            self.v[i] += (1 - self.beta2) * (grads[i] ** 2 - self.v[i])
            params[i] -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + 1e-7)


class NN:
    def __init__(self, optimizer):
        self.ids = {}
        self.layers = []
        self.optimizer = optimizer
        self.params = []
        self.grads = []

    def preprocess(self, corpus):
        ids = defaultdict(lambda: len(ids))
        labels, features = [], []
        for line in open(corpus):
            label, sentence = line.strip().split("\t")
            labels.append(int(label))
            phi = defaultdict(int)
            words = sentence.split(" ")
            for word in words:
                phi[ids[f"UNI:{word}"]] += 1
            features.append(phi)
        y = np.array(labels)
        X = np.zeros((len(features), len(ids)))
        for i, phi in enumerate(features):
            for idx, freq in phi.items():
                X[i][idx] = freq
        return Dataset(X, y), dict(ids)

    def init_net(self, dimensions):
        assert self.ids is not None
        d_in = len(self.ids)
        for d_out in dimensions:
            self.layers.append(Affine(d_in, d_out))
            self.layers.append(Tanh())
            d_in = d_out
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def fit(self, corpus, dimensions, epochs):
        train, self.ids = self.preprocess(corpus)
        self.init_net(dimensions)
        for epoch in range(1, epochs + 1):
            logging.info(f"Epoch {epoch}")
            train.shuffle()
            err_train = 0
            acc_train = 0
            for cnt, (phi, y) in tqdm.tqdm(enumerate(train, start=1), leave=False):
                phi = self.forward(phi)
                err_train += self.criterion(phi, y).item()
                acc_train += (self.sign(phi) == y).sum().item()
                self.backward(phi - y)
                self.update()
            err_train /= cnt
            acc_train /= cnt
            logging.info(f"  train -- Err: {err_train:f}, Acc: {acc_train:f}")
        return self

    def forward(self, phi):
        for layer in self.layers:
            phi = layer.forward(phi)
        return phi

    def backward(self, delta):
        for layer in self.layers[::-1]:
            delta = layer.backward(delta)

    def update(self):
        self.optimizer(self.params, self.grads)

    def criterion(self, preds, labels):
        return ((preds - labels) ** 2 / 2).sum()

    def create_features(self, sentence):
        phi = np.zeros((len(self.ids)))
        words = sentence.split(" ")
        for word in words:
            feature = f"UNI:{word}"
            if feature in self.ids:
                phi[self.ids[feature]] += 1
        return phi

    def sign(self, val):
        return 1 * (val >= 0) - 1 * (val < 0)

    def predict(self, path):
        preds = []
        for line in open(path):
            sentence = line.strip()
            phi = self.create_features(sentence)
            preds.append(self.sign(self.forward(phi)).item())
        return preds


if __name__ == "__main__":
    # train_path = "../../test/03-train-input.txt"
    train_path = "../../data/titles-en-train.labeled"
    test_path = "../../data/titles-en-test.word"
    optimizer = SGD(lr=0.1)
    nn = NN(optimizer).fit(corpus=train_path, dimensions=[2, 1], epochs=10)
    res = "\n".join(map(str, nn.predict(test_path)))
    # output_path = "./my_answer.txt"
    # with open(output_path, "w") as f:
    #     f.write(res)
    with tempfile.NamedTemporaryFile() as fp:
        fp.write(res.encode())
        acc = subprocess.run(
            [
                "python2",
                "../../script/grade-prediction.py",
                "../../data/titles-en-test.labeled",
                fp.name,
            ],
            encoding="utf-8",
            stdout=subprocess.PIPE,
        ).stdout
    print(acc)


"""
INFO:root:Epoch 1
INFO:root:  train -- Err: 0.161455, Acc: 0.895730
INFO:root:Epoch 2
INFO:root:  train -- Err: 0.117362, Acc: 0.925496
INFO:root:Epoch 3
INFO:root:  train -- Err: 0.104727, Acc: 0.934266
INFO:root:Epoch 4
INFO:root:  train -- Err: 0.092936, Acc: 0.943657
INFO:root:Epoch 5
INFO:root:  train -- Err: 0.082965, Acc: 0.950567
INFO:root:Epoch 6
INFO:root:  train -- Err: 0.076538, Acc: 0.955351
INFO:root:Epoch 7
INFO:root:  train -- Err: 0.069201, Acc: 0.961286
INFO:root:Epoch 8
INFO:root:  train -- Err: 0.078261, Acc: 0.953225
INFO:root:Epoch 9
INFO:root:  train -- Err: 0.061792, Acc: 0.965273
INFO:root:Epoch 10
INFO:root:  train -- Err: 0.057730, Acc: 0.966425
Accuracy = 94.899044%
"""
