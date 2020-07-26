r"""tutorial08.py
系列ラベリングのための RNN を実装
学習 train-rnn とテスト test-rnn

[Ref]
- https://github.com/oreilly-japan/deep-learning-from-scratch-2/tree/master/ch05

[Result]
Accuracy = 90.967056% (#05 Perceptron)
Accuracy = 93.765498% (#06 SVM)
Accuracy = 94.899044% (#07 NN)
Accurary = 80.89%     (#08 RNN - 完全勾配計算 - 昨年実装しました)
Accuracy = 26.58%     (#08 RNN - BPTT - SGD(lr=0.1))
Accuracy = 89.96%     (#08 RNN - BPTT - Adam(lr=0.001))
"""
import logging
import os
import subprocess
import sys
import tempfile
from collections import defaultdict

import numpy as np
import tqdm

from nn import RNN, TimeAffine, TimeEmbedding, TimeRNN, TimeSoftmaxWithLoss

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from tutorial07.tutorial07 import Dataset, SGD, Momentum, Adam  # noqa: E402 isort:skip


seed = 2020
np.random.seed(seed)
logging.basicConfig(level=logging.DEBUG)


class RnnTrainer:
    def __init__(self, optimizer):
        self.UNK = "<UNK>"
        self.EOS = "<EOS>"
        self.word2id = None
        self.label2id = None
        self.layers = []
        self.rnn_layer = None
        self.loss_layer = None
        self.optimizer = optimizer
        self.params = []
        self.grads = []

    def preprocess(self, path):
        word2id = defaultdict(lambda: len(word2id))
        word2id[self.UNK]
        word2id[self.EOS]
        label2id = defaultdict(lambda: len(label2id))
        label2id[self.EOS]
        xs, ts = [], []
        for line in open(path):
            words, labels = [], []
            for word, label in map(lambda x: x.split("_"), line.split()):
                words.append(word2id[word])
                labels.append(label2id[label])
            words.append(word2id[self.EOS])
            labels.append(label2id[self.EOS])
            xs.extend(words)
            ts.extend(labels)
        self.word2id, self.label2id = dict(word2id), dict(label2id)
        return Dataset(np.array(xs), np.array(ts))

    def init_net(self, hidden_size):
        V = D = len(self.word2id)
        H = hidden_size
        O = len(self.label2id)
        rn = np.random.randn
        embed_W = (rn(V, D) / 100).astype("f")
        rnn_Wx = (rn(D, H) / np.sqrt(D)).astype("f")
        rnn_Wh = (rn(H, H) / np.sqrt(H)).astype("f")
        rnn_b = np.zeros(H).astype("f")
        affine_W = (rn(H, O) / np.sqrt(H)).astype("f")
        affine_b = np.zeros(O).astype("f")
        self.layers = [
            TimeEmbedding(embed_W),
            TimeRNN(rnn_Wx, rnn_Wh, rnn_b, stateful=True),
            TimeAffine(affine_W, affine_b),
        ]
        self.rnn_layer = self.layers[1]
        self.loss_layer = TimeSoftmaxWithLoss()
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def fit(self, corpus, hidden_size, epochs, batch_size=512):
        time_size = 5  # Truncated BPTTの展開する時間サイズ
        train = self.preprocess(corpus)
        self.init_net(hidden_size)
        corpus_size = len(train)
        jump = (corpus_size - 1) // batch_size
        offsets = [i * jump for i in range(batch_size)]
        max_iters = corpus_size // (batch_size * time_size)
        time_idx = 0
        for epoch in range(1, epochs + 1):
            logging.info(f"Epoch {epoch}")
            # train.shuffle()
            xs, ts = train.X, train.y
            loss = []
            for it in tqdm.tqdm(range(max_iters), leave=False):
                batch_x = np.empty((batch_size, time_size), dtype="i")
                batch_t = np.empty((batch_size, time_size), dtype="i")
                for t in range(time_size):
                    for i, offset in enumerate(offsets):
                        batch_x[i, t] = xs[(offset + time_idx) % corpus_size]
                        batch_t[i, t] = ts[(offset + time_idx) % corpus_size]
                    time_idx += 1
                loss.append(self.forward(batch_x, batch_t))
                self.backward()
                self.optimizer(self.params, self.grads)
            loss = np.mean(loss)
            logging.info(f"  train -- Loss: {loss:f}")
        return self

    def forward(self, xs, ts):
        for layer in self.layers:
            xs = layer.forward(xs)
        loss = self.loss_layer.forward(xs, ts)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def predict(self, path):
        id2label = {id_: label for label, id_ in self.label2id.items()}
        for line in open(path):
            ids = [self.word2id.get(w, 0) for w in line.split()] + [self.word2id[self.EOS]]
            xs = np.array(ids)[np.newaxis]
            for layer in self.layers:
                xs = layer.forward(xs)
            labels = [id2label[id_] for id_ in np.argmax(xs.squeeze(), axis=1)]
            labels.pop()
            print(labels, file=sys.stderr)
            yield " ".join(labels)


if __name__ == "__main__":
    # train_path = "../../test/05-train-input.txt"
    # test_path = "../../test/05-test-input.txt"
    train_path = "../../data/wiki-en-train.norm_pos"
    test_path = "../../data/wiki-en-test.norm"
    # optimizer = SGD(lr=0.1)
    # optimizer = Momentum(lr=0.01)
    optimizer = Adam(lr=0.001)
    nn = RnnTrainer(optimizer).fit(corpus=train_path, hidden_size=64, epochs=10)
    nn.rnn_layer.reset_state()
    res = "\n".join(map(str, nn.predict(test_path)))
    # output_path = "./my_answer.txt"
    # with open(output_path, "w") as f:
    #     f.write(res)
    with tempfile.NamedTemporaryFile() as fp:
        fp.write(res.encode())
        acc = subprocess.run(
            ["perl", "../../script/gradepos.pl", "../../data/wiki-en-test.pos", fp.name],
            encoding="utf-8",
            stdout=subprocess.PIPE,
        ).stdout
    print(acc)


"""result: SGD(lr=0.1)
INFO:root:Epoch 1
INFO:root:  train -- Loss: 3.586725
INFO:root:Epoch 2
INFO:root:  train -- Loss: 3.104531
INFO:root:Epoch 3
INFO:root:  train -- Loss: 2.987290
INFO:root:Epoch 4
INFO:root:  train -- Loss: 2.949515
INFO:root:Epoch 5
INFO:root:  train -- Loss: 2.923757
INFO:root:Epoch 6
INFO:root:  train -- Loss: 2.899622
INFO:root:Epoch 7
INFO:root:  train -- Loss: 2.873066
INFO:root:Epoch 8
INFO:root:  train -- Loss: 2.842058
INFO:root:Epoch 9
INFO:root:  train -- Loss: 2.804854
INFO:root:Epoch 10
INFO:root:  train -- Loss: 2.760474
['NN', 'NN', 'NN', ',', 'NN', 'NN', 'NN', 'NN', 'NN', 'NN', 'NN', 'NN', 'NN', 'IN', 'NN', 'NN', 'NN', 'NN', 'NN', 'NN', 'DT', 'NN', 'IN', 'NN', 'NN', 'NN', 'IN', 'DT', 'NN', 'NN', 'NN', 'NN', 'NN', 'NN', 'NN', 'NN', 'NN', 'NN', 'NN', 'NN', 'DT', 'NN', 'NN', 'NN', 'NN', 'NN', 'NN', 'NN', 'NN']
...
Accuracy: 26.58% (1213/4563)

Most common mistakes:
JJ --> NN       414
NNS --> NN      395
IN --> NN       387
RB --> NN       206
DT --> NN       200
, --> NN        171
. --> NN        171
VBN --> NN      164
CC --> NN       130
VBZ --> NN      124
"""

"""result: Adam(lr=0.001)
INFO:root:Epoch 1
INFO:root:  train -- Loss: 3.100499
INFO:root:Epoch 2
INFO:root:  train -- Loss: 1.571680
INFO:root:Epoch 3
INFO:root:  train -- Loss: 0.757226
INFO:root:Epoch 4
INFO:root:  train -- Loss: 0.381059
INFO:root:Epoch 5
INFO:root:  train -- Loss: 0.220701
INFO:root:Epoch 6
INFO:root:  train -- Loss: 0.147566
INFO:root:Epoch 7
INFO:root:  train -- Loss: 0.111867
INFO:root:Epoch 8
INFO:root:  train -- Loss: 0.091992
INFO:root:Epoch 9
INFO:root:  train -- Loss: 0.079182
INFO:root:Epoch 10
INFO:root:  train -- Loss: 0.070049
['IN', 'JJ', 'NNS', ',', 'NNP', 'NN', '-LRB-', 'NN', '-RRB-', 'VBZ', 'DT', 'JJ', 'NN', 'IN', 'JJ', 'NN', 'NN', ',', 'WDT', 'VBZ', 'DT', 'NN', 'IN', 'VBG', 'WDT', 'NN', 'IN', 'DT', 'NN', '-LRB-', 'FW', 'NN', '-RRB-', 'VBZ', 'VBN', 'IN', 'DT', 'NN', ',', 'WRB', 'DT', 'NN', 'VBZ', 'JJ', 'NNS', '-LRB-', 'VB', '-RRB-', '.']
Accuracy: 89.96% (4105/4563)

Most common mistakes:
NNS --> NN      38
NN --> JJ       35
JJ --> NN       23
NN --> NNP      23
JJ --> NNP      18
JJ --> VBN      17
NNS --> NNP     13
NNP --> NN      13
NNS --> JJ      11
NNP --> JJ      11
"""
