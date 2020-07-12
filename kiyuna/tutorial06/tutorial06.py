r"""tutorial06.py
２つのプログラムを作成
- L1 正則化とマージンで学習を行う train-svm

[Small]
# train
CORPUS_PATH=./test/03-train-input.txt
MODEL_PATH=./model_test.txt
python tutorial06.py train ../../$CORPUS_PATH $MODEL_PATH
diff -s $MODEL_PATH ../../test/03-train-answer.txt

[Large]
# train
CORPUS_PATH=./data/titles-en-train.labeled
MODEL_PATH=./model_titles.txt
python tutorial06.py train ../../$CORPUS_PATH $MODEL_PATH
# test
MODEL_PATH=./model_titles.txt
INPUT_PATH=./data/titles-en-test.word
OUTPUT_PATH=./result_titles.labeled
python tutorial06.py test $MODEL_PATH ../../$INPUT_PATH $OUTPUT_PATH
python2 ../../script/grade-prediction.py ../../data/titles-en-test.labeled $OUTPUT_PATH
# make_graph
python tutorial06.py graph ../../$CORPUS_PATH ../../$INPUT_PATH ../../data/titles-en-test.labeled

[Result]
Accuracy = 90.967056% (#05 Perceptron)
Accuracy = 93.765498% (#06 SVM)
"""
import argparse
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from sklearn.metrics import accuracy_score

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from tutorial05.tutorial05 import Perceptron  # noqa: E402 isort:skip


class SVM(Perceptron):
    __slots__ = ["w", "params", "lazy_update", "total", "lazy"]
    __existing__ = [
        "train",
        "test",
        "sign",
        "create_features",
        "dump",
        "load",
    ]

    def __init__(self, *, margin=0, c=1e-4):
        super().__init__()
        # sign のせいで margin = 0 でも Perceptron と一致しない
        self.params = {"margin": margin, "c": c}
        self.lazy_update = False
        self.total = 0
        self.lazy = defaultdict(int)

    def l1_regularize(self, x, cnt):
        if abs(self.w[x]) < cnt * self.params["c"]:
            self.w[x] = 0
        else:
            self.w[x] -= self.sign(self.w[x]) * cnt * self.params["c"]

    def get_w(self, x):
        if self.lazy_update and self.total != self.lazy[x]:
            self.l1_regularize(x, self.total - self.lazy[x])
            self.lazy[x] = self.total
        return self.w[x]

    def dot_w(self, phi):
        score = sum(self.get_w(x) * phi[x] for x in phi)
        return score

    def update_weights(self, phi, y):
        self.total += 1
        super().update_weights(phi, y)

    def gen_predict_all(self, input_file, eval_mode=False):
        for y_label, x_sentence in self.read_line(input_file, eval_mode):
            phi = self.create_features(x_sentence)
            if eval_mode:
                yield self.predict_one(phi), x_sentence
            elif self.dot_w(phi) * y_label <= self.params["margin"]:  # 正しい分類なら正
                yield self.update_weights(phi, y_label)
        if not eval_mode and self.lazy_update:
            for x in self.w:
                self.get_w(x)


def train(args):
    SVM().train(args.corpus, args.epoch).dump(args.weights)


def test(args):
    SVM().load(args.weights).test(args.input, args.output)


def make_graph(args):
    y_true = [int(line.split()[0]) for line in open(args.answer)]
    epochs = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    margins = [0, 1, 5, 10, 15, 20, 25, 30]
    E = np.zeros((len(margins), len(epochs)))
    for i, epoch in tqdm.tqdm(enumerate(epochs), leave=False):
        for j, margin in tqdm.tqdm(enumerate(margins), leave=False):
            svm = SVM(margin=margin).train(args.corpus, epoch)
            y_pred = tuple(label for label, _ in svm.gen_predict_all(args.input, eval_mode=True))
            E[j][i] = float(accuracy_score(y_true, y_pred))
    fig, ax = plt.subplots()
    fig.colorbar(ax.pcolor(E, cmap="jet", edgecolors="k", alpha=0.8))
    ma_y, ma_x = np.where(E == E.max())
    ax.scatter(ma_x + 0.5, ma_y + 0.5, c="r", label="max")
    mi_y, mi_x = np.where(E == E.min())
    ax.scatter(mi_x + 0.5, mi_y + 0.5, c="b", label="min")
    print("[+] max:", E.max(), np.where(E == E.max()))
    print("[+] min:", E.min(), np.where(E == E.min()))
    ax.set_xticks(np.arange(len(epochs)) + 0.5, minor=False)
    ax.set_yticks(np.arange(len(margins)) + 0.5, minor=False)
    ax.set_xticklabels(epochs, minor=False)
    ax.set_yticklabels(margins, minor=False)
    ax.set_xlabel("$Epoch$")
    ax.set_ylabel("$Margin$")
    ax.set_aspect("equal")
    ax.legend(loc="lower right")
    ax.set_title("c = 0.0001")
    plt.savefig("result_.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="SVM")
    subparsers = parser.add_subparsers(help="sub-comman help")

    parser_train = subparsers.add_parser("train", help="see `train -h`")
    parser_train.add_argument("corpus", help="コーパスのパス")
    parser_train.add_argument("weights", help="重みの出力先")
    parser_train.add_argument("--epoch", type=int, default=1, help="the number of iterations")
    parser_train.set_defaults(handler=train)

    parser_test = subparsers.add_parser("test", help="see `test -h`")
    parser_test.add_argument("weights", help="重みのパス")
    parser_test.add_argument("input", help="二値予想したいファイルのパス")
    parser_test.add_argument("output", help="二値予想結果の出力先")
    parser_test.set_defaults(handler=test)

    parser_graph = subparsers.add_parser("graph", help="see `graph -h`")
    parser_graph.add_argument("corpus", help="コーパスのパス")
    parser_graph.add_argument("input", help="二値予想したいファイルのパス")
    parser_graph.add_argument("answer", help="Gold ファイルのパス")
    parser_graph.set_defaults(handler=make_graph)

    args = parser.parse_args()
    if hasattr(args, "handler"):
        args.handler(args)
    else:
        parser.print_help()
