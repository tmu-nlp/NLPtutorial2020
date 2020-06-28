r"""tutorial05.py
２つのプログラムを作成
- train-perceptron: パーセプトロンを用いた分類器学習
- test-perceptron: 重みを読み込み、予測を１行ずつ出力

[Small]
# train
CORPUS_PATH=./test/03-train-input.txt
MODEL_PATH=./model_test.txt
python tutorial05.py train ../../$CORPUS_PATH $MODEL_PATH
diff -s $MODEL_PATH ../../test/03-train-answer.txt

[Large]
echo "[Result]" > out
for epoch in `seq 1 3`; do
# train
CORPUS_PATH=./data/titles-en-train.labeled
MODEL_PATH=./model_titles.txt
python tutorial05.py train ../../$CORPUS_PATH $MODEL_PATH --epoch=$epoch
# test
MODEL_PATH=./model_titles.txt
INPUT_PATH=./data/titles-en-test.word
OUTPUT_PATH=./result_titles.labeled
python tutorial05.py test $MODEL_PATH ../../$INPUT_PATH $OUTPUT_PATH
result=$(python2 ../../script/grade-prediction.py ../../data/titles-en-test.labeled $OUTPUT_PATH)
echo "epoch $epoch \t-> $result" >> out
done
cat out | pbcopy; rm out
# make_graph
CMD=(python2 ../../script/grade-prediction.py ../../data/titles-en-test.labeled)
python tutorial05.py graph ../../$CORPUS_PATH ../../$INPUT_PATH $CMD --max_epoch=100 > out

[Result]
epoch 1 	-> Accuracy = 90.967056%
epoch 2 	-> Accuracy = 91.781792%
epoch 3 	-> Accuracy = 92.773645%
"""
import argparse
import subprocess
import sys
import tempfile
from collections import defaultdict

import matplotlib.pyplot as plt
import tqdm


def verbose(s):
    if False:
        print(s, file=sys.stderr)


class Perceptron:
    __slots__ = ["w"]

    def __init__(self):
        self.w = defaultdict(float)

    def train(self, corpus_file, epoch=1):
        for _ in tqdm.tqdm(range(epoch)):
            list(self.gen_predict_all(corpus_file))
        return self

    def test(self, input_file, output_file):
        with open(output_file, "w") as f:
            f.writelines(
                f"{label}\t{sent}\n"
                for label, sent in self.gen_predict_all(input_file, eval_mode=True)
            )

    def read_line(self, input_file, eval_mode=False):
        with open(input_file) as f:
            for line in map(lambda x: x.strip(), f):
                if eval_mode:
                    yield None, line
                else:
                    y_label, x_sentence = line.split("\t")
                    yield int(y_label), x_sentence

    def gen_predict_all(self, input_file, eval_mode=False):
        for y_label, x_sentence in self.read_line(input_file, eval_mode):
            phi = self.create_features(x_sentence)
            y_predicted = self.predict_one(phi)
            if eval_mode:
                yield y_predicted, x_sentence
            elif y_predicted != y_label:
                yield self.update_weights(phi, y_label)

    def create_features(self, sentence):
        phi = defaultdict(int)
        for word in sentence.split():
            phi[f"UNI:{word}"] += 1
        return phi

    def sign(self, v):
        return 1 if v >= 0 else -1

    def predict_one(self, phi):
        score = sum(self.w[x] * phi[x] for x in phi)
        return self.sign(score)

    def update_weights(self, phi, y):
        verbose("-" * 60)
        for x in phi:
            verbose(f"{x:35}\t{self.w[x]: 5.1f} ->{self.w[x] + phi[x] * y: 5.1f}")
            self.w[x] += phi[x] * y

    def dump(self, weights_file):
        with open(weights_file, "w") as f:
            f.writelines(f"{k}\t{v:f}\n" for k, v in sorted(self.w.items()))

    def load(self, weights_file):
        with open(weights_file) as f:
            for line in f:
                key, value = line.split("\t")
                self.w[key] = float(value)
        return self


def train(args):
    Perceptron().train(args.corpus, args.epoch).dump(args.weights)


def test(args):
    Perceptron().load(args.weights).test(args.input, args.output)


def make_graph(args):
    accs = []
    p = Perceptron()
    for _ in tqdm.tqdm(range(args.max_epoch)):
        list(p.gen_predict_all(args.corpus))
        res = "\n".join(str(label) for label, _ in p.gen_predict_all(args.input, True))
        with tempfile.NamedTemporaryFile() as fp:
            fp.write(res.encode())
            ret = subprocess.run(
                args.command + [fp.name], encoding="utf-8", stdout=subprocess.PIPE
            ).stdout
        acc = float(ret.replace("Accuracy = ", "").replace("%\n", ""))
        accs.append(acc)
    print("Epoch\tAccuracy")
    for i, acc in enumerate(accs, start=1):
        print(f"{i}\t{acc}")
    plt.plot(range(1, args.max_epoch + 1), accs)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    # plt.ylim(0, 100)
    plt.savefig(f"fig_{args.max_epoch}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Perceptron")
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
    parser_graph.add_argument("command", type=str, nargs="*", help="スコア計算のコマンド")
    parser_graph.add_argument("--max_epoch", type=int, default=10, help="最大エポック数")
    parser_graph.set_defaults(handler=make_graph)

    args = parser.parse_args()
    if hasattr(args, "handler"):
        args.handler(args)
    else:
        parser.print_help()
