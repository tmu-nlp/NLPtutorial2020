r"""tutorial01.py
train-unigram: 1-gram モデルを学習
test-unigram: 1-gram モデルを読み込み、エントロピーとカバレージを計算

[Ref]
- マングリング
    - https://qiita.com/mounntainn/items/e3fb1a5757c9cf7ded63

[Small]
NAME=test
CORPUS_PATH=./test/01-train-input.txt
MODEL_PATH=./model_test.txt
TRAIN_ANSWER_PATH=./test/01-train-answer.txt
TEST_PATH=./test/01-test-input.txt
TEST_ANSWER_PATH=./test/01-test-answer.txt

[Large]
NAME=wiki
CORPUS_PATH=./data/wiki-en-train.word
MODEL_PATH=./model_wiki.txt
TEST_PATH=./data/wiki-en-test.word

[Usage]
python tutorial01.py train ../../$CORPUS_PATH $MODEL_PATH
diff -s $MODEL_PATH ../../$TRAIN_ANSWER_PATH
python tutorial01.py test $MODEL_PATH ../../$TEST_PATH --name=$NAME | pbcopy
code ../../$TEST_ANSWER_PATH
"""
import argparse
import collections
import math
import os
import sys
from typing import Callable, Dict, List, Optional, Type, TypeVar

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
import kiyuna.utils.pickle as pickle  # noqa: E402 isort:skip
from kiyuna.utils.message import message  # noqa: E402 isort:skip
from kiyuna.utils.time import Timer  # noqa: E402 isort:skip

T = TypeVar("T")
F = Callable[[str], T]
Prob = TypeVar("Prob", bound=float)
Model = Dict[T, Prob]


class UnigramLM:
    model: Model
    trans: F
    BOS: Optional[str]
    EOS: Optional[str]

    def __init__(
        self,
        *,
        trans: F = str,
        BOS: Optional[str] = "<s>",
        EOS: Optional[str] = "</s>",
    ) -> None:
        self.trans = trans
        self.BOS = BOS
        self.EOS = EOS

    def _get_words(self, line: str) -> List[str]:
        """append BOS and EOS
        """
        words = line.strip().split()
        if self.BOS:
            words = [self.BOS] + words
        if self.EOS:
            words = words + [self.EOS]
        return words

    def train(self, path_corpus: str) -> Type["Unigram"]:
        self.model = self.__build_model(path_corpus)
        message(f"train model from {path_corpus}", type="success")
        return self

    def load(self, path_model: str) -> Type["Unigram"]:
        self.model = self.__load_model(path_model)
        message(f"load model from {path_model}", type="success")
        return self

    def __build_model(self, path_corpus: str) -> Model:
        # cnter = collections.Counter()
        cnter = collections.defaultdict(int)
        W_train = 0
        with open(path_corpus) as f:
            for line in f:
                words = self._get_words(line)
                # cnter += collections.Counter(map(self.trans, words))
                # W_train += len(words)
                for token in map(self.trans, words):
                    cnter[token] += 1
                    W_train += 1
        model = {token: freq / W_train for token, freq in cnter.items()}
        return model

    def __load_model(self, path_model: str) -> Model:
        model = {}
        with open(path_model) as f:
            for line in f:
                token, prob = line.split("\t")
                model[token] = float(prob)
        return model

    def dump(self, path_model: str) -> None:
        with pickle.SaveHelper(path_model), open(path_model, "w") as f:
            for token, prob in sorted(self.model.items()):
                f.write(f"{token}\t{prob:f}\n")

    def test(
        self, path_test: str, vocab_size: int = 1_000_000, λ_unk: float = 0.05
    ) -> Dict[str, float]:
        r"""calculate entropy, perplexity and coverage
        :math:`P(w_i) = λ_1 * P_ML(w_i) + (1 − λ_1) / V`
        """
        V: int = vocab_size  # 未知語を含む語彙数
        λ_1: float = 1 - λ_unk
        W_test: int = 0  # 単語数
        unk: int = 0  # 未知語の数
        nll: float = 0  # 負の対数尤度（negative log-likelihood）

        with open(path_test) as f:
            for line in f:
                words = self._get_words(line)
                for token in map(self.trans, words):
                    W_test += 1
                    P = λ_unk / V
                    if token in self.model:
                        P += λ_1 * self.model[token]
                    else:
                        unk += 1
                    nll += -math.log2(P)

        ret = {}
        ret["entropy_H"] = nll / W_test
        ret["perplexity_PPL"] = 2 ** ret["entropy_H"]
        ret["coverage"] = (W_test - unk) / W_test
        return ret


def train(args):
    with Timer():
        UnigramLM(BOS=None).train(args.corpus).dump(args.model)


def test(args):
    with Timer():
        res = UnigramLM(BOS=None).load(args.model).test(args.test)

    if args.name:
        print(f"[{args.name}]")
    for k, v in res.items():
        print(f"{k:15s} = {v:f}")

    """result
    [test]
    entropy_H       = 6.709899
    perplexity_PPL  = 104.684170
    coverage        = 0.800000

    [wiki]
    entropy_H       = 10.526656
    perplexity_PPL  = 1475.160635
    coverage        = 0.895226
    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Unigram Language Model")
    subparsers = parser.add_subparsers(help="sub-comman help")

    parser_train = subparsers.add_parser("train", help="see `train -h`")
    parser_train.add_argument("corpus", help="the location of a corpus file")
    parser_train.add_argument(
        "model", help="a file location in which to save the LM"
    )
    parser_train.set_defaults(handler=train)

    parser_test = subparsers.add_parser("test", help="see `test -h`")
    parser_test.add_argument("model", help="the location of a model file")
    parser_test.add_argument("test", help="the location of a test set")
    parser_test.add_argument("--name", help="set the display name")
    parser_test.set_defaults(handler=test)

    args = parser.parse_args()
    if hasattr(args, "handler"):
        args.handler(args)
    else:
        parser.print_help()
