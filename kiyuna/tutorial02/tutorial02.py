r"""tutorial02.py
train-bigram: 2-gram モデルを学習
test-bigram: 2-gram モデルに基づいて評価データのエントロピーを計算

[TODO]
- 任意な文脈長が利用可能なプログラム

[Small]
NAME=test
CORPUS_PATH=./test/02-train-input.txt
MODEL_PATH=./model_test.txt
TRAIN_ANSWER_PATH=./test/02-train-answer.txt

[Large]
NAME="wiki - bigram"
CORPUS_PATH=./data/wiki-en-train.word
MODEL_PATH=./model_wiki.txt
TEST_PATH=./data/wiki-en-test.word

[Usage]
python tutorial02.py train ../../$CORPUS_PATH $MODEL_PATH
diff -s $MODEL_PATH ../../$TRAIN_ANSWER_PATH
python tutorial02.py test $MODEL_PATH ../../$TEST_PATH --name=$NAME | pbcopy
python tutorial02.py test $MODEL_PATH ../../$TEST_PATH --name=$NAME --WittenBell | pbcopy
"""
import argparse
import collections
import math
import os
import sys
from typing import Callable, DefaultDict, Dict, Optional, Tuple, Type, TypeVar

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.message import message, Renderer  # noqa: E402 isort:skip
from kiyuna.utils.time import Timer  # noqa: E402 isort:skip
from kiyuna.common import n_gram  # noqa: E402 isort:skip
from kiyuna.tutorial01.tutorial01 import UnigramLM  # noqa: E402 isort:skip


T = TypeVar("T")
F = Callable[[str], T]
Prob = TypeVar("Prob", bound=float)
Model = DefaultDict[T, Prob]


class BigramLM(UnigramLM):
    model: Model
    n: int = 2
    trans: F
    BOS: str
    EOS: str
    WittenBell: bool
    updated_for_WB: bool

    def __init__(
        self,
        *,
        trans: F = str,
        BOS: str = "<s>",
        EOS: str = "</s>",
        WittenBell: bool = False,
    ) -> None:
        super().__init__()
        self.WittenBell = WittenBell
        self.updated_for_WB = False

    def train(self, path_corpus: str) -> Type["Bigram"]:
        self.model = self.__build_model(path_corpus)
        message(f"train model from {path_corpus}", type="success")
        return self

    def load(self, path_model: str) -> Type["Bigram"]:
        self.model = self.__load_model(path_model)
        message(f"load model from {path_model}", type="success")
        return self

    def __build_model(self, path_corpus: str) -> Model:
        cnter = collections.defaultdict(int)
        cnter_context = collections.defaultdict(int)
        with open(path_corpus) as f:
            for line in f:
                words = self._get_words(line)
                for ngram in n_gram(list(map(self.trans, words)), 2):
                    # 2-gram
                    cnter[" ".join(ngram)] += 1
                    cnter_context[" ".join(ngram[:-1])] += 1
                    # 1-gram
                    cnter[ngram[-1]] += 1
                    cnter_context[""] += 1
        model = collections.defaultdict(float)
        for ngram, freq in cnter.items():
            context = " ".join(ngram.split()[:-1])
            prob = cnter[ngram] / cnter_context[context]
            model[ngram] = prob
            model["cnt::" + ngram] = cnter[ngram]
        return model

    def __load_model(self, path_model: str) -> Model:
        model = collections.defaultdict(float)
        with open(path_model) as f:
            for line in f:
                ngram, prob = line.split("\t")
                model[ngram] = float(prob)
        return model

    def _check_update(self):
        if not self.WittenBell:
            return
        if self.updated_for_WB:
            return
        for ngram in tuple(self.model.keys()):
            if ngram.startswith("cnt::"):
                continue
            context = " ".join(ngram.split()[:-1])
            self.model["cnt::" + context + " *"] += 1
        self.updated_for_WB = True

    def test(
        self,
        path_test: str,
        *,
        vocab_size: int = 1_000_000,
        λ_1: float = 0.95,
        λ_2: float = 0.95,
    ) -> Dict[str, float]:
        r"""calculate entropy, perplexity and coverage
        :math:`P(w_i ∣ w_{i−1}) = λ2 P_ML(w_i ∣ w_{i−1}) + (1 − λ_2) P(w_i)`
        :math:`P(w_i) = λ_1 P_ML(w_i) + (1 − λ_1) / N`
        """
        V: int = vocab_size  # 未知語を含む語彙数
        W_test: int = 0  # 単語数
        unk: int = 0  # 未知語の数
        nll: float = 0  # 負の対数尤度（negative log-likelihood）

        def get_λ_2(context: str) -> float:
            r"""
            :math:`λ_{w_{i-1}} = \frac{c(w_{i-1})}{u(w_{i-1}) + c(w_{i-1})}`
            """
            if not self.WittenBell:
                return λ_2
            c = self.model["cnt::" + context]
            u = self.model["cnt::" + context + " *"]
            if c == 0:
                return λ_2
            return c / (u + c)

        self._check_update()
        with open(path_test) as f:
            for line in f:
                words = self._get_words(line)
                for ngram in n_gram(list(map(self.trans, words)), 2):
                    λ_2_ = get_λ_2(" ".join(ngram[:-1]))
                    W_test += 1
                    P1 = λ_1 * self.model[ngram[-1]] + (1 - λ_1) / V
                    P2 = λ_2_ * self.model[" ".join(ngram)] + (1 - λ_2_) * P1
                    nll += -math.log2(P2)
                    unk += self.model[" ".join(ngram)] == float()

        ret = {}
        ret["entropy_H"] = nll / W_test
        ret["perplexity_PPL"] = 2 ** ret["entropy_H"]
        ret["coverage"] = (W_test - unk) / W_test
        return ret


def train(args: argparse.Namespace) -> None:
    with Timer():
        BigramLM().train(args.corpus).dump(args.model)


def get_ext(WittenBell: bool):
    return "WittenBell" + ("なし", "あり")[WittenBell]


def grid_search(
    model: Model,
    path_test: str,
    *,
    rng: Optional[Tuple[float, float, float]] = None,
    rng1: Optional[Tuple[float, float, float]] = None,
    rng2: Optional[Tuple[float, float, float]] = None,
    save: Optional[str] = None,
) -> Tuple[float, float]:
    def get_param(idx: np.ndarray) -> np.ndarray:
        return (
            np.array([rng1[0], rng2[0]]) + np.array([rng1[2], rng2[2]]) * idx
        )

    if rng:
        rng1 = rng2 = rng
    assert rng1 is not None
    assert rng2 is not None

    with Renderer("grid search") as out:
        cnt1 = len(np.arange(*rng1))
        cnt2 = len(np.arange(*rng2))
        E = np.zeros((cnt2, cnt1))
        for j, λ_2 in enumerate(np.arange(*rng2)):
            message(f"{j + 1:2d} / {cnt2}", CR=True, type="status")
            for i, λ_1 in enumerate(np.arange(*rng1)):
                E[j, i] = model.test(path_test, λ_1=λ_1, λ_2=λ_2)["entropy_H"]
        message("", CR=True)

        ma_y, ma_x = np.where(E == E.max())
        mi_y, mi_x = np.where(E == E.min())
        out.result("max", (E.max(), get_param(np.hstack([ma_x, ma_y]))))
        out.result("min", (E.min(), get_param(np.hstack([mi_x, mi_y]))))

    if save:
        fig = plt.figure()
        ax = fig.add_subplot(111)

        mappable = ax.pcolor(E, cmap="jet", edgecolors="k", alpha=0.8)
        fig.colorbar(mappable)

        ax.scatter(ma_x + 0.5, ma_y + 0.5, c="r", label="max")
        ax.scatter(mi_x + 0.5, mi_y + 0.5, c="b", label="min")

        ax.set_xticks(np.arange(cnt1) + 0.5, minor=False)
        ax.set_yticks(np.arange(cnt2) + 0.5, minor=False)
        ax.set_xticklabels(
            map(lambda x: f"{x:.2f}"[1:], np.arange(*rng1)),
            minor=False,
            rotation=45,
        )
        ax.set_yticklabels(
            map(lambda x: f"{x:.2f}"[1:], np.arange(*rng2)), minor=False,
        )
        ax.set_title(f"エントロピー {get_ext(model.WittenBell)}")
        ax.set_xlabel("$λ_1$")
        ax.set_ylabel("$λ_2$")
        ax.set_aspect("equal")
        ax.legend(loc="lower right")
        plt.savefig(save)

    return get_param(np.hstack([mi_x, mi_y]))


def test(args: argparse.Namespace) -> None:
    model = BigramLM(WittenBell=args.WittenBell).load(args.model)

    res = model.test(args.test)
    if args.name:
        message(
            f"[{args.name} | {get_ext(args.WittenBell)}"
            f" default(λ_1={0.95:.2f}, λ_2={0.95:.2f})]",
            file=sys.stdout,
        )
    for k, v in res.items():
        message(f"{k:15s} = {v:f}", file=sys.stdout)

    # λ_1, λ_2 = grid_search(
    #     model,
    #     args.test,
    #     rng=(0.01, 1, 0.01),
    #     save=f"result_{get_ext(args.WittenBell)}.png",
    # )
    λ_1, λ_2 = grid_search(
        model,
        args.test,
        rng=(0.1, 1, 0.1),
        save=f"fig1_{get_ext(args.WittenBell)}.png",
    )
    λ_1, λ_2 = grid_search(
        model,
        args.test,
        rng1=(λ_1 - 0.1, λ_1 + 0.1, 0.01),
        rng2=(λ_2 - 0.1, λ_2 + 0.1, 0.01),
        save=f"fig2_{get_ext(args.WittenBell)}.png",
    )

    res = model.test(args.test, λ_1=λ_1, λ_2=λ_2)
    if args.name:
        message(
            f"[{args.name} | {get_ext(args.WittenBell)}"
            f" optimized(λ_1={λ_1:.2f}, λ_2={λ_2:.2f})]",
            file=sys.stdout,
        )
    for k, v in res.items():
        message(f"{k:15s} = {v:f}", file=sys.stdout)

    """result
    [wiki - unigram | optimized(λ_1=0.89)]
    entropy_H       = 10.491366
    perplexity_PPL  = 1439.514219
    coverage        = 0.895226

    [wiki - bigram | WittenBellなし default(λ_1=0.95, λ_2=0.95)]
    entropy_H       = 11.284334
    perplexity_PPL  = 2494.151707
    coverage        = 0.455851
    [wiki - bigram | WittenBellなし optimized(λ_1=0.83, λ_2=0.37)]
    entropy_H       = 9.661475
    perplexity_PPL  = 809.830100
    coverage        = 0.455851

    [wiki - bigram | WittenBellあり default(λ_1=0.95, λ_2=0.95)]
    entropy_H       = 10.145161
    perplexity_PPL  = 1132.394330
    coverage        = 0.455851
    [wiki - bigram | WittenBellあり optimized(λ_1=0.82, λ_2=0.16)]
    entropy_H       = 9.654343
    perplexity_PPL  = 805.836431
    coverage        = 0.455851
    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Bigram Language Model")
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
    parser_test.add_argument(
        "--WittenBell", help="WittenBell smoothing flag", action="store_true"
    )
    parser_test.set_defaults(handler=test)

    args = parser.parse_args()
    if hasattr(args, "handler"):
        args.handler(args)
    else:
        parser.print_help()
