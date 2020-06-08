"""
Usage:
    N=3
    CORPUS_PATH=../../data/wiki-en-train.word
    TEST_PATH=../../data/wiki-en-test.word
    python NgramLM.py $CORPUS_PATH $TEST_PATH
Result:
    trans=str                           trans=lambda x: x.lower()
    -----------------------------------*-----------------------------------
    n=1 19it
    (10.492087432401616, (90,))         (10.23944943795005, (90,))
    n=2 361it
    (9.597385371042458, (80, 5))        (9.384889989064993, (85, 5))
    n=3 6859it
    (9.825519964848326, (80, 5, 5))     (9.609565402001232, (85, 5, 5))
"""
import argparse
import collections
import math
import os
import sys
from itertools import product
from typing import Callable, DefaultDict, List, Tuple, Type, TypeVar, Union

from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
import kiyuna.common as common  # noqa: E402 isort:skip

F = Callable[[str], str]


class NgramLM:
    def __init__(
        self,
        *,
        n: int,
        BOS: str = "<s>",
        EOS: str = "</s>",
        SEP: str = " ",
        trans: F = str,
    ) -> None:
        self.n: int = n
        self.BOS: str = BOS
        self.EOS: str = EOS
        self.SEP: str = SEP
        self.trans: F = trans
        self.freqs: DefaultDict[str, int] = collections.defaultdict(int)

    def get_tokens(self, line: str) -> Tuple[str, ...]:
        """Append BOS and EOS"""
        words = line.strip().split()
        words = [self.BOS] * (self.n - 1) + words
        words.append(self.EOS)
        return tuple(map(self.trans, words))

    def get_ngrams(
        self, tokens: Tuple[str, ...], n_: int
    ) -> List[Tuple[str, ...]]:
        start = self.n - n_
        return common.n_gram(tokens, n_)[start:]

    def get_context(self, ngram: Tuple[str, ...]) -> str:
        return "context::" + self.SEP.join(ngram[:-1])

    def train(self, path_corpus: str) -> Type["NgramLM"]:
        with open(path_corpus) as f:
            for line in f:
                tokens = self.get_tokens(line)
                for n_ in range(1, self.n + 1):
                    for ngram in self.get_ngrams(tokens, n_):
                        self.freqs[self.SEP.join(ngram)] += 1
                        self.freqs[self.get_context(ngram)] += 1
        for ngram in tuple(self.freqs.keys()):
            if "::" in ngram:
                continue
            tokens = ngram.split(self.SEP)
            self.freqs[self.get_context(tokens) + self.SEP + "*"] += 1
        return self

    def get_prob(self, ngram: str, context: str) -> float:
        assert context.startswith("context::")
        return self.freqs[ngram] / self.freqs[context]

    def show_probs(self) -> None:
        for ngram in sorted(self.freqs):
            if "::" in ngram:
                continue
            context = self.get_context(ngram.split(self.SEP))
            prob = self.get_prob(ngram, context)
            print(f"{ngram}\t{prob:f}")

    def get_lambda(
        self, ngram: Tuple[str], n: int, params: dict,
    ) -> Tuple[float, str]:
        r"""Witten-Bell smoothing
        :math:`λ_{context} = 1 - \frac{u(context)}{u(context) + c(context)}`
        """
        context = self.get_context(ngram)
        if n == 1:
            return params[f"lambda_{n}"], context
        c = self.freqs[context]
        u = self.freqs[context + self.SEP + "*"]
        if c == 0:
            return params[f"lambda_{n}"], context
        return 1 - u / (u + c), context

    def test(
        self, path_test: str, *, vocab_size: int = 1_000_000, params: dict,
    ) -> float:
        r"""Calculate entropy
        math::
            P(w_i ∣ w_i-n+1:i) = λ_{w_i-n+1:i} * P_ML(w_i ∣ w_i-n+1:i)
                               + (1 − λ_{w_i-n+1:i}) * P(w_i | w_i-n+2:i)
            P(w_i | w_i:i) = λ_{w_i} * P_ML(w_i) + (1 - λ_{w_i}) / vocab_size
        """
        nll: float = 0  # 負の対数尤度（negative log-likelihood）
        total: int = 0  # 単語数
        with open(path_test) as f:
            for line in f:
                tokens = self.get_tokens(line)
                for ngram in self.get_ngrams(tokens, self.n):
                    prob = 1 / vocab_size
                    for n_ in range(1, self.n + 1):
                        sub_ = ngram[-n_:]
                        lambda_, context_ = self.get_lambda(sub_, n_, params)
                        prob *= 1 - lambda_
                        if self.freqs[context_]:
                            ngram_ = self.SEP.join(sub_)
                            prob += lambda_ * self.get_prob(ngram_, context_)
                    nll += -math.log2(prob)
                    total += 1
        entropy = nll / total
        return entropy


def main(args: argparse.Namespace) -> None:
    for n in [args.n] if args.n else range(1, 4):
        lm = NgramLM(n=n, trans=lambda x: x.lower()).train(args.corpus)
        rng = range(5, 100, 5)
        opt: Tuple[float, dict] = (float("inf"), {})
        for lambdas in tqdm(product(rng, repeat=n)):
            d = {
                f"lambda_{num}": val / 100
                for num, val in enumerate(lambdas, start=1)
            }
            opt = min(opt, (lm.test(args.test, params=d), lambdas))
        res, lambdas = opt
        d = {f"λ_{num}": val / 100 for num, val in enumerate(lambdas, start=1)}
        print(f"{res:f}", d)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="N-gram Language Model")
    parser.add_argument("corpus", help="a corpus path")
    parser.add_argument("test", help="a test set path")
    parser.add_argument("-n", type=int, help="max length")

    args = parser.parse_args()
    main(args)
