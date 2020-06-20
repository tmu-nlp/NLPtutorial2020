r"""tutorial04.py
train-hmm と test-hmm を実装

[MEMO]
key を (T_遷移, prv, tag) にすると 7.5 s
key を f"str {prv} {tag}" にすると 9 s

[Ref]
- PEP 563: アノテーションの遅延評価
    - https://docs.python.org/ja/3/whatsnew/3.7.html#pep-563-postponed-evaluation-of-annotations
    - https://stackoverflow.com/questions/33533148/how-do-i-specify-that-the-return-type-of-a-method-is-the-same-as-the-class-itsel

[Small]
# train
CORPUS_PATH=./test/05-train-input.txt
MODEL_PATH=./model_test.txt
python tutorial04.py train ../../$CORPUS_PATH $MODEL_PATH
diff -s $MODEL_PATH ../../test/05-train-answer.txt
# test
MODEL_PATH=./model_test.txt
INPUT_PATH=./test/05-test-input.txt
OUTPUT_PATH=./test.pos
python tutorial04.py test $MODEL_PATH ../../$INPUT_PATH $OUTPUT_PATH
diff -s $OUTPUT_PATH ../../test/05-test-answer.txt

[Large]
# train
CORPUS_PATH=./data/wiki-en-train.norm_pos
MODEL_PATH=./model_wiki.txt
python tutorial04.py train ../../$CORPUS_PATH $MODEL_PATH
# test
MODEL_PATH=./model_wiki.txt
INPUT_PATH=./data/wiki-en-test.norm
OUTPUT_PATH=./my_answer.pos
python tutorial04.py test $MODEL_PATH ../../$INPUT_PATH $OUTPUT_PATH
perl ../../script/gradepos.pl ../../data/wiki-en-test.pos $OUTPUT_PATH | pbcopy
"""
import argparse
import os
import sys
from collections import defaultdict
from itertools import product
from math import log2
from typing import Callable, DefaultDict, List, Sequence, Set, Tuple, TypeVar

import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
import kiyuna.utils.pickle as pickle  # noqa: E402 isort:skip
from kiyuna.utils.message import message  # noqa: E402 isort:skip
from kiyuna.utils.time import Timer  # noqa: E402 isort:skip

F = Callable[[str], str]
Prob = TypeVar("Prob", bound=float)
Probs = DefaultDict[tuple, Prob]
Tags = List[str]
History = DefaultDict[Tuple[int, str], Tuple[int, str]]

T_遷移: str = "T"
C_文脈: str = "C"
E_生成: str = "E"
INF: float = float("inf")


class POSTagger:
    probs: Probs
    possible_tags: Set[str]
    params = {
        "lambda_1": 0.95,
        "vocab_size": 1_000_000,
    }

    def __init__(
        self, *, trans: F = str, BOS: str = "<s>", EOS: str = "</s>",
    ) -> None:
        self.trans: F = trans
        self.BOS: str = BOS
        self.EOS: str = EOS

    def train(self, path_corpus: str) -> "POSTagger":
        self.probs, self.possible_tags = self.__build_model(path_corpus)
        message(f"train model from {path_corpus}", type="success")
        return self

    def load(self, path_model: str) -> "POSTagger":
        self.probs, self.possible_tags = self.__load_model(path_model)
        message(f"load  model from {path_model}", type="success")
        return self

    def _get_context(self, key: Sequence) -> str:
        return key[1]

    def __build_model(self, path_corpus: str) -> Tuple[Probs, Set[str]]:
        cnter: DefaultDict[tuple, int] = defaultdict(int)
        with open(path_corpus) as f:
            for line in f:
                prv_tag = self.BOS
                cnter[C_文脈, prv_tag] += 1
                for wordtag in line.split():
                    word, cur_tag = wordtag.split("_")
                    word = self.trans(word)
                    cnter[T_遷移, prv_tag, cur_tag] += 1
                    cnter[C_文脈, cur_tag] += 1
                    cnter[E_生成, cur_tag, word] += 1
                    prv_tag = cur_tag
                cnter[T_遷移, prv_tag, self.EOS] += 1
        probs, possible_tags = defaultdict(float), set()
        for key in filter(lambda x: x[0] is not C_文脈, cnter):
            tag = self._get_context(key)
            probs[key] = cnter[key] / cnter[C_文脈, tag]
            possible_tags.add(tag)
        return probs, possible_tags

    def __load_model(self, path_model: str) -> Tuple[Probs, Set[str]]:
        probs, possible_tags = defaultdict(float), set()
        with open(path_model) as f:
            for line in f:
                *key, prob = line.split(" ")
                probs[tuple(key)] = float(prob)
                possible_tags.add(self._get_context(key))
        return probs, possible_tags

    def dump(self, path_model: str) -> None:
        with pickle.SaveHelper(path_model), open(path_model, "w") as f:
            for key in sorted(self.probs, key=lambda x: (x[0] is E_生成, x)):
                print(*key, f"{self.probs[key]:f}", file=f)

    def __forward(self, words: List[str]) -> History:
        best_score: DefaultDict[tuple, float] = defaultdict(lambda: INF)
        best_edge: History = defaultdict(tuple)  # type: ignore
        best_score[0, self.BOS] = 0
        λ_1 = self.params["lambda_1"]
        V = self.params["vocab_size"]
        for i, word in enumerate(words, start=1):
            for prv, cur in product(self.possible_tags, repeat=2):
                if (i - 1, prv) not in best_score:
                    continue
                if (T_遷移, prv, cur) not in self.probs:
                    continue
                score = best_score[i - 1, prv]
                # P_T(y_i | y_{i-1}) = P_ML(y_i | y_{i-1}
                P_t = self.probs[T_遷移, prv, cur]
                score += -log2(P_t)
                # P_E(x_i | y_i) = λ * P_ML(x_i | y_i) + (1 - λ) / V
                P_e = λ_1 * self.probs[E_生成, cur, word] + (1 - λ_1) / V
                score += -log2(P_e)
                if best_score[i, cur] > score:
                    best_score[i, cur] = score
                    best_edge[i, cur] = i - 1, prv
        if "EOS に対する処理":
            i += 1
            cur = self.EOS
            for prv in self.possible_tags:
                if (i - 1, prv) not in best_score:
                    continue
                if (T_遷移, prv, cur) not in self.probs:
                    continue
                score = best_score[i - 1, prv]
                P_t = self.probs[T_遷移, prv, cur]
                score += -log2(P_t)
                # P_e = λ_1 * self.probs[E_生成, cur, word] + (1 - λ_1) / V
                # score += -log2(P_e)
                if best_score[i, cur] > score:
                    best_score[i, cur] = score
                    best_edge[i, cur] = i - 1, prv
        return best_edge

    def __backward(self, words: List[str], best_edge: History) -> Tags:
        tags = []
        next_edge = best_edge[len(words) + 1, self.EOS]
        while next_edge != (0, self.BOS):
            position, tag = next_edge
            tags.append(tag)
            next_edge = best_edge[next_edge]
        tags.reverse()
        return tags

    def tag(self, path_input: str, path_output: str, **kwargs) -> None:
        self.params.update(**kwargs)
        res = []
        with open(path_input) as f_in:
            for line in tqdm.tqdm(f_in):
                words = [self.trans(word) for word in line.split()]
                best_edge = self.__forward(words)
                tags = self.__backward(words, best_edge)
                res.append(" ".join(tags) + "\n")
        with open(path_output, "w") as f_out:
            f_out.writelines(res)
        message(f"saved : {path_output}", type="success")


def train(args: argparse.Namespace) -> None:
    POSTagger().train(args.corpus).dump(args.probs)


def test(args: argparse.Namespace) -> None:
    with Timer():
        POSTagger().load(args.probs).tag(
            args.input,
            args.output,
            lambda_1=args.lambda_1,
            vocab_size=args.vocab_size,
        )

    """
    [+] trans=str
    Accuracy: 90.82% (4144/4563)

    Most common mistakes:
    NNS --> NN	45
    NN --> JJ	27
    NNP --> NN	22
    JJ --> DT	22
    JJ --> NN	12
    VBN --> NN	12
    NN --> IN	11
    NN --> DT	10
    NNP --> JJ	8
    VBN --> JJ	7

    [+] trans=lambda x: x.lower()
    Accuracy: 90.86% (4146/4563)

    [+] lambda_1 の影響は小さい
    --lambda_1=0.3 -> 4143/4563
    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="POS tagger using hidden Markov probs"
    )
    subparsers = parser.add_subparsers(help="sub-comman help")

    parser_train = subparsers.add_parser("train", help="see `train -h`")
    parser_train.add_argument("corpus", help="the location of a corpus file")
    parser_train.add_argument(
        "probs", help="a file location in which to save the LM"
    )
    parser_train.set_defaults(handler=train)

    parser_test = subparsers.add_parser("test", help="see `test -h`")
    parser_test.add_argument("probs", help="the location of a probs file")
    parser_test.add_argument("input", help="品詞推定したいファイルのパス")
    parser_test.add_argument("output", help="品詞推定結果の保存先")
    parser_test.add_argument(
        "--lambda_1", type=float, default=0.95, help="λ_1"
    )
    parser_test.add_argument(
        "--vocab_size", type=int, default=1_000_000, help="V"
    )
    parser_test.set_defaults(handler=test)

    args = parser.parse_args()
    if hasattr(args, "handler"):
        args.handler(args)
    else:
        parser.print_help()
