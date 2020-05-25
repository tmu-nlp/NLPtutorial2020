r"""tutorial00.py
ファイルの中の単語の頻度を数えるプログラムを作成

[Tips]
>>> # 大文字と小文字の区別をなくしたいとき
>>> (lambda w: w.lower())("aBc")
'abc'

[NOTE]
- ただし読み込ませるデータが多い時は逐次読み込ませる（OOM）

[Small]
INPUT_PATH=./test/00-input.txt
ANSWER_PATH=./test/00-answer.txt

[Large]
INPUT_PATH=./data/wiki-en-train.word

[Docker]
docker-compose up -d
docker-compose exec nlp_tutorial2020 pip install colorama
docker-compose exec nlp_tutorial2020 python ./kiyuna/tutorial00/tutorial00.py $INPUT_PATH
docker-compose stop; docker-compose rm -f

[Usage]
python tutorial00.py MEMO
python tutorial00.py ../../$INPUT_PATH
diff -s <(python tutorial00.py ../../$INPUT_PATH) ../../$ANSWER_PATH
"""
import collections
import dis
import doctest
import os
import sys
import typing
from typing import Callable, Counter, List, Set, Tuple, TypeVar

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.message import Renderer, message  # noqa: E402 isort:skip
from kiyuna.utils.pickle import dump, load  # noqa: E402 isort:skip

T = TypeVar("T")
F = Callable[[str], T]


def list_word_freq(path: str, trans: F = str) -> List[Tuple[T, int]]:
    cnter = collections.defaultdict(int)
    with open(path) as f:
        for line in f:
            for word in map(trans, line.split()):
                cnter[word] += 1
    return sorted(cnter.items())


def build_word_frequency_cnter(path: str, trans: F = str) -> Counter[T]:
    with open(path) as f:
        return collections.Counter(map(trans, f.read().split()))


def get_vocab(path: str, trans: F = str) -> Set[T]:
    with open(path) as f:
        return {trans(w) for w in f.read().split()}


if __name__ == "__main__":
    path = sys.argv[1]

    if path == "MEMO":
        with Renderer("個人メモ") as out:
            out.result(
                "type hints",
                (
                    typing.get_type_hints(list_word_freq),
                    build_word_frequency_cnter.__annotations__,
                ),
            )
            out.header("with 内で return しても大丈夫なはず")
            dis.dis(build_word_frequency_cnter, file=sys.stderr)
            out.header("doctest")
            doctest.testmod(verbose=True)
            out.header("check serialize")
            cnter = list_word_freq("../../test/00-input.txt")
            dump(cnter, "cnter")
            cnter = load("cnter")
        exit(0)

    with Renderer("単語の異なり数") as out:
        out.result("map", len(list_word_freq(path)))
        out.result("set", len(get_vocab(path)))

    num = 10
    with Renderer(f"数単語の頻度（上位 {num} 単語のみ）") as out:
        out.result(
            "大文字と小文字の区別をする",
            build_word_frequency_cnter(path, str).most_common(num),
        )
        trans = lambda w: w.lower()  # noqa: E731
        out.result(
            "大文字と小文字の区別をしない",
            build_word_frequency_cnter(path, trans).most_common(num),
        )

    if "test" in path:
        for k, v in list_word_freq(path):
            print(k, v, sep="\t")

    message("DONE.", type="status")
