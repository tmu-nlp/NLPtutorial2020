r"""tutorial03.py
単語分割プログラムを作成

[Small]
OUTPUT_PATH=./test.word
python tutorial03.py test ../../test/04-model.txt ../../test/04-input.txt $OUTPUT_PATH
diff -s $OUTPUT_PATH ../../test/04-answer.txt

[Large]
MODEL_PATH=./model_wiki.txt
python tutorial03.py train ../../data/wiki-ja-train.word ./$MODEL_PATH
L1=95  # PPL が小さい L1=89 にしても変化なし
OUTPUT_PATH=./my_answer_${L1}.word
python tutorial03.py test ./$MODEL_PATH ../../data/wiki-ja-test.txt $OUTPUT_PATH --lambda_1 0.$L1
perl ../../script/gradews.pl ../../data/wiki-ja-test.word $OUTPUT_PATH | pbcopy

[Challenge]
MODEL_PATH=./data/big-ws-model.txt
L1=95  # PPL が小さい L1=89 にすると 57 行目が悪化
OUTPUT_PATH=./my_answer_${L1}_big.word
python tutorial03.py test ../../$MODEL_PATH ../../data/wiki-ja-test.txt $OUTPUT_PATH --lambda_1 0.$L1
perl ../../script/gradews.pl ../../data/wiki-ja-test.word $OUTPUT_PATH | pbcopy
"""
import argparse
import math
import os
import sys
from typing import List, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.message import message  # noqa: E402 isort:skip
from kiyuna.tutorial01.tutorial01 import UnigramLM  # noqa: E402 isort:skip
from kiyuna.tutorial01.tutorial01 import train  # noqa: E402 isort:skip


class Viterbi(UnigramLM):
    def solve(
        self,
        path_input: str,
        path_output: str,
        *,
        λ_1: float = 0.95,
        vocab_size: int = 1_000_000,
    ) -> None:
        def forward_step(
            line: str, V: int = vocab_size
        ) -> List[Tuple[int, int]]:
            size = len(line)
            best_edge = [None] * (size + 1)
            best_score = [float("inf")] * (size + 1)
            best_score[0] = 0
            for word_end in range(1, size + 1):
                for word_begin in range(size):
                    word = line[word_begin:word_end]
                    if word in self.model or len(word) == 1:
                        prob = λ_1 * self.model.get(word, 0) + (1 - λ_1) / V
                        my_score = best_score[word_begin] + -math.log2(prob)
                        if my_score < best_score[word_end]:
                            best_score[word_end] = my_score
                            best_edge[word_end] = (word_begin, word_end)
            return best_edge

        def backward_step(
            line: str, best_edge: List[Tuple[int, int]]
        ) -> List[str]:
            words = []
            next_edge = best_edge[-1]
            while next_edge:
                words.append(line[next_edge[0] : next_edge[1]])
                next_edge = best_edge[next_edge[0]]
            words.reverse()
            return words

        res = []
        with open(path_input) as f_in:
            for line in map(lambda x: x.strip(), f_in):
                best_edge = forward_step(line)
                words = backward_step(line, best_edge)
                res.append(" ".join(words) + "\n")
        with open(path_output, "w") as f_out:
            f_out.writelines(res)
        message(f"saved {path_output}", type="success")


def main(args: argparse.Namespace) -> None:
    Viterbi().load(args.model).solve(
        args.input, args.output, λ_1=args.lambda_1
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Viterbi")
    subparsers = parser.add_subparsers(help="sub-comman help")

    parser_train = subparsers.add_parser("train", help="see `train -h`")
    parser_train.add_argument("corpus", help="the location of a corpus file")
    parser_train.add_argument(
        "model", help="a file location in which to save the LM"
    )
    parser_train.set_defaults(handler=train)

    parser_test = subparsers.add_parser("test", help="see `test -h`")
    parser_test.add_argument("model", help="the location of a model file")
    parser_test.add_argument("input", help="分割したいファイルのパス")
    parser_test.add_argument("output", help="分割結果の保存先")
    parser_test.add_argument(
        "--lambda_1", type=float, default=0.95, help="λ_1"
    )
    parser_test.set_defaults(handler=main)

    args = parser.parse_args()
    if hasattr(args, "handler"):
        args.handler(args)
    else:
        parser.print_help()

"""result
# λ_1 = 0.95
Sent Accuracy: 23.81% (20/84)
Word Prec: 68.93% (1861/2700)
Word Rec: 80.77% (1861/2304)
F-meas: 74.38%
Bound Accuracy: 83.25% (2683/3223)

# λ_1 = 0.89
Sent Accuracy: 23.81% (20/84)
Word Prec: 68.93% (1861/2700)
Word Rec: 80.77% (1861/2304)
F-meas: 74.38%
Bound Accuracy: 83.25% (2683/3223)

# λ_1 = 0.95, big
Sent Accuracy: 17.86% (15/84)
Word Prec: 85.66% (1972/2302)
Word Rec: 85.59% (1972/2304)
F-meas: 85.63%
Bound Accuracy: 91.13% (2937/3223)

# λ_1 = 0.89, big
Sent Accuracy: 17.86% (15/84)
Word Prec: 85.61% (1970/2301)
Word Rec: 85.50% (1970/2304)
F-meas: 85.56%
Bound Accuracy: 91.10% (2936/3223)
"""
