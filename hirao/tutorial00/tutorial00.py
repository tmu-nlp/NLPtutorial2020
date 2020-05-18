"""
tutorial00
ファイル中の単語の頻度を数えるプログラムを作成

input_file
small_test: ../../test/00-input.txt
large_test: ../../data/wiki-en-train.word

stdinput command
$ python tutorial00.py < ../../data/wiki-en-train.word
"""
import sys
from typing import List, Tuple
from collections import defaultdict

def cnt_word_from_std_input() -> List[Tuple[int, int]]:
    # 辞書の初期値を0に設定
    cnt = defaultdict(lambda: 0)

    for line in sys.stdin.readlines():
        # 文を単語に分割
        for word in line.rstrip().split():
            cnt[word] += 1

    ret = [(word, count) for word, count in cnt.items()]
    return ret

def cnt_word_from_path(filepath: str) -> List[Tuple[int, int]]:
    cnt = defaultdict(lambda: 0)

    with open(input_path) as f:
        for line in f:
            for word in line.split():
                cnt[word] += 1

    ret = [(word, count) for word, count in cnt.items()]
    return ret

if __name__ == "__main__":
    result = cnt_word_from_std_input()
    # result = cnt_word_from_path("../../data/wiki-en-train.word")

    for w, cnt in result:
        print(w, cnt)

    # ↑よりこっちの方が高速に動作する
    # print("\n".join([f"{w} {cnt}" for w, cnt in result]))
