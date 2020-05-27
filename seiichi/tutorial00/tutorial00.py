# tutorial00
# ファイルの中の単語の頻度を数えるプログラムを作成

import sys, os
from itertools import chain
from collections import Counter

def count(path: str):
    """
    Args: path, str
    Return: word_cnt, dict
    """
    if type(path) != str:
        raise TypeError
    cnt = 0
    word_cnt = {}
    with open(path, "r") as f:
        file = f.readlines()
    for line in file:
        for word in line.strip().split():
            if word not in word_cnt.keys():
                word_cnt[word] = 0
            word_cnt[word] += 1
    return word_cnt

# def count(path: str):
#     if type(path) != str:
#         raise TypeError
#     with open(path, "r") as f:
#         return Counter(chain.from_iterable(map(str.split, f)))

if __name__ == "__main__":
    # path = sys.argv[1]
    path = "../../test/00-input.txt"
    # path = "../../data/wiki-en-train.word"
    word_cnt = count(path)
    # 昇順に出力
    for k, v in sorted(word_cnt.items()):
        print("{}\t{}".format(k, v))
    print(word_cnt)
