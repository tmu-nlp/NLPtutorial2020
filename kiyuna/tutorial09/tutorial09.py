r"""tutorial09.py
learn-lda

[Usage]
python tutorial09.py main > out
"""
import math
import os
import random
import sys
from collections import defaultdict
from pprint import pprint

import nltk
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.message import message  # noqa: E402 isort:skip

random.seed(42)


def sampleone(probs):
    """ #09 p14 """
    z = sum(probs)
    remaining = random.random() * z
    for i in range(len(probs)):
        remaining -= probs[i]
        if remaining <= 0:
            return i
    raise Exception("remaining =", remaining)


def add_counts(xcounts, ycounts, word, topic, docid, amount):
    """ #09 p23 """
    xcounts[topic] += amount
    xcounts[f"{word}|{topic}"] += amount
    ycounts[docid] += amount
    ycounts[f"{topic}|{docid}"] += amount


def initialize(train_file, num_topics):
    """ #09 p22 """
    xcorpus, ycorpus = [], []
    xcounts, ycounts = defaultdict(int), defaultdict(int)
    wordtype = set()
    for line in open(train_file):
        docid = len(xcorpus)
        words = line.split()
        topics = []
        for word in words:
            topic = random.randint(0, num_topics)
            topics.append(topic)
            add_counts(xcounts, ycounts, word, topic, docid, 1)
            wordtype.add(word)
        xcorpus.append(words)
        ycorpus.append(topics)
    return xcorpus, ycorpus, xcounts, ycounts, len(wordtype)


def sample(test_path, epochs=1, α=0.01, β=0.01, num_topics=2):
    """ #09 p24 """
    xcorpus, ycorpus, xcounts, ycounts, wordtype = initialize(test_path, num_topics)
    for epoch in range(1, epochs + 1):
        message("epoch =", epoch, type="status")
        ll = 0
        for i in tqdm(range(len(xcorpus)), leave=False):
            for j in range(len(xcorpus[i])):
                x = xcorpus[i][j]
                y = ycorpus[i][j]
                add_counts(xcounts, ycounts, x, y, i, -1)
                probs = []
                for k in range(num_topics):
                    p_xk = (xcounts[f"{x}|{k}"] + α) / (xcounts[k] + α * wordtype)
                    p_ky = (ycounts[f"{k}|{i}"] + β) / (ycounts[i] + β * num_topics)
                    probs.append(p_xk * p_ky)
                new_y = sampleone(probs)
                ll += math.log(probs[new_y])
                add_counts(xcounts, ycounts, x, new_y, i, 1)
                ycorpus[i][j] = new_y
        message("ll =", ll, type="success")
    return xcorpus, ycorpus


def learn_lda(test_path, epochs=1, α=0.01, β=0.01, num_topics=2, stop_words=[]):
    xcorpus, ycorpus = sample(test_path, epochs, α, β, num_topics)
    cnter = defaultdict(lambda: [0] * num_topics)
    for i in range(len(xcorpus)):
        for j in range(len(xcorpus[i])):
            # print(f"{xcorpus[i][j]}_{ycorpus[i][j]}", end=" ")
            cnter[xcorpus[i][j]][ycorpus[i][j]] += 1
        # print()
    groups = [[] for _ in range(num_topics)]
    tmp = []
    for k, v in cnter.items():
        idx = v.index(max(v))
        groups[idx].append((v[idx], k))
        tmp.append((v[idx], k))
    print_cnt = min(300, len(tmp))
    th = sorted(tmp, reverse=True)[print_cnt - 1][0]
    for i, group in enumerate(groups):
        print("=" * 5, i, "=" * 5)
        group.sort(reverse=True)
        res = [word for freq, word in group if word not in stop_words and freq >= th]
        pprint(res, width=80, compact=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["test"]:
        message("test", type="status")
        learn_lda(test_path="../../test/07-train.txt", epochs=50)
    else:
        message("main", type="status")
        stop_words = nltk.corpus.stopwords.words("english")
        symbols = [
            "'",
            '"',
            ":",
            ";",
            ".",
            ",",
            "-",
            "!",
            "?",
            ")",
            "(",
            "/",
            "&apos;",
            "&apos;s",
            "&quot;",
        ]
        stop_words += symbols
        learn_lda(
            test_path="../../data/wiki-en-documents.word",
            epochs=50,
            num_topics=7,
            stop_words=stop_words,
        )
