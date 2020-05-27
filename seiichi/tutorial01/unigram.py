# unigram model trainer
import os, sys
import math
import pickle
import numpy as np

class UnigramModel(object):
    def __init__(self):
        self.data = []
        self.word_cnt = {}
        return 
    def load_data(self, path: str):
        with open(path, "r") as f:
            # data = list(map(str.split, f))
            tmp = f.readlines()
        data = [line.strip().split() + ["</s>"] for line in tmp]
        return data
    def train(self, train_path: str):
        self.data = self.load_data(train_path)
        for line in self.data:
            for word in line:
                if word not in self.word_cnt.keys():
                    self.word_cnt[word] = 0
                self.word_cnt[word] += 1
        return
    def save(self, model_path: str):
        with open(model_path, "wb") as f:
            pickle.dump(self.word_cnt, f)
        return
    def load(self, model_path: str):
        with open(model_path, "rb") as f:
            self.word_cnt = pickle.load(f)
        return
    def calc_word_prob(self, word: str, v: int, ld: float):
        prob = (1-ld) / v
        if word not in self.word_cnt.keys():
            return prob
        return prob + ld * float(self.word_cnt[word] / sum(self.word_cnt.values()))
    def report(self, test_path: str, ld=0.95, v=1e6):
        test = self.load_data(test_path)
        test_cnt = 0
        unk_cnt = 0
        for line in test:
            for word in line:
                if word not in self.word_cnt.keys():
                    unk_cnt += 1
                test_cnt += 1
        h = 0
        for line in test:
            for word in line:
                h += (-1) * math.log2(self.calc_word_prob(word, v, ld))
        return float(h/test_cnt), float((test_cnt-unk_cnt)/test_cnt)
    def word_probabilities(self):
        for k, _ in sorted(self.word_cnt.items()):
            print("{}\t{:.6f}".format(k, self.calc_word_prob(k, sum(self.word_cnt.values()), 1.0)))
        return