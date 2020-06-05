# bigram model trainer
import os, sys
import math
# import pickle
import dill
import numpy as np
from collections import defaultdict

class BigramModel(object):
    def __init__(self, BOS="<s>", EOS="</s>"):
        self.context_cnt = defaultdict(lambda: defaultdict(int))
        self.BOS = BOS
        self.EOS = EOS
        self.lambda_1 = 0.95
        self.lambda_2 = 0.05
        self.V = 1e6
        self.EMP = "_"
        return

    def load_data(self, path: str):
        with open(path, "r") as f:
            tmp = f.readlines()
        data = [[self.BOS] + line.strip().split() + [self.EOS] for line in tmp]
        return data

    def train(self, train_path: str):
        data = self.load_data(train_path)
        for line in data:
            for i in range(1, len(line)):
                self.context_cnt[line[i-1]][line[i]] += 1
                self.context_cnt[self.EMP][line[i]] += 1
        return self
                
    def save_model(self, model_path: str):
        with open(model_path, "wb") as f:
            dill.dump(self.context_cnt, f)
        return self

    def load_model(self, model_path: str):
        with open(model_path, "rb") as f:
            self.context_cnt = dill.load(f)
        return self

    def calc_bigram_word_prob(self, context: str, word: str, smoothing=True):
        prob1 = self.calc_unigram_word_prob(word)
        if context not in self.context_cnt.keys():
            prob2 = 0
        else:
            prob2 = self.context_cnt[context][word] / sum(self.context_cnt[context].values())
        if smoothing and (context in self.context_cnt.keys()):
            lambda_2 = self.calc_lambda_with_witten_bell_smoothing(context)
            # print(lambda_2)
        else:
            lambda_2 = self.lambda_2
        prob = lambda_2 * prob2 + (1 - lambda_2) * prob1
        return prob

    def calc_unigram_word_prob(self, word: str):
        prob = (1-self.lambda_1) / self.V
        if word not in self.context_cnt[self.EMP].keys():
            return prob
        return prob + self.lambda_1 * float(self.context_cnt[self.EMP][word] / sum(self.context_cnt[self.EMP].values()))

    def calc_lambda_with_witten_bell_smoothing(self, context):
        u = len(self.context_cnt[context])
        c = sum(self.context_cnt[context].values())
        return 1 - (u / (u + c))

    def calc_entropy(self, tar_path: str):
        tar_data = self.load_data(tar_path)
        H, W = 0, 0
        for line in tar_data:
            for i in range(1, len(line)):
                p = self.calc_bigram_word_prob(line[i-1],line[i])
                H += (-1) * math.log2(p)
                W += 1
        return H / W

    def out_word_prob(self, smoothing=True):
        for context, word_cnt in sorted(self.context_cnt.items()):
            for word, cnt in sorted(word_cnt.items()):
                print("{} {} {:.6f}".format(context, word, self.calc_bigram_word_prob(context, word, smoothing=smoothing)))
        return 