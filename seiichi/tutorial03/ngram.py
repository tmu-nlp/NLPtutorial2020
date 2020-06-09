# bigram model trainer
import os, sys
import math
# import pickle
import dill
import numpy as np
from collections import defaultdict

class NgramModel(object):
    def __init__(self, n=2, BOS="<s>", EOS="</s>", EMP="<none>", SEP=" "):
        self.n = n
        # self.context_cnt = defaultdict(int)
        self.ngram_context_cnt = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.BOS = BOS
        self.EOS = EOS
        self.lambda_1 = 0.95
        self.lambda_2 = 0.05
        self.V = 1e6
        self.EMP = EMP
        self.SEP = SEP
        return

    def load_data(self, path: str):
        with open(path, "r") as f:
            tmp = f.readlines()
        data = [[self.BOS] * (self.n-1) + line.strip().split() + [self.EOS] for line in tmp]
        return data

    def _get_ngrams(self, seq, n):
        return list(zip(*[seq[i:] for i in range(n)]))

    def train(self, train_path: str):
        data = self.load_data(train_path)
        for line in data:
            for i in range(1, self.n+1):
                for ngram in self._get_ngrams(line, i): # obtain i-gram
                    word, context = "".join(ngram[-1:]), tuple(ngram[:-1])
                    self.ngram_context_cnt[i][context][word] += 1
        return self
                
    def save_model(self, model_path: str):
        with open(model_path, "wb") as f:
            dill.dump(self.ngram_context_cnt, f)
        return self

    def load_model(self, model_path: str):
        with open(model_path, "rb") as f:
            self.ngram_context_cnt = dill.load(f)
        return self

    def calc_ngram_word_prob(self, ngram: list):
        prob = 1 / self.V
        word, context = "".join(ngram[-1:]), tuple(ngram[:-1])
        for i in range(1, self.n+1):
            subst = ngram[-i:]
            subst_word = "".join(subst[-1:])
            subst_context = tuple(subst[:-1])
            _lambda = self.get_lambda(subst_context, i)
            prob *= (1 - _lambda)
            if self.ngram_context_cnt[i][subst_context]:
                tmp = self.ngram_context_cnt[i][subst_context][subst_word] / sum(self.ngram_context_cnt[i][subst_context].values())
                prob += _lambda * tmp
        return prob

    def get_lambda(self, context: tuple, n: int):
        if n == 1:
            return self.lambda_1
        else:
            return self.calc_lambda_with_witten_bell_smoothing(context, n)
    
    def calc_lambda_with_witten_bell_smoothing(self, context: tuple, i: int):
        u = len(self.ngram_context_cnt[i][context])
        c = sum(self.ngram_context_cnt[i][context].values())
        if c == 0:
            return self.lambda_2
        return 1 - (u / (u + c))

    def calc_entropy(self, tar_path: str):
        tar_data = self.load_data(tar_path)
        H, W = 0, 0
        for line in tar_data:
            # for i in range(1, len(line)):
            for ngram in self._get_ngrams(line, self.n):
                # p = self.calc_bigram_word_prob(line[i-1],line[i])
                p = self.calc_ngram_word_prob(ngram)
                H += (-1) * math.log2(p)
                W += 1
        return H / W

    def out_word_prob(self, tar_path: str):
        for i in range(1, self.n+1):
            for context, word_cnt in self.ngram_context_cnt[i].items():
                for word, cnt in word_cnt.items():
                    print("{} {} {:.6f}".format(context, word, self.calc_ngram_word_prob(list(context)+[word])))
        return 


if __name__ == "__main__":
    if not os.path.exists("./model"):
        os.mkdir("model")
    # train_path = sys.argv[1]
    # train_path = "../../test/02-train-input.txt"
    # test_path = "../../test/02-test-input.txt"
    train_path = "../../data/wiki-en-train.word"
    test_path = "../../data/wiki-en-test.word"
    model = NgramModel(n=1)
    model.train(train_path)
    model.save_model("./model/wikien.model")
    model.out_word_prob(test_path)
    print("entropy: ", model.calc_entropy(test_path))