import os, sys
import math
# import pickle
import dill
from collections import defaultdict
from ngram import NgramModel

class Tokenizer(NgramModel):
    def __init__(self, n=1, V=1e6, lambda_1=0.95):
        self.n = n
        self.V = V
        self.lambda_1 = lambda_1
        super().__init__(n=self.n)

    def forward(self, line):
        best_edge, best_score = [None] * (len(line)+1), [float("inf")] * (len(line)+1)
        best_score[0] = 0
        for word_end in range(1, len(line)+1):
            for word_begin in range(0, word_end):
                sub_word = line[word_begin:word_end]
                if self.ngram_context_cnt[self.n][tuple()][sub_word] or len(sub_word) == 1:
                    prob = self.calc_ngram_word_prob([sub_word])
                    my_score = best_score[word_begin] + (-1) * math.log2(prob)
                    if my_score < best_score[word_end]:
                        best_score[word_end] = my_score
                        best_edge[word_end] = (word_begin, word_end)
        return best_edge

    def backward(self, line, best_edge):
        words = []
        next_edge = best_edge[-1]
        while next_edge != None:
            words.append(line[next_edge[0]:next_edge[1]])
            next_edge = best_edge[next_edge[0]]
        return " ".join(reversed(words))
    
    def load_raw_data(self, path):
        with open(path, "r") as f:
            data = f.readlines()
        return list(map(lambda x: x.strip(), data))

    def tokenize(self, tar_path):
        results = []
        data = self.load_raw_data(tar_path)
        for line in data:
            best_edge = self.forward(line)
            ret = self.backward(line, best_edge)
            results.append(ret)
        return results

if __name__ == "__main__":
    # train_path = "../../test/04-input.txt"
    train_path = "../../data/wiki-ja-train.word"
    test_path = "./data/wiki-ja-test.txt"
    save_path = "./log/my_answer.word"
    model_path = "./model/wikija.model"
    # sample_model_path = "../../test/04-model.txt"
    tk = Tokenizer(n=1)
    tk.train(train_path).save_model(model_path)
    results = tk.tokenize(test_path)
    with open(save_path, "w") as f:
        for res in results:
            f.write(res+"\n")

"""result
Sent Accuracy: 23.81% (20/84)
Word Prec: 68.93% (1861/2700)
Word Rec: 80.77% (1861/2304)
F-meas: 74.38%
Bound Accuracy: 83.25% (2683/3223)
"""