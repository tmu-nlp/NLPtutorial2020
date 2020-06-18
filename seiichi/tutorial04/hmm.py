import os, sys
import dill
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# sys.path.append("../../")
# from tutorial03.ngram import NgramModel

class HMM(object):
    def __init__(self, BOS="<s>", EOS="</s>", lambda_1=0.95, lambda_2=0.05):
        # self.n_context_cnt = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.context = defaultdict(int)
        self.transition = defaultdict(lambda: defaultdict(int))
        self.emit = defaultdict(lambda: defaultdict(int))
        self.BOS = BOS
        self.EOS = EOS
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.V = 1e6
        self.tar_data = None

    def load_data(self, path: str):
        with open(path, "r") as f:
            tmp = f.readlines()
        data = [line.strip().split() for line in tmp]
        return data

    def train(self, train_path: str):
        data = self.load_data(train_path)
        for line in data:
            prev = self.BOS
            self.context[prev] += 1
            for word_tag in line:
                word, tag = word_tag.split("_")
                self.transition[prev][tag] += 1
                self.context[tag] += 1
                self.emit[tag][word] += 1
                prev = tag
            self.transition[prev][self.EOS] += 1   
        return self
    
    def save_model(self, model_path):
        with open(model_path, "wb") as f:
            dill.dump([self.context, self.transition, self.emit, self.BOS, self.EOS], f)
        return self
    
    def load_model(self, model_path):
        with open(model_path, "rb") as f:
            model = dill.load(f)
            self.context = model[0]
            self.transition = model[1]
            self.emit = model[2]
            self.BOS = model[3]
            self.EOS = model[4]
        return self

    def calc_transition_prob(self, prev, tag):
        # p_T(y_i | y_{i-1}) = p_ML(y_i | y_{i-1})
        prob = self.transition[prev][tag] / self.context[prev]
        return prob
    
    def calc_emittion_prob(self, tag, word):
        # p_E(x_i | y_i) = \lambda * p_ML(x_i | y_i) + (1 - \lambda) (1/N)
        prob = (1 - self.lambda_1) * 1 / self.V
        if not self.context[tag]:
            return prob
        prob += self.lambda_1 * self.emit[tag][word] / self.context[tag]
        return prob

    def out_probs(self):
        for context, word_prob in self.transition.items():
            for word, prob in word_prob.items():
                print(context, word, self.calc_transition_prob(context, word))
        for context, word_prob in self.emit.items():
            for word, prob in word_prob.items():
                print(context, word, self.calc_emittion_prob(context, word))
        return 
    def forward(self, words):
        best_score = defaultdict(lambda: defaultdict(int))
        best_edge = defaultdict(lambda: defaultdict(tuple))
        best_score[0][self.BOS] = 0
        best_edge[0][self.BOS] = None
        score = 0
        for i in range(len(words)):
            for prev in self.context.keys():
                for next in self.context.keys():
                    if best_score[i].get(prev, -1) != -1 and self.transition[prev].get(next, -1) != -1:
                        score = best_score[i][prev] + (-1) * np.log2(self.calc_transition_prob(prev, next)) + (-1) * np.log2(self.calc_emittion_prob(next, words[i]))
                        if best_score[i+1].get(next, -1) == -1 or (best_score[i+1][next] > score):
                            best_score[i+1][next] = score
                            best_edge[i+1][next] = tuple([i, prev])
        # best_score[len(words)+1][self.EOS] = min(best_score[len(words)].values())
        best_score[len(words)+1][self.EOS] = 1e10
        best_edge[len(words)+1][self.EOS] = None
        for tag in self.context.keys():
            if best_score[len(words)].get(tag, -1) != -1 and best_score[len(words)][tag] < best_score[len(words)+1][self.EOS]:
                best_score[len(words)+1][self.EOS] = best_score[len(words)][tag]
                best_edge[len(words)+1][self.EOS] = tuple([len(words),tag])
        return best_edge
    
    def backward(self, line, best_edge):
        tags = []
        next_edge = best_edge[len(line)+1][self.EOS]
        while next_edge != tuple([0, self.BOS]):
            position, tag = next_edge[0], next_edge[1]
            tags.append(tag)
            next_edge = best_edge[position][tag]
        return " ".join(reversed(tags))
    
    def pos_tagging(self, tar_path):
        data = self.load_data(tar_path)
        pos_tags = []
        for line in data:
            edge = self.forward(line)
            tags = self.backward(line, edge)
            pos_tags.append(tags)
        return pos_tags
    

if __name__ == "__main__":
    # train_path = "../../test/05-train-input.txt"
    train_path = "../../data/wiki-en-train.norm_pos"
    # tar_path = "../../test/05-test-input.txt"
    tar_path = "../../data/wiki-en-test.norm"
    save_path = "./model/test.model"
    hmm = HMM()
    hmm.train(train_path).save_model(save_path)
    hmm.load_model(save_path)
    # hmm.out_probs()
    res = hmm.pos_tagging(tar_path)
    with open("./out/test.txt", "w") as f:
        for r in res: f.write(r+"\n")


# 改良版: https://github.com/seiichiinoue/hmm
# 解説: https://seiichiinoue.github.io/post/hmm/

"""results
% perl ../../script/gradepos.pl ../../data/wiki-en-test.pos ./out/test.txt 
Accuracy: 90.82% (4144/4563)

Most common mistakes:
NNS --> NN      45
NN --> JJ       27
NNP --> NN      22
JJ --> DT       22
JJ --> NN       12
VBN --> NN      12
NN --> IN       11
NN --> DT       10
NNP --> JJ      8
JJ --> RB       7
"""