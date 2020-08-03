import nltk
import os, sys, re
import numpy as np
from tqdm import tqdm
from collections import defaultdict

INF = float('inf')

class CKY:
    def __init__(self, grammar_path, data_path):
        self.nonterm = []
        self.preterm = defaultdict(list)
        self.words = []
        self._load_grammar(grammar_path)
        self._load_data(data_path)

    def _load_grammar(self, path):
        with open(path, "r") as f:
            data = f.readlines()
        for i in range(len(data)):
            data[i] = data[i].strip()
        for rule in data:
            lhs, rhs, prob = rule.split("\t")
            prob = float(prob)
            rhs_symbols = rhs.split()
            if len(rhs_symbols) == 1:
                self.preterm[rhs].append((lhs, np.log(prob)))
            else:
                self.nonterm.append((lhs, rhs_symbols[0], rhs_symbols[1], np.log(prob)))
        return None
    
    def _load_data(self, path):
        with open(path, "r") as f:
            self.words = f.readlines()
        for i in range(len(self.words)):
            self.words[i] = self.words[i].strip().split()

    def _print_tree(self, w, sym):
        _sym, ind = re.sub(r"(.*?)\((.+?),\s(.+?)\)", r"\1 \2", sym).split()
        ind = int(ind)
        if sym in self.best_edge.keys():
            return "(" + _sym + " " \
                    + self._print_tree(w, self.best_edge[sym][0]) + " " \
                    + self._print_tree(w, self.best_edge[sym][1]) + ")"
        else:
            return "(" + _sym + " " + self.words[w][ind] + ")"
    
    def main(self):
        for w in range(len(self.words)):
            self.best_score = defaultdict(lambda: -INF)
            self.best_edge = dict()
            for i in range(len(self.words[w])):
                if self.preterm[self.words[w][i]] == []:
                    continue
                for lhs, log_prob in self.preterm[self.words[w][i]]:
                    self.best_score[f"{lhs}({i}, {i+1})"] = log_prob
            for j in range(2, len(self.words[w])+1):
                for i in range(j-2, -1, -1):
                    for k in range(i+1, j):
                        for sym, lsym, rsym, log_prob in self.nonterm:
                            if self.best_score[f"{lsym}({i}, {k})"] > -INF and self.best_score[f"{rsym}({k}, {j})"] > -INF:
                                my_lp = self.best_score[f"{lsym}({i}, {k})"] + self.best_score[f"{rsym}({k}, {j})"] + log_prob
                                if my_lp > self.best_score[f"{sym}({i}, {j})"]:
                                    self.best_score[f"{sym}({i}, {j})"] = my_lp
                                    self.best_edge[f"{sym}({i}, {j})"] = (f"{lsym}({i}, {k})", f"{rsym}({k}, {j})")
            print(self._print_tree(w, f"S(0, {len(self.words[w])})"))


if __name__ == "__main__":
    gram_path, token_path = "../../data/wiki-en-test.grammar", "../../data/wiki-en-short.tok"
    c = CKY(gram_path, token_path)
    c.main()