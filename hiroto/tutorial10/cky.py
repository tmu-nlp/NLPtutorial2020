import re, sys
from math import log
from collections import defaultdict
MINF = -float('inf')

class CKY():
    def load_grammer(self, filename):
        self.nonterm = []
        self.preterm = defaultdict(lambda: [])
        with open(filename) as f:
            for rule in f:
                lhs, rhs, prob = rule.strip().split('\t')
                rhs_symbol = rhs.split()
                prob = float(prob)
                #rhs_symbols = rhs.split()
                if len(rhs_symbol) == 1:
                    #if rhs not in preterm.keys():
                    #    preterm[rhs] = []
                    self.preterm[rhs].append((lhs, log(prob)))
                else:
                    self.nonterm.append((lhs, rhs_symbol[0], rhs_symbol[1], log(prob)))
    
    def exec(self, filename):
        with open(filename) as f:
            for line in f:
                self.words = line.strip().split()
                self.best_score = defaultdict(lambda:MINF)
                self.best_edge = {}
                for i in range(len(self.words)):
                    if self.preterm[self.words[i]] != []:
                        for lhs, log_prob in self.preterm[self.words[i]]:
                            self.best_score[f"{lhs}({i}, {i+1})"] = log_prob
                for j in range(2, len(self.words)+1):
                    for i in range(j-2, -1, -1):#range(j-1)[::-1]:#range(j-2, -1, -1):
                        for k in range(i+1, j):
                            for sym, lsym, rsym, logprob in self.nonterm:
                                if self.best_score[f"{lsym}({i}, {k})"] > MINF and self.best_score[f"{rsym}({k}, {j})"] > MINF:
                                    my_lp = self.best_score[f"{lsym}({i}, {k})"] + self.best_score[f"{rsym}({k}, {j})"] + logprob
                                    if my_lp > self.best_score[f"{sym}({i}, {j})"]:
                                        self.best_score[f"{sym}({i}, {j})"] = my_lp
                                        self.best_edge[f"{sym}({i}, {j})"] = (f"{lsym}({i}, {k})", f"{rsym}({k}, {j})")
                
                # S式の始め
                print(self.print_tree(f"S(0, {len(self.words)})"))
    
    def print_tree(self, sym_ij):
        sym, i = re.sub(r"(.*?)\((.+?),\s(.+?)\)", r"\1 \2", sym_ij).split()
        i = int(i)
        if sym_ij in self.best_edge.keys():
            return "(" + sym + " " \
                    + self.print_tree(self.best_edge[sym_ij][0]) + " " \
                    + self.print_tree(self.best_edge[sym_ij][1]) + ")"
        else:
            return "(" + sym + " " + self.words[i] + ")"
    
if __name__ == '__main__':
    cky = CKY()
    cky.load_grammer(sys.argv[1])
    cky.exec(sys.argv[2])

#python cky.py ../test/08-grammar.txt ../test/08-input.txt
#python cky.py ../data/wiki-en-test.grammar ../data/wiki-en-short.tok