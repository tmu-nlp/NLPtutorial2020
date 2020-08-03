#python tutorial10.py ../../test/08-grammar.txt ../../test/08-input.txt
import sys
from collections import defaultdict
import math

class CKY():
    def __init__(self):
        self.nonterm = []
        self.preterm = defaultdict(lambda:[])
        self.minusInf = -100000000
    
    def grammer_input(self, grammer_file):
        g_f = open(grammer_file, "r", encoding="utf-8")
        for rule in g_f:
            lhs, rhs, prob = rule.strip().split("\t")
            rhs_symbols = rhs.split()
            if len(rhs_symbols) == 1:
                self.preterm[rhs].append((lhs,math.log(float(prob))))
            else:
                self.nonterm.append((lhs, rhs_symbols[0],rhs_symbols[1], math.log(float(prob))))
        g_f.close()
    
    def solve(self, input_file):
        i_f = open(input_file, "r", encoding="utf-8")
        for line in i_f:
            self.words = line.strip().split()
            self.best_score = defaultdict(lambda:self.minusInf)
            self.best_edge = {}
            for i in range(len(self.words)):
                for lhs, log_prob in self.preterm[self.words[i]]:
                    self.best_score[f"{lhs} {i} {i+1}"] = log_prob
            for j in range(len(self.words)-1):
                j=j+2
                for i in range(j-2,-1,-1):
                    for k in range(i+1,j):
                        for sym, lsym, rsym, logprob in self.nonterm:
                            if self.best_score[f"{lsym} {i} {k}"] > self.minusInf and self.best_score[f"{rsym} {k} {j}"] > self.minusInf:
                                my_lp = self.best_score[f"{lsym} {i} {k}"] + self.best_score[f"{rsym} {k} {j}"] + logprob
                                if my_lp > self.best_score[f"{sym} {i} {j}"]:
                                    self.best_score[f"{sym} {i} {j}"] = my_lp
                                    self.best_edge[f"{sym} {i} {j}"] = (f"{lsym} {i} {k}", f"{rsym} {k} {j}")
            print(self.output_tree(f"S 0 {len(self.words)}"))
        i_f.close()

    def output_tree(self, sym_i_j):
        sym, i, j = sym_i_j.split(" ")
        if sym_i_j in self.best_edge.keys():
            return "(" + sym + " " + self.output_tree(self.best_edge[sym_i_j][0]) + " " + self.output_tree(self.best_edge[sym_i_j][1]) + ")"
        else:
            return "(" + sym + " " + self.words[int(i)] + ")"

if __name__ == "__main__":
    cky = CKY()
    cky.grammer_input(sys.argv[1])
    cky.solve(sys.argv[2])

'''
[結果]
(S (NP_PRP i) (VP (VBD saw) (VP' (NP (DT a) (NN girl)) (PP (IN with) (NP (DT a) (NN telescope))))))
'''