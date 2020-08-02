import re
import math
from collections import defaultdict

MINUS_INF = -float('inf')

class CKY():
    def __init__(self, grammar_file):
        """ definition of model
        
        :param grammar_file: path, grammar text file
        :param self.nonterm: list, CNF rule and log-probability
        :param self.preterm: dict, convert word into non-terminal symbol and log-probability
        """
        self.nonterm = []
        self.preterm = defaultdict(lambda: [])
        with open(grammar_file) as fp:
            for rule in fp:
                lhs, rhs, prob = rule.strip().split('\t')
                prob = float(prob)
                rhs_symbols = rhs.split()
                if len(rhs_symbols) == 1:
                    self.preterm[rhs].append((lhs, math.log(prob)))
                else:
                    self.nonterm.append((lhs, rhs_symbols[0], rhs_symbols[1], math.log(prob)))

    def main(self, input_file):
        """ predict non-terminal symbols from words (CKY algorithm)

        :param input_file: path, input text file
        """
        with open(input_file) as fp:
            for line in fp:
                self.words = line.strip().split()
                self.best_score = defaultdict(lambda: MINUS_INF)
                self.best_edge = {}
                for i in range(len(self.words)):
                    for lhs, log_prob in self.preterm[self.words[i]]:
                        self.best_score[f'{lhs} {i} {i+1}'] = log_prob
                for j in range(2, len(self.words)+1):
                    for i in range(j-2, -1, -1):
                        for k in range(i+1, j):
                            for sym, lsym, rsym, logprob in self.nonterm:
                                if self.best_score[f'{lsym} {i} {k}'] > MINUS_INF and self.best_score[f'{rsym} {k} {j}'] > MINUS_INF:
                                    my_lp = self.best_score[f'{lsym} {i} {k}'] + self.best_score[f'{rsym} {k} {j}'] + logprob
                                    if my_lp > self.best_score[f'{sym} {i} {j}']:
                                        self.best_score[f'{sym} {i} {j}'] = my_lp
                                        self.best_edge[f'{sym} {i} {j}'] = (f'{lsym} {i} {k}', f'{rsym} {k} {j}')
                print(self.print_tree(f'S 0 {len(self.words)}'))

        return self

    def print_tree(self, sym_ij):
        sym, i, _ = sym_ij.split()

        i = int(i)
        if sym_ij in self.best_edge.keys():
            return '(' + sym + ' ' \
                    + self.print_tree(self.best_edge[sym_ij][0]) + ' ' \
                    + self.print_tree(self.best_edge[sym_ij][1]) + ')'
        else:
            return '(' + sym + ' ' + self.words[i] + ')'

if __name__ == '__main__':
    #grammar_file = './test/08-grammar.txt'
    #input_file = './test/08-input.txt'
    grammar_file = './data/wiki-en-test.grammar'
    input_file = './data/wiki-en-short.tok'
    cky = CKY(grammar_file)
    cky.main(input_file)

