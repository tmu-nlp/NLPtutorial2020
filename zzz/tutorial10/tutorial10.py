import os
from collections import defaultdict
from math import log, inf

TEST = True
if TEST:
    PATH = '../test'
    INPUT_FILE = '08-input.txt'
    GRAMMAR_FILE = '08-grammar.txt'
else:
    PATH = '../data'
    INPUT_FILE = 'wiki­en­short.tok'
    GRAMMAR_FILE = 'wiki­en­test.grammar'
OUTPUT_FILE = '08-output.txt'


def load_grammar(filename):
    nonterm = []
    preterm = {}
    with open(filename) as f:
        for line in f:
            lhs, rhs, prob = line.split('\t')
            rhs = rhs.split(' ')
            if len(rhs) == 1:
                preterm[rhs] = (lhs, log(prob))
            else:
                nonterm.append((lhs, rhs[0], rhs[1], prob))
    return {'nonterm': nonterm,
            'preterm': preterm}


def cky(text, grammar):
    nonterm = grammar['nonterm']
    preterm = grammar['preterm']
    for line in text:
        best_score = defaultdict(lambda: -inf)
        best_edge = defaultdict(lambda: -inf)

        for (i, word) in enumerate(line.split(' ')):
            for lhs, log_prob in preterm:
                if preterm[word]:
                    best_score[f'{lhs}_{i}_{i + 1}'] = log_prob

        for (j, word) in enumerate(line.split(' ')[1:]):
            for i in range(j - 2, -1, -1):
                for k in range(i + 1, j):
                    for sym, lsym, rsym, log_prob in nonterm:
                        if best_score[f'{lsym}_{i}_{k}'] > -inf and best_score[f'{rsym}_{k}_{j}'] > -inf:
                            my_lp = best_score[f'{lsym}_{i}_{k}'] + best_score[f'{rsym}_{k}_{j}']
                            if my_lp > best_score[f'{sym}_{i}_{j}']:
                                best_score[f'{sym}_{i}_{j}'] = my_lp
                                best_edge[f'{sym}_{i}_{j}'] = (f'{lsym}_{i}_{k}', f'{rsym}_{k}_{j}')

        return best_edge


def print_tree(best_edge, sym):
    if sym in best_edge:
        return f'({sym} {print_tree(best_edge, best_edge[0])} {print_tree(best_edge, best_edge[1])})'
    else:
        return f'({sym} {sym.split("_")[1]}'


if __name__ == '__main__':
    grammar = load_grammar(os.path.join(PATH, GRAMMAR_FILE))
    with open(os.path.join(PATH, INPUT_FILE)) as f:
        text = [line for line in f]
        tree = cky(text, grammar)
