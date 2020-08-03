import math
from collections import defaultdict

def read_grammar(grammar_file):
    nonterm = []
    preterm = defaultdict(list)
    for rule in grammar_file:
        rule = rule.strip()
        lhs, rhs, prob = rule.split('\t')
        rhs_symbols = rhs.split(' ')
        if len(rhs_symbols) == 1: # If this is a pre-terminal
            preterm[rhs].append((lhs, math.log2(float(prob))))
        else: # Otherwise, it is a non-terminal
            nonterm.append((lhs, rhs_symbols[0], rhs_symbols[1], math.log2(float(prob))))
    
    return nonterm, preterm

def process_input_line(input_line, nonterm, preterm):
    line = input_line.strip()
    words = line.split(' ')
    best_score = defaultdict(lambda: -math.inf)
    best_edge = {}

    # Add the pre-terminal sym
    for i in range(len(words)):
        for lhs, log_prob in preterm[words[i]]:
            best_score[f'{lhs} {i} {i+1}'] = log_prob
    # print(best_score)
    
    for j in range(2, len(words)): # j is right side of the span
        for i in range(j-2, -1, -1): # i is left side (Note: Reverse order!)
            for k in range(i+1, j): # k is beginning of the second child

                # Try every grammar rule log(P(sym → lsym rsym)) = logprob
                for sym, lsym, rsym, logprob in nonterm:
                    # Both children must have a probability
                    if best_score[f'{lsym} {i} {k}'] > -math.inf and best_score[f'{rsym} {k} {j}'] > -math.inf:
                        # Find the log probability for this node/edge
                        my_lp = best_score[f'{lsym} {i} {k}'] + best_score[f'{rsym} {k} {j}'] + logprob
                        # print(my_lp)

                        # If this is the best edge, update
                        if my_lp > best_score[f'{sym} {i} {j}']:
                            best_score[f'{sym} {i} {j}'] = my_lp
                            best_edge[f'{sym} {i} {j}'] = (f'{lsym} {i} {k}', f'{rsym} {k} {j}')

    return words, best_edge

def subroutine(sym_i_j, words, best_edge):
    sym, i, j = sym_i_j.split(' ')
    if sym_i_j in best_edge:  # for non-terminals
        return f'({sym} {subroutine(best_edge[sym_i_j][0], words, best_edge)} {subroutine(best_edge[sym_i_j][1], words, best_edge)})'
    else:  # for terminals
        return f'({sym} {words[int(i)]})'

if __name__ == "__main__":
    # grammar_file = open('08-grammar.txt')
    # input_file = open('08-input.txt')
    grammar_file = open('wiki-en-test.grammar')
    input_file = open('wiki-en-short.tok')

    nonterm, preterm = read_grammar(grammar_file)
    
    for line in input_file:
        words, best_edge = process_input_line(line, nonterm, preterm)
        print(subroutine(f'S 0 {len(words)-1}', words, best_edge))
