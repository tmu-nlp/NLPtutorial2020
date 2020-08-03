r"""tutorial10.py
CKY

[Usage]
python tutorial10.py main &> out
python tutorial10.py main 2>&1 | pbcopy
"""
import math
import os
import re
import sys
from collections import defaultdict
from itertools import islice

from nltk.tree import Tree

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.message import message, Renderer  # noqa: E402 isort:skip

INF = float("inf")
prog = re.compile(r"[()]")


def load_grammar(grammar_file):
    """ #10 p.65 """
    nonterm = []
    preterm = defaultdict(list)  # preterm[右] := [(左, 確率) ...]
    for rule in open(grammar_file):
        lhs, rhs, prob = rule.split("\t")  # P(左 -> 右) = 確率
        rhs_symbols = rhs.split()
        prob = float(prob)
        if len(rhs_symbols) == 1:  # 前終端記号
            preterm[rhs] += [(lhs, math.log(prob))]
        else:  # 非終端記号
            nonterm += [(lhs, rhs_symbols[0], rhs_symbols[1], math.log(prob))]
    return nonterm, preterm


def cky(grammar_file, input_file, s=0, t=57):
    """ #10 pp.66-67 """
    nonterm, preterm = load_grammar(grammar_file)
    for line in islice(open(input_file), s, t):
        words = line.split()
        # best_score[sym_{i, j}] := 最大対数確率
        best_score = defaultdict(lambda: -INF)
        # best_edge[sym_{i, j}] := (lsym_{i, k}, rsym_{k, j})
        best_edge = {}
        # 前終端記号を追加
        for i in range(len(words)):
            if preterm[words[i]]:
                for lhs, log_prob in preterm[words[i]]:
                    best_score[f"{lhs} ({i} {i+1})"] = log_prob
        # 非終端記号の組み合わせ
        for j in range(2, len(words) + 1):
            for i in range(j - 2, -1, -1):
                for k in range(i + 1, j):
                    # log(P(sym -> lsym rsym)) = log prob
                    for sym, lsym, rsym, logprob in nonterm:
                        par = f"{sym} ({i} {j})"
                        left = f"{lsym} ({i} {k})"
                        right = f"{rsym} ({k} {j})"
                        # 両方の子供の確率が 0 より大きい
                        if best_score[left] == -INF:
                            continue
                        if best_score[right] == -INF:
                            continue
                        # このノード・辺の対数確率を計算
                        my_lp = best_score[left] + best_score[right] + logprob
                        # この辺が確率最大のものなら更新
                        if my_lp > best_score[par]:
                            best_score[par] = my_lp
                            best_edge[par] = (left, right)
        yield get_S_expr(f"S (0 {len(words)})", best_edge, words)


def get_S_expr(sym_ij, best_edge, words):
    """ #10 p.68 """
    sym, i, _ = prog.sub("", sym_ij).split()
    if sym_ij in best_edge:
        left = get_S_expr(best_edge[sym_ij][0], best_edge, words)
        right = get_S_expr(best_edge[sym_ij][1], best_edge, words)
        return f"({sym} {left} {right})"
    else:
        return f"({sym} {words[int(i)]})"


if __name__ == "__main__":
    if sys.argv[1] == "test":
        grammar_file = "../../test/08-grammar.txt"
        input_file = "../../test/08-input.txt"
    else:
        grammar_file = "../../data/wiki-en-test.grammar"
        input_file = "../../data/wiki-en-short.tok"

    s, t = 0, 1
    with Renderer(sys.argv[1]) as out:
        for i, s_expr in enumerate(cky(grammar_file, input_file, s=s, t=t)):
            message("=" * 3, "line:", s + i, "=" * 3)
            tree = Tree.fromstring(s_expr)
            out.result("S-expression", s_expr)
            out.result("nltk.tree.Tree", tree)
            out.header("nltk.tree.Tree.pretty_print")
            tree.pretty_print()
            # tree.draw()


"""result
[+] main
=== line: 0 ===
[*]  1. S-expression
(S (PP (IN Among) (NP (DT these) (NP' (, ,) (NP' (JJ supervised) (NP' (NN learning) (NNS approaches)))))) (S' (VP (VBP have) (VP (VBN been) (VP' (NP (DT the) (NP' (ADJP (RBS most) (JJ successful)) (NNS algorithms))) (PP (TO to) (NP_NN date))))) (. .)))
[*]  2. nltk.tree.Tree
(S
  (PP
    (IN Among)
    (NP
      (DT these)
      (NP'
        (, ,)
        (NP' (JJ supervised) (NP' (NN learning) (NNS approaches))))))
  (S'
    (VP
      (VBP have)
      (VP
        (VBN been)
        (VP'
          (NP
            (DT the)
            (NP' (ADJP (RBS most) (JJ successful)) (NNS algorithms)))
          (PP (TO to) (NP_NN date)))))
    (. .)))
[*]  3. nltk.tree.Tree.pretty_print
                                                             S
              _______________________________________________|_____________________
             |                                                                     S'
             |                                                _____________________|________________________________
             |                                               VP                                                     |
             |                                           ____|____                                                  |
             PP                                         |         VP                                                |
   __________|______                                    |     ____|________________                                 |
  |                 NP                                  |    |                    VP'                               |
  |      ___________|_______                            |    |              _______|______________________          |
  |     |                  NP'                          |    |             NP                             |         |
  |     |     ______________|_____                      |    |     ________|_______                       |         |
  |     |    |                   NP'                    |    |    |               NP'                     |         |
  |     |    |       _____________|______               |    |    |         _______|__________            |         |
  |     |    |      |                   NP'             |    |    |       ADJP                |           PP        |
  |     |    |      |              ______|______        |    |    |    ____|_______           |        ___|____     |
  IN    DT   ,      JJ            NN           NNS     VBP  VBN   DT RBS           JJ        NNS      TO     NP_NN  .
  |     |    |      |             |             |       |    |    |   |            |          |       |        |    |
Among these  ,  supervised     learning     approaches have been the most      successful algorithms  to      date  .
"""
