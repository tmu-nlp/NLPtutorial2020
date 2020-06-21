#python test_HMM.py ../test/05-test-input.txt
#python test_HMM.py ../data/wiki-en-test.norm
#../script/gradepos.pl ../data/wiki-en-test.pos my_answer.pos
import sys
from math import log
from collections import defaultdict
N = 1000000
LAMBDA = 0.95
emission = defaultdict(lambda: 0)
transition = defaultdict(lambda: 0)
possible_tags = defaultdict(lambda: 0)

#モデル読み込み
with open('./model.txt') as model_file:
    for line in model_file:
        type, context, word, prob = line.split()
        possible_tags[context] = 1
        #遷移確率
        if type == 'T':
            transition[f"{context} {word}"] = float(prob)
        #生成確率
        else:
            emission[f"{context} {word}"] = float(prob)

with open(sys.argv[1]) as f\
    , open("my_answer.pos", mode='w') as outfile:
    for line in f:
        #前向きステップ
        line = line.strip()
        words = line.split()
        l = len(words)
        #best_score, best_edge = defaultdict(lambda: float('inf')), defaultdict(lambda: None)
        best_score, best_edge = {}, {}
        #<s>から始まる
        best_score["0 <s>"] = 0
        best_edge["0 <s>"] = None
        for i in range(l):
            for prev in possible_tags.keys():
                for next in possible_tags.keys():
                    if f"{i} {prev}" in best_score.keys() and transition[f"{prev} {next}"]:
                        P_T = transition[f"{prev} {next}"]
                        #平滑化
                        P_E = LAMBDA*emission[f"{next} {words[i]}"] + (1-LAMBDA)/N
                        score = best_score[f"{i} {prev}"] + (-log(P_T)) + (-log(P_E))
                        if f"{i+1} {next}" not in best_score.keys() \
                            or best_score[f"{i+1} {next}"] > score:
                            best_score[f"{i+1} {next}"] = score
                            best_edge[f"{i+1} {next}"] = f"{i} {prev}"

        #</s>に対しての操作
        best_score[f"{l+1} </s>"] = float('inf')
        for prev in possible_tags.keys():
            if f"{l} {prev}" in best_score.keys() and transition[f"{prev} </s>"]:
                P_T = transition[f"{prev} </s>"]
                score = best_score[f"{l} {prev}"] + (-log(P_T))
                if f"{l+1} </s>" not in best_score.keys() \
                    or best_score[f"{l+1} </s>"] > score:
                    best_score[f"{l+1} </s>"] = score
                    best_edge[f"{l+1} </s>"] = f"{l} {prev}"

        #後ろ向きステップ
        tags = []
        next_edge = best_edge[f"{l+1} </s>"]
        while next_edge != "0 <s>":
            #このエッジの品詞を出力に追加
            position, tag = next_edge.split()
            tags.append(tag)
            next_edge = best_edge[next_edge]
        tags.reverse()
        string = ' '.join(tags)
        outfile.write(f"{string}\n")