#python train_HMM.py ../test/05-train-input.txt
#python train_HMM.py ../data/wiki-en-train.norm_pos
import sys
from collections import defaultdict
emit = defaultdict(lambda: 0)
transition = defaultdict(lambda: 0)
context = defaultdict(lambda: 0)

with open(sys.argv[1]) as f\
    , open("model.txt", mode='w') as outfile:
    for line in f:
        previous = '<s>'
        context[previous] += 1
        wordtags = line.split()
        for wordtag in wordtags:
            word, tag = wordtag.split('_')
            transition[f"{previous} {tag}"] += 1
            context[tag] += 1
            emit[f"{tag} {word}"] += 1
            previous = tag
        transition[f"{previous} </s>"] += 1
    for key, value in transition.items():
        previous, tag = key.split()
        #P_T, "前のタグ 次のタグ", c(previous tag)/c(previous)
        #遷移確率
        prob = value/context[previous]
        outfile.write(f"T {key} {prob}\n")
    for key, value in emit.items():
        tag, word = key.split()
        #生成確率
        prob = value/context[tag]
        outfile.write(f"E {key} {prob}\n")
