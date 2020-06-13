from collections import defaultdict
import math

test = "../../data/wiki-en-test.word"
model = "./model_file.txt"

lambda1 = 0.85
lambda2 = 0.35
V = 1000000
W = 0
H = 0

probs = defaultdict(int)


with open(model, encoding="utf-8") as model_file:
    line = model_file.readline()
    while(line):
        ngram, prob = line.split("\t")
        probs[ngram] = float(prob)
        line = model_file.readline()

with open(test, encoding="utf-8") as test_file:
    line = test_file.readline()
    while(line):
        words = line.split()
        line = test_file.readline()
        words.append("</s>")
        words.insert(0, "<s>")
        for i in range(len(words) - 1):
            P1 = lambda1*probs[words[i+1]] + (1-lambda1)/V
            P2 = lambda2*probs[words[i] + " " + words[i+1]] + (1-lambda2)*P1
            H += -math.log(P2, 2)
            W += 1

print("entropy = {}".format(H/W))

