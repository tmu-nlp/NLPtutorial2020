from collections import defaultdict
import math

"""
file1 = "train-input-model.txt"
file2 = "01-test-input.txt"
"""

file1 = "model_file.txt"
file2 = "wiki-en-test.word"


prbs = defaultdict(int)
lambda_1 = 0.95
lambda_unk = 1 - lambda_1
V = 1000000
W = 0
H = 0
unk = 0

with open(file1, encoding="utf-8") as mdl_file:
    line = mdl_file.readline()
    while(line):
        w, P = line.split()
        prbs[w] = P
        line = mdl_file.readline()

with open(file2, encoding="utf-8") as tst_file:
    line = tst_file.readline()
    while(line):
        words = line.split()
        words.append("</s>")
        for word in words:
            W += 1
            P = lambda_unk/V
            if word in prbs:
                P += lambda_1*float(prbs[word])
            else:
                unk += 1
            H += -math.log(P, 2)
        line = tst_file.readline()

    print("entropy = {}".format(H/W))
    print("coverage = {}".format((W-unk)/W))

"""
entropy = 10.526656347101143
coverage = 0.895226024503591
"""