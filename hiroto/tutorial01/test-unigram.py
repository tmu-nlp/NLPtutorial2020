import sys, math
from collections import defaultdict
lambda_1 = 0.95
lambda_unk = 1 - lambda_1
V, W, H = 1000000, 0, 0
unk = 0
probas_dic = defaultdict(lambda: 0)

with open(sys.argv[1]) as model_file\
    , open(sys.argv[2]) as test_file:
    #モデル読み込み
    for line in model_file:
        word, proba = line.split()
        probas_dic[word] = proba
    #評価と結果表示
    for line in test_file:
        words = line.split()
        words.append('</s>')
        for word in words:
            W += 1
            P = lambda_unk / V
            if probas_dic[word]:
                P += lambda_1 * float(probas_dic[word])
            else:
                unk += 1
            H += -math.log(P, 2)

    print(f'entropy = {H/W: .6f}')
    print(f'coverage = {(W-unk)/W: .6f}')
    print(f'perplexity = {2 ** (H/W): .6f}')
