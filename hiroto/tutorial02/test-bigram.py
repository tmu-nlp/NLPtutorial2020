#線形補間でlambda_1, lambda_2を決める
import sys, math
import numpy as np
from collections import defaultdict
from tqdm import tqdm
#lambdaの取りうる値の範囲 0.05~0.95, 0,05ずつ
lambda_range = np.arange(0.05, 1.00, 0.05)
V = 1000000
#judge関数でエントロピーを比較する時に使う
min_entropy = float('inf')
probs = defaultdict(lambda: 0)

#それぞれのラムダの組み合わせでエントロピーが最も小さいか比較する
def judge(entropy):
    if entropy < min_entropy:
        return True
    else: return False

def cal_entropy(lambda_1, lambda_2):
    #スライド通り
    W, H = 0, 0
    for line in f:
        words = line.split()
        words.insert(0, '<s>')
        words.append('</s>')
        for i in range(1, len(words)):
            P1 = lambda_1*probs[words[i]] + (1-lambda_1)/V
            P2 = lambda_2*probs[f'{words[i-1]} {words[i]}'] + (1-lambda_2)*P1
            H += -math.log(P2, 2)
            W += 1
    entropy = H/W
    return entropy

with open(sys.argv[1]) as model_file\
    , open(sys.argv[2]) as test_file:
    #モデル読み込み
    for line in model_file:
        n_gram, prob = line.split('\t')
        probs[n_gram] = float(prob)
    #評価と結果表示
    f = test_file.readlines()
    for lambda_1 in tqdm(lambda_range):
        for lambda_2 in lambda_range:
            entropy = cal_entropy(lambda_1, lambda_2)
            if judge(entropy):
                min_entropy = entropy
                LAMBDA_1 = lambda_1
                LAMBDA_2 = lambda_2
            else: pass

    print(f'entropy = {min_entropy:.6f}')
    print(f'lambda_1 = {LAMBDA_1:.2f}, lambda_2 = {LAMBDA_2:.2f}')

'''出力結果
entropy = 9.663336
lambda_1 = 0.85, lambda_2 = 0.35
'''