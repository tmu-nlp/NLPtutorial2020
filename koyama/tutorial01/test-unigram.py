import sys
import math

def get_prob(input_file_path):
    prob = {}
    with open(input_file_path, "r") as input_file:
        for line in input_file:
            line = line.strip().split()
            # word := line[0], probability := line[1]
            prob[line[0]] = float(line[1])
    return prob

def test_unigram(input_file_path, probabilities):
    lambda_1 = 0.95
    lambda_unk = 1 - lambda_1
    V = 1000000 # 未知語を含む語彙数
    W = 0       # 単語数
    unk = 0     # 未知語数
    H = 0       # 対数尤度
    with open(input_file_path, "r") as input_file:
        for line in input_file:
            words = line.strip().split()
            words.append("</s>")
            for w in words:
                W += 1
                P = lambda_unk / V
                if w in probabilities:
                    P += lambda_1 * probabilities[w]
                else:
                    unk += 1
                H += -math.log(P, 2)
    entropy = H / W
    coverage = (W - unk) / W
    return entropy, coverage


if __name__ == "__main__":
    prob_file_path = sys.argv[1] # wiki-en-output.word
    test_file_path = sys.argv[2] # ../nlptutorial/data/wiki-en-test.word
    probabilities = get_prob(prob_file_path)
    entropy, coverage = test_unigram(test_file_path, probabilities)
    print(f"{entropy=}")
    print(f"{coverage=}")

# 実行結果
# entropy=10.527337238682652
# coverage=0.895226024503591
