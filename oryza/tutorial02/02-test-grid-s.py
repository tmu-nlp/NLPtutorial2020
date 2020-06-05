import sys
from collections import defaultdict
import math



def test_ngram(model,test): 
    lambda_1 = 0.95
    lambda_2 = 0.85
    V = 1000000
    W = 0
    H = 0

    model_file = open(model)
    test_file = open(test)
    probs = defaultdict(lambda:0)

    for line in model_file:
        words = line.split()
        w = words[0].replace('_',' ')
        P = words[1]
        probs[w] = float(P)

    for line in test_file:
        words = line.split()
        words.append('</s>')
        words.insert(0,'<s>')

        for i in range(1,len(words)):
            P1 = lambda_1 * probs[words[i]] + (1 - lambda_1) / V
            P2 = lambda_2 * probs[words[i-1] + '_' + words[i]] + (1 - lambda_2) * P1
            H += -math.log2(P2)
            W += 1

    print('Entropy = ' + str(abs(round(H / W, 6))))

if __name__ == '__main__':
    test_ngram(sys.argv[1],sys.argv[2])

# 02-test.py model-file.txt 02-test-input.txt
# 02-test.py model-file-wiki.txt wiki-en-test.word


