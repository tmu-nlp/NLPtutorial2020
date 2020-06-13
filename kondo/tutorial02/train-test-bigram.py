from collections import defaultdict
import math

counts = defaultdict(int)
context_counts = defaultdict(int)
probs = defaultdict(int)

#train = "../../test/02-train-input.txt"
train = "../../data/wiki-en-train.word"

model = "./model_file.txt"
test = "../../data/wiki-en-test.word"


def train_bigram(file):
    with open(file, encoding="utf-8") as tra_file:
        line = tra_file.readline()
        while(line):
            words = line.split()
            words.append("</s>")
            words.insert(0, "<s>")
            for i in range(len(words) - 1):
                counts[words[i]+" "+words[i+1]] += 1
                context_counts[words[i]] += 1
                counts[words[i+1]] += 1
                context_counts[""] += 1
            line = tra_file.readline()

def culc_prob(file):
    with open(file, "w", encoding="utf-8") as model:
        for ngram, count in counts:
            words = ngram.split()
            words[-1] = ""
            prob = count/context_counts[words[0]]
            model.write("{}\t{:.6f}\n".format(ngram, prob))

def get_probs(file):
    with open(file, encoding="utf-8") as model_file:
        line = model_file.readline()
        while(line):
            ngram, prob = line.split("\t")
            probs[ngram] = float(prob)
            line = model_file.readline()

def get_H(file):
    ans = []
    for lambda2 in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        lambda1 = 0.5
        V = 1000000
        W = 0
        H = 0
        with open(file, encoding="utf-8") as test_file:
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
        ans.append(H/W)
    return ans

if __name__ == "__main__":
    train_bigram(train)
    counts = sorted(counts.items())
    culc_prob(model)
    get_probs(model)
    for i in range(len(get_H(test))):
        print("lambda2={:.1f}\t{}".format((i + 1)*0.1, get_H(test)[i]))

