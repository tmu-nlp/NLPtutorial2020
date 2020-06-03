import os
import joblib
from math import log2
from collections import defaultdict, OrderedDict

class UnigramLanguageModel():
    def __init__(self):
        self.total_cnt = 0
        self.cnts = defaultdict(lambda: 0)
        self.probs = defaultdict(lambda: 0)
        self.lambda_1 = 0.95
        self.lambda_unk = (1 - self.lambda_1)

    def train(self, filename: str):
        # Validate
        assert os.path.exists(filename), f"{filename} does not exist."
        # Read text
        with open(filename) as f:
            for line in f:
                words = line.split()
                words.append("</s>")
                # Count total word and individual words
                for word in words:
                    self.cnts[word] += 1
                    self.total_cnt += 1
        # Calculate probability
        for word, cnt in self.cnts.items():
            self.probs[word] = cnt / self.total_cnt
        # Sort dictionary with word
        self.cnts = OrderedDict(sorted(self.cnts.items()))
        self.probs = OrderedDict(sorted(self.probs.items()))

    def save(self, filename: str):
        # defaultdictは保存できないので、dillとかdictで工夫する
        joblib.dump(dict(self.probs), filename, compress=3)

    def load(self, filename: str):
        self.probs = joblib.load(filename)

    def save_txt(self, filename: str):
        with open(filename, mode='w') as f:
            for word, w_prob in self.probs.items():
                f.write(f"{word} {w_prob}\n")

    def load_txt(self, filename: str):
        with open(filename) as f:
            for word, w_prob in self.probs.items():
                for line in f:
                    word, prob = line.split()
                    self.probs[word] = float(prob)

    def test(self, filename: str):
        # Validate
        V = 1e6; W = 0; H = 0; unk_cnt = 0
        assert os.path.exists(filename), f"{filename} does not exist."
        with open(filename) as f:
            for line in f:
                words = line.split()
                words.append("</s>")
                for w in words:
                    W += 1
                    P = self.lambda_unk / V
                    if w in self.probs:
                        P += self.lambda_1 * self.probs[w]
                    else:
                        unk_cnt += 1
                    H -= log2(P)
        print(f"Entropy   : {H/W:.6f}")
        print(f"Perplexity: {2**(H/W):.6f}")
        print(f"Coverage  : {(W-unk_cnt)/W:.6f}")

if __name__ == "__main__":
    UniLM = UnigramLanguageModel()

    # train_file = "../../test/01-train-input.txt"
    # test_file = "../../test/01-test-input.txt"
    """
    Entropy   : 6.709899
    Perplexity: 104.684170
    Coverage  : 0.800000
    """

    train_file = "../../data/wiki-en-train.word"
    test_file = "../../data/wiki-en-test.word"
    """
    Entropy   : 10.527337
    Perplexity: 1475.857013
    Coverage  : 0.895226
    """

    UniLM.train(train_file)

    txt_file = "tutorial01.txt"
    UniLM.save_txt(txt_file)
    UniLM.load_txt(txt_file)
    UniLM.test(test_file)