from collections import defaultdict
import math
import numpy


class Ngram:
    def __init__(self, lambda_1, lambda_2):
        self.counts = defaultdict(int)
        self.context_counts = defaultdict(int)
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

    def trainNgram(self, inputFile):
        model = defaultdict(int)
        with open(inputFile, "r") as train:
            for line in train:
                line = line.lower().split(" ")
                line.append("</s>")
                line.insert(0, "<s>")

                for i in range(1, len(line)):
                    self.counts[" ".join(line[i - 1 : i + 1])] += 1
                    self.context_counts[line[i - 1]] += 1
                    self.counts[line[i]] += 1
                    self.context_counts[""] += 1

        for ngram in self.counts:
            context = ngram.split(" ")
            context = "".join(context[:-1])
            probability = self.counts[ngram] / self.context_counts[context]
            model[ngram] = probability
        return model

    def testNgram(self, modeldic, testFile):
        W = 0
        H = 0
        V = 10 ** 6
        with open(testFile, "r") as test:
            for line in test:
                line = line.lower().split()
                line.append("</s>")
                line.insert(0, "<s>")
                for i in range(1, len(line) - 1):
                    P1 = self.lambda_1 * modeldic[line[i]] + (1 - self.lambda_1) / V
                    P2 = (
                        self.lambda_2 * modeldic[" ".join(line[i - 1 : i])]
                        + (1 - self.lambda_2) * P1
                    )
                    H += math.log(1 / P2, 2)
                    W += 1
        return str(round(H / W, 4))


if __name__ == "__main__":
    trainpath = "/work/data/wiki-en-train.word"
    testpath = "/work/data/wiki-en-test.word"

    """
    #for gridsearch
    lambdas = numpy.arange(0, 1, 0.05)

    results = {}
    for param1 in lambdas:
        for param2 in lambdas:
            NgramLM = Ngram(param1, param2)
            modelDic = NgramLM.trainNgram(trainpath)
            results[NgramLM.testNgram(modelDic, testpath)] = str(param1) + "_" + str(param2)

    #for elem in sorted(results):
    #    print(elem +" "+ results[elem])

    #8.45892168895307 0.9500000000000001_0.5
    #8.460034910530998 0.9500000000000001_0.45
    #8.467973251551467 0.9500000000000001_0.55
    """

    NgramLM = Ngram(0.95, 0.5)
    model = NgramLM.trainNgram(trainpath)
    print("entropy = " + NgramLM.testNgram(model, testpath), end=", ")
    print("when lambda1:0.95, lambda2:0.5")
