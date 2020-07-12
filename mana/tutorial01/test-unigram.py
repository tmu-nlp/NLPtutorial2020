import sys
import math


def test_unigram(model_file, test_file):
    """
    what this function does:
        model validation with a given text

    args:
        model_file(file), word entry + its probability
        test_file(file), a given text

    variables:
        W(int), total word count
        H(float), likelihood
        unk_words(int), total unkown word occurence
        probabilities(dict), dict from a model

    params:
        lambda_1, lambda_unk(float)
        V(int), total word entry including unknown words

    output:
        entropy(float)
        coverage(float)

    """

    lambda_1 = 0.95
    lambda_unk = 1 - lambda_1
    V = 1000000
    W = 0
    H = 0
    unk_words = 0
    probabilities = {}

    modelFile = open(model_file, "r").readlines()
    for line in modelFile:
        word, prob = line.split()
        probabilities[word] = float(prob)

    testFile = open(test_file, "r").readlines()
    for line in testFile:
        words = line.split()
        words.append("</s>")
        for word in words:
            word = word.lower()
            W += 1
            P = lambda_unk / V
            if word in probabilities:
                P += lambda_1 * probabilities[word]
            else:
                unk_words += 1
            H += math.log(1 / P, 2)
    return "entropy = " + str(H / W) + "\n" + "coverage = " + str((W - unk_words) / W)


if __name__ == "__main__":
    print(test_unigram("model_file.txt", "wiki-en-test.word"))
