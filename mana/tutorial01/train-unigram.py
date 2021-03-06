import sys
from collections import defaultdict


def train_unigram(file):

    """
    what this function does:
        create a Uni-Gram language model from a given text

    arg:
        file(file), a given text
    
    variables:
        counts(dict), word counts
        total_count(int), total word occurence in a given text

    output:
        textfile(word entry + its probability in a given text)
    """

    counts = defaultdict(lambda: 0)
    total_count = 0
    dataFile = open(file, "r").readlines()
    for line in dataFile:
        words = line.split()
        words.append("</s>")
        for word in words:
            counts[word.lower()] += 1  # using .lower() --> +0.05 (coverage)
            total_count += 1

    modelFile = open("model_file.txt", "w")
    for word in counts:
        probability = counts[word] / total_count
        modelFile.write(word + " " + str(probability) + "\n")


if __name__ == "__main__":
    train_unigram("./data/wiki-ja-train.word")
