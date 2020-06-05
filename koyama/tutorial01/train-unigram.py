import sys
from collections import defaultdict

def train_unigram(input_file_path):
    vocab = defaultdict(lambda: 0)
    word_sum = 0
    with open(input_file_path, "r") as input_file:
        for line in input_file:
            words = line.strip().split()
            words.append("</s>")
            for word in words:
                vocab[word] += 1
                word_sum += 1
    return vocab, word_sum

if __name__ == "__main__":
    input_file_path = sys.argv[1] # ../nlptutorial/data/wiki-en-train.word
    output_file_path = sys.argv[2] # wiki-en-output.word
    counts, total_count = train_unigram(input_file_path)
    with open(output_file_path, "w") as output_file:
        for key, value in sorted(counts.items(), key=lambda x:x[0]):
            probability = value / total_count
            print(f"{key}   {probability}", file=output_file)
