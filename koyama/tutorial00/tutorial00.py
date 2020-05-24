import sys
from collections import defaultdict

def word_count(input_file_path):
    vocabulary = defaultdict(lambda: 0)
    with open(input_file_path, 'r') as input_file:
        for line in input_file:
            words = line.strip().split(" ")
            for w in words:
                vocabulary[w] += 1
    return vocabulary

if __name__ == "__main__":
    input_file_path = sys.argv[1] # ../nlptutorial/data/wiki-en-train.word
    output_file_path = sys.argv[2] # wiki-en-output.word
    counts = word_count(input_file_path)
    with open(output_file_path, "w") as output_file:
        # 頻度の降順に出力する
        for key, value in sorted(counts.items(), key=lambda x:x[1], reverse=True):
            print(f"{key}	{value}", file=output_file)