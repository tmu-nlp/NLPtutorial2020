import sys
from collections import defaultdict

def train_unigram(train, model): 
    counts = defaultdict(lambda:0)
    total_count = 0
    tr_file = open(train)

    for line in tr_file:
        words = line.split()
        words.append('</s>')
    
        for w in words:
            counts[w] += 1
            total_count += 1

    model_file = open(model,'w')
    for word, count in sorted(counts.items()):
        probability = counts[word]/total_count
        model_file.write(word + '\t' + str(probability) + '\n')

if __name__ == '__main__':
    train_unigram(sys.argv[1],sys.argv[2])

# 01-train.py 01-train-input.txt model_file.txt