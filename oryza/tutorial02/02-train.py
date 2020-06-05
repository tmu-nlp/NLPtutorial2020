import sys
from collections import defaultdict

def train_bigram(train, model): 
    counts = defaultdict(lambda:0)
    context_counts = defaultdict(lambda:0)
    
    tr_file = open(train)

    for line in tr_file:
        words = line.split()
        words.append('</s>')
        words.insert(0,'<s>')
    
        for i in range(1,len(words)):
            counts[words[i-1] + '_' + words[i]] += 1
            context_counts[words[i-1]] += 1
            counts[words[i]] += 1
            context_counts[''] += 1

    model_file = open(model, 'w')
    for ngram, count in sorted(counts.items()):
        w_split = ngram.split('_')
        if len(w_split) > 1:
            w = w_split[0]
        else:
            w = ''
        
        probability = counts[ngram]/context_counts[w]
        model_file.write(ngram + '\t' + str(round(probability,6)) + '\n')

if __name__ == '__main__':
    train_bigram(sys.argv[1],sys.argv[2])

# 02-train.py 02-train-input.txt model-file.txt
# 02-train.py wiki-en-train.word model-file-wiki.txt