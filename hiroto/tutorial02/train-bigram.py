from collections import defaultdict
import sys
cnts = defaultdict(lambda: 0)
context_cnts = defaultdict(lambda: 0)

with open(sys.argv[1]) as training_file\
    , open(sys.argv[2], mode='w') as model_file:
    for line in training_file:
        words = line.split()
        words.insert(0, '<s>')
        words.append('</s>')
        for i in range(1, len(words)):
            cnts[f'{words[i-1]} {words[i]}'] += 1
            context_cnts[words[i-1]] += 1
            cnts[words[i]] += 1
            context_cnts[''] += 1

    for n_gram, cnt in sorted(cnts.items(), key=lambda x:x[0]):
        words = n_gram.split()
        #uni-gramの時，分母は文中の単語数
        if len(words) == 1:
            prob = float(cnts[n_gram] / context_cnts[''])
        #bi-gram
        else:
            prob = float(cnts[n_gram] / context_cnts[words[0]])
            
        model_file.write(f'{n_gram}\t{prob:.6f}\n')