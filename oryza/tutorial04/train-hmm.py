import sys
from collections import defaultdict

def train_hmm(train_file, model_file):
    emit = defaultdict(lambda:0)
    transition = defaultdict(lambda:0)
    context = defaultdict(lambda:0)

    for line in train_file:
        previous = '<s>'
        context[previous] += 1
        wordtags = line.strip().split(' ')
        for wt in wordtags:
            wt = wt.split('_')
            word = wt[0].lower()
            tag = wt[1]
            transition[previous + ' ' + tag] += 1   # count the transition
            context[tag] += 1                       # count the context
            emit[tag + ' ' + word] += 1             # count the emission
            previous = tag
        transition[previous + ' </s>'] += 1
    
    with open(model_file, 'w') as model:
        for key, val_trans in transition.items():
            previous = key.split(' ')[0]
            model.write('T ' + key + ' ' + str(round(val_trans/context[previous],6)) + '\n')
        for key, val_emit in emit.items():
            previous = key.split(' ')[0]
            model.write('E ' + key + ' ' + str(round(val_emit/context[previous],6)) + '\n')


if __name__ == "__main__":
    train_f = open(sys.argv[1])
    train_hmm(train_f,sys.argv[2])

# python3 train-hmm.py 05-train-input.txt model_file.txt