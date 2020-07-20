import numpy as np
from train_rnn import create_one_hot, forward_rnn
import dill

if __name__ == '__main__':
    test_file = open('wiki-en-test.norm')
    # test_file = '05-test-input.txt'

    # Load model
    net = dill.load(open('network_weight_2', 'rb'))
    word_ids = dill.load(open('word2ids_2', 'rb'))
    tag_ids = dill.load(open('tag2ids_2', 'rb'))

    text = []
    for line in test_file:
        words = []
        line = line.rstrip()
        for word in line.split(' '):
            if word in word_ids:
                words.append(create_one_hot(word_ids[word], len(word_ids)))
            else:
                words.append(np.zeros(len(word_ids)))

        h, p, tags_predict = forward_rnn(net, words)

        lines = []
        for tag in tags_predict:
            for key, value in tag_ids.items():
                if value == tag:
                    lines.append(key)
        text.append(' '.join(lines))

    pred_out = open('my_answer.pos','w')
    for sent in text:
        pred_out.writelines(sent + '\n')

# perl gradepos.pl wiki-en-test.pos my_answer.pos
# Accuracy: 87.90% (4011/4563)

# Most common mistakes:
# JJ --> NN	95
# NNS --> NN	76
# NNP --> NN	60
# VBN --> NN	36
# VBP --> NN	30
# VBG --> NN	25
# RB --> NN	25
# IN --> WDT	17
# VB --> NN	17
# VBZ --> NN	14
