import math
import pickle

from tqdm import tqdm
from pprint import pprint

def load_dic(model_path):
    fp = open(model_path, 'rb')
    word_prob = pickle.load(fp)
    return word_prob

def forward(line, word_to_prob):
    unk_rate = 0.05
    N = 1000000
    best_edge = [None] * (len(line)+1)
    best_score = [0] * (len(line)+1)
    for word_end in range(1, len(line)+1):
        best_score[word_end] = 10000000000
        for word_begin in range(word_end):
            word = line[word_begin:word_end]
            if word in word_to_prob or len(word) == 1:
                if len(word) == 1:
                    prob = unk_rate / N
                if word in word_to_prob:
                    prob = (1 - unk_rate) * word_to_prob[word] + unk_rate / N
                my_score = best_score[word_begin] - math.log2(prob)
                if my_score < best_score[word_end]:
                    best_score[word_end] = my_score
                    best_edge[word_end] = [word_begin, word_end]
    #pprint(best_score)
    #pprint(best_edge)
    return best_score, best_edge

def backward(line, best_edge):
    words = []
    next_edge = best_edge[-1]
    while next_edge != None:
        #print(next_edge)
        word = line[next_edge[0]:next_edge[1]]
        next_edge = best_edge[next_edge[0]]
        words.append(word)
    words.reverse()
    return words

def main(file_path, model_path):
    """
    """
    word_to_prob = load_dic(model_path)
    with open(file_path) as fp:
        for line in fp:
            line = line.strip()
            #print('target line: {}'.format(line))
            best_score, best_edge = forward(line, word_to_prob)
            words = backward(line, best_edge)
            #print('segmented: ', end='')
            print(' '.join(words))

if __name__ == '__main__':
    file_path = './data/wiki-ja-test.txt'
    #file_path = './test/04-input.txt'
    model_path = './tutorial03/04-data_word_to_prob.pkl'
    #model_path = './tutorial03/04-test_word_to_prob.pkl'
    main(file_path, model_path)
    
