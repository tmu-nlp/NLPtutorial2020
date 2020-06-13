import sys
from collections import defaultdict
import math

def load_model(model_file):
    model = open(model_file)
    probabilities = defaultdict(lambda:0)

    for line in model:
        words = line.split()
        w = words[0]
        P = words[1]
        probabilities[w] = float(P)
    return probabilities

def word_segment(input_file,model_file):
    model_prob = load_model(model_file)
    lambda_1 = 0.95
    V = 1000000

    sentence = []
    input = open(input_file, 'r', encoding='utf-8')
    for line in input:
        best_edge = defaultdict()
        best_score = defaultdict()
        best_edge[0] = None
        best_score[0] = 0
        for word_end in range(1,len(line)):
            best_score[word_end] = 10 ** 10
            for word_begin in range(len(line)):
                word = line[word_begin:word_end]
                if word in model_prob.keys() or len(word) == 1:
                    P = lambda_1 * model_prob[word] + (1 - lambda_1) / V
                    my_score = best_score[word_begin] + -math.log(P)
                    if my_score < best_score[word_end]:
                        best_score[word_end] = my_score
                        best_edge[word_end] = [word_begin,word_end]
        
        words = []
        next_edge = best_edge[len(best_edge) - 1]
        while next_edge is not None:
            word = line[next_edge[0]:next_edge[1]]
            words.append(word)
            next_edge = best_edge[next_edge[0]]
        words.reverse()
        sent = ' '.join(words)
        sentence.append(sent)
    return sentence



if __name__ == "__main__":
    with open('my_answer.word','w') as fout:
        text = word_segment(sys.argv[1],sys.argv[2])
        for line in text:
            fout.write(line + '\n')


'''
python3 word-segmentation.py wiki-ja-test.txt model-ja-04.txt
perl gradews.pl wiki-ja-test.word my_answer.word

Sent Accuracy: 23.81% (20/84)
Word Prec: 71.87% (1942/2702)
Word Rec: 84.22% (1942/2306)
F-meas: 77.56%
Bound Accuracy: 86.29% (2783/3225)
'''
