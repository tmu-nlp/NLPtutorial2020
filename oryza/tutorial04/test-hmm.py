import sys
import math
from collections import defaultdict

def load_model(model_file):
    transition = defaultdict(lambda:0)
    emission = defaultdict(lambda:0)
    possible_tags = defaultdict(lambda:0)

    for line in model_file:
        line = line.strip().split(' ')
        type = line[0]
        context = line[1]
        word = line[2]
        prob = line[3]

        possible_tags[context] = 1
        if type == 'T':
            transition[context + ' ' + word] = float(prob)
        else:
            emission[word + ' ' + context] = float(prob)
    
    return transition, emission, possible_tags

def fwd_bwd_steps(text, transition, emission, possible_tags):
    sentence = []
    for line in text:
        lambda_1 = 0.95
        V = 1000000
        words = line.lower().strip().split()
        words.append('</s>')
        l = len(words)

        best_score = defaultdict(lambda: 10 ** 10)
        best_edge = defaultdict(str)
        best_score['0 <s>'] = 0 # Start with <s>
        best_edge['0 <s>'] = None
        for i in range(l):
            for prev in possible_tags.keys():
                for next in possible_tags.keys():
                    if best_score[str(i) + ' ' + prev] is not 10 ** 10 and transition[prev + ' ' + next] is not 0:
                        score = best_score[str(i) + ' ' + prev] + -math.log(transition[prev + ' ' + next]) + -math.log((lambda_1 * emission[words[i] + ' ' + next]) + ((1 - lambda_1) / V))

                        if best_score[str(i + 1) + ' ' + next] > score:
                            best_score[str(i + 1) + ' ' + next] = score
                            best_edge[str(i + 1) + ' ' + next] = str(i) + ' ' + prev
                
                next = '</s>'
                if best_score[str(i) + ' ' + prev] is not 10 ** 10 and transition[prev + ' ' + next] is not 0:
                    score = best_score[str(i) + ' ' + prev] + -math.log(transition[prev + ' ' + next])
                    
                    if best_score[str(i + 1) + ' ' + next] > score:
                        best_score[str(i + 1) + ' ' + next] = score
                        best_edge[str(i + 1) + ' ' + next] = str(i) + ' ' + prev
    
        tags = []
        next_edge = best_edge[str(l) + ' </s>']
        while next_edge != '0 <s>': 
            tag = next_edge.split(' ')[1]
            tags.append(tag)
            next_edge = best_edge[next_edge]
        tags.reverse()        
        sent = ' '.join(tags)
        sentence.append(sent)
    return sentence

if __name__ == "__main__":
    model_f = open(sys.argv[2])
    trans, emis, psb_tags = load_model(model_f)
    # input = open('05-test-input.txt')
    input = open(sys.argv[1])

    with open('my_answer.pos','w') as fout:
        text = fwd_bwd_steps(input, trans, emis, psb_tags)
        for line in text:
            fout.write(line + '\n')

'''
python3 test-hmm.py wiki-en-test.norm model_file_pos.txt
perl gradepos.pl wiki-en-test.pos my_answer.pos

Accuracy: 90.86% (4146/4563)

Most common mistakes:
NNS --> NN	44
NNP --> NN	29
NN --> JJ	27
JJ --> DT	18
NNP --> JJ	15
JJ --> NN	12
VBN --> NN	11
VBN --> JJ	10
NN --> IN	9
JJ --> VBN	7
'''    