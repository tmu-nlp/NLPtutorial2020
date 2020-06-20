import sys
from collections import defaultdict
from math import log

from pprint import pprint

class HiddenMarkovModel():
    def __init__(self):
        """ Hidden Markov Model (HMM)
        :param self.transition: dict, [prevtag_nexttag]: count
        :param self.context: dict, [prevtag]: count
        :param self.emit: dict, [nexttag_nextword]: count
        :param self.prob_trans: dict, [prevtag_nexttag]: probability
        :param self.prob_emit: dict, [nexttag_nextword]: probability
        """
        self.transition = defaultdict(lambda: 0)
        self.context = defaultdict(lambda: 0)
        self.emit = defaultdict(lambda: 0)
        self.prob_trans = {}
        self.prob_emit = {}

    def train(self, file_path):
        """ train Hidden Markov Model (HMM)
        :param file_path: the path of training data
        :param possible_tags: list(self), candidates of tags

        :return self: trained HMM
        """
        # count words and tags
        with open(file_path) as fp:
            for line in fp:
                previous = '<s>'
                self.context[previous] += 1
                word_tags = line.strip().split(' ')
                for word_tag in word_tags:
                    word = word_tag.split('_')[0]
                    tag = word_tag.split('_')[1]
                    self.transition[f'{previous} {tag}'] += 1
                    self.context[tag] += 1
                    self.emit[f'{tag} {word}'] += 1
                    previous = tag
                self.transition[f'{previous} </s>'] += 1
    
        # compute probability
        for previous_tag, count in self.transition.items():
            previous = previous_tag.split(' ')[0]
            self.prob_trans[previous_tag] = count / self.context[previous]
        for tag_word, count in self.emit.items():
            tag = tag_word.split(' ')[0]
            self.prob_emit[tag_word] = count / self.context[tag]
        # obtain possible tags 
        self.possible_tags = [tag for tag in self.context.keys()]
        return self

    def forward(self, words):
        """ viterbi forward
        :param unknown_rate: smoothing parameter
        :param N: vocabulary size

        :return best_score: dict, [id_tag]: score. lower is better.
        :return best_edge: dict,  [nextid_nexttag]: previd_prevtag
        :return endid_eos: str, len(words)+1_</s> for backward
        """
        unknown_rate = 0.05
        N = 1000000
        best_score = {}
        best_edge = {}
        best_score['0 <s>'] = 0
        best_edge['0 <s>'] = None
        for i in range(len(words)):
            for prev in self.possible_tags:
                for next in self.possible_tags:
                    previd_prevtag = f'{i} {prev}'
                    prevtag_nexttag = f'{prev} {next}'
                    nexttag_prevword = f'{next} {words[i]}'

                    if previd_prevtag in best_score and prevtag_nexttag in self.transition:
                        score = best_score[previd_prevtag] - log(self.prob_trans[prevtag_nexttag], 2)
                        prob_e = unknown_rate / N
                        if nexttag_prevword in self.prob_emit:
                            prob_e += (1 - unknown_rate) * self.prob_emit[nexttag_prevword]
                        score += -log(prob_e, 2)

                        nextid_nexttag = f'{i+1} {next}'
                        if nextid_nexttag not in best_score or score < best_score[nextid_nexttag]:
                            best_score[nextid_nexttag] = score
                            best_edge[nextid_nexttag] = previd_prevtag
        #consider EOS, </s>
        for tag in self.possible_tags:
            secondendid_tag = f'{len(words)} {tag}'
            tag_eos = f'{tag} </s>'
            if secondendid_tag in best_score and tag_eos in self.prob_trans:
                score = best_score[secondendid_tag] - log(self.prob_trans[tag_eos], 2)
                endid_eos = f'{len(words)+1} </s>'
                if endid_eos not in best_score or score < best_score[endid_eos]:
                    best_score[endid_eos] = score
                    best_edge[endid_eos] = secondendid_tag

        return best_score, best_edge, endid_eos

    def backward(self, best_score, best_edge, endid_eos):
        """ viterbi backword
        :param best_score: dict, best scores[id_tag]: score
        :param best_edge: dict, best edge[nextid_nexttag]: previd_prevtag

        :return join(tags): predicted tags
        """
        tags = []
        next_edge = best_edge[endid_eos]
        while next_edge != '0 <s>':
            position, tag = next_edge.split()
            tags.append(tag)
            next_edge = best_edge[next_edge]
        tags = tags[::-1]
        print(' '.join(tags))

    def test(self, file_path):
        """ estimate tags
        :param file_path: the path of test data
        
        :return tags(from 'backward'): estimated tags for each words
        """
        with open(file_path) as fp:
            for line in fp:
                words = line.strip().split()
                #print(words)
                best_score, best_edge, endid_eos = self.forward(words)
                self.backward(best_score, best_edge, endid_eos)
        return

if __name__ == '__main__':
    #train_file_path = '../test/05-train-input.txt'
    train_file_path = '../data/wiki-en-train.norm_pos'
    hmm = HiddenMarkovModel()
    hmm.train(file_path=train_file_path)
    #pprint(HMM.transition)
    #pprint(HMM.emit)
    #pprint(HMM.context)
    #pprint(HMM.prob_trans)
    #pprint(HMM.prob_emit)
    #pprint(HMM.possible_tags)
    #test_file_path = '../test/05-test-input.txt'
    test_file_path = '../data/wiki-en-test.norm'
    hmm.test(file_path=test_file_path)

