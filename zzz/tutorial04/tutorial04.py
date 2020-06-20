from collections import defaultdict
import math
from zzz.tutorial02.tutorial02 import Ngram

PATH = '/Users/zz_zhang/勉強会/NLPチュートリアル/NLPtutorial2020/test/'
TRAIN_FILENAME = '05-train-input.txt'
TEST_FILENAME = '05-test-input.txt'
MODEL_NAME = 'tutorial05.model'

N = 100000

class HMM:
    def __init__(self):
        self.tag_bigram = Ngram(2)
        self.word_tag_bigram = Ngram(2)

    def fit(self, text, tags):
        self.tag_bigram.fit('\n'.join([' '.join(line) for line in tags]))
        self.word_tag_bigram.fit_tag(text, tags)

        # for (gram, prob) in self.tag_bigram.prob.items():
        #     print(gram, prob)
        # print()
        # for (gram, prob) in self.word_tag_bigram.prob.items():
        #     print(gram, prob)

    def save_model(self, filename):
        with open(filename, 'w') as file:
            file.write('T\n')  # Start writing Transformation probability (P(tag2|tag1))
            for (gram, prob) in self.tag_bigram.prob.items():
                if ' ' in gram:
                    file.write(gram + ' ' + str(prob) + '\n')

            file.write('E\n')  # Start writing generation probability (P(word|tag))
            for (gram, prob) in self.word_tag_bigram.prob.items():
                file.write(gram + ' ' + str(prob) + '\n')

    def load_model(self, filename):
        with open(filename) as file:
            cache = ''.join([line for line in file])
            mode = None
            for line in cache:
                if line == 'T\n':
                    mode = 'T'
                elif line == 'E\n':
                    mode = 'E'
                else:
                    if mode == 'T':
                        gram1, gram2, prob = line.split(' ')[:-1]   # estimate '\n'
                        self.tag_bigram.prob[gram1 + ' ' + gram2] = float(prob)
                    elif mode == 'E':
                        gram1, gram2, prob = line.split(' ')[:-1]
                        self.word_tag_bigram[gram1 + ' ' + gram2] = float(prob)

    def predict(self, text: str, linear_lambda = [0.05, 0.95]):
        result = []
        cache = filter(lambda x: x != '', text.split('\n'))
        tags = list(set([gram.split(' ')[-1] for (gram, _) in self.word_tag_bigram.prob.items()]))
        best_score = defaultdict(lambda: 0)
        for line in cache:
            words = ['<s>'] + line.split(' ') + ['<\\s>']
            route = defaultdict(lambda :'')
            for index in range(1, len(words)):
                word = words[index]
                pre_word = words[index - 1]

                if pre_word == '<s>':
                    for tag in tags:
                        pt = self.tag_bigram.prob['<s> ' + tag] * linear_lambda[1] + linear_lambda[0] / len(tags)
                        pe = self.word_tag_bigram.prob[word + ' ' + tag] * linear_lambda[1] + linear_lambda[0] / N
                        best_score[str(index) + ' ' + tag] = -math.log(pt, 2) - math.log(pe, 2)
                        route[str(index) + ' ' + tag] = '<s> ' + tag
                elif word != '<\\s>':
                    for tag in tags:
                        best_score[str(index) + ' ' + tag] = 0x3fffffff
                        pe = self.word_tag_bigram.prob[word + ' ' + tag] * linear_lambda[1] + linear_lambda[0] / N
                        for pre_tag in tags:
                            pt = self.tag_bigram.prob[pre_tag + ' ' + tag] * linear_lambda[1] + linear_lambda[0] / len(
                                tags)
                            if best_score[str(index) + ' ' + tag] > best_score[str(index - 1) + ' ' + pre_tag] - math.log(pt, 2) - math.log(pe, 2):
                                best_score[str(index) + ' ' + tag] = best_score[str(index - 1) + ' ' + pre_tag] - math.log(pt, 2) - math.log(pe, 2)
                                route[str(index) + ' ' + tag] = route[str(index - 1) + ' ' + pre_tag] + ' ' + tag
                else:
                    best_score[str(index) + ' ' + word] = 0x3fffffff
                    pt = self.tag_bigram.prob[tag + '<\\s>'] * linear_lambda[1] + linear_lambda[0] / len(tags)

                    for tag in tags:
                        if best_score[str(index) + ' ' + word] > best_score[str(index - 1) + ' ' + tag] - math.log(pt, 2):
                            best_score[str(index) + ' ' + word] = best_score[str(index - 1) + ' ' + tag] - math.log(pt, 2)
                            route[str(index) + ' ' + word] = route[str(index - 1) + ' ' + tag] + ' ' + word

            # print(route[str(len(words) - 1) + ' ' + '<\\s>'])
            res = route[str(len(words) - 1) + ' ' + '<\\s>']
            result.append(res.replace('<s> ', '').replace(' <\\s>', ''))
        return result

def split_word_tag(text: str):
    words = []
    tags = []
    for line in text.split('\n'):
        if len(line) > 0:
            word = list(map(lambda x: x.split('_')[0], line.split(' ')))
            tag = list(map(lambda x: x.split('_')[1], line.split(' ')))
            words.append(word)
            tags.append(tag)
    return words, tags


if __name__ == '__main__':
    hmm = HMM()
    with open(PATH + TRAIN_FILENAME) as file:
        text = ''.join([line for line in file])
        words, tags = split_word_tag(text)


        hmm.fit(words, tags)
        # for (gram, prob) in hmm.tag_bigram.prob.items():
        #     print(gram, prob)
        # for (gram, prob) in hmm.word_tag_bigram.prob.items():
        #     print(gram, prob)
        hmm.save_model(MODEL_NAME)

        # print()
        hmm.load_model(MODEL_NAME)
        # for (gram, prob) in hmm.tag_bigram.prob.items():
        #     print(gram, prob)
        # for (gram, prob) in hmm.word_tag_bigram.prob.items():
        #     print(gram, prob)
    with open(PATH + TEST_FILENAME) as file:
        text = ''.join([line for line in file])
        tags = hmm.predict(text)
        print(tags)
