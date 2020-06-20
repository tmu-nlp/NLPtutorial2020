from collections import Counter
from collections import defaultdict
import re
import math
import numpy as np

PATH = '/Users/zz_zhang/勉強会/NLPチュートリアル/NLPtutorial2020/test/'
TRAIN_FILENAME = '02-train-input.txt'
TEST_FILENAME = '02-train-input.txt'
TRAIN_ANSWER_FILENAME = '02-train-answer.txt'
MODEL_FILENAME = 'tutorial02.model'
V = 1000000
N = 2


class Ngram:
    def __init__(self, n=2):
        self.n = n
        self.gram_counter = Counter()                                                       #(w_i|w_(i-1), w(i-2)) -> str: 'w_i w_(i-1) w_(i-2)'
        self.context_counter = Counter()
        self.prob = defaultdict(lambda: 0)
        self.p = defaultdict(lambda: 0)

    def fit(self, text: str):
        for line in text.split('\n'):
            if len(line) > 0:
                words = ['<s>'] + line.replace('\n', '').split(' ') + ['<\\s>']
                for n in range(self.n, 0, -1):
                    for index in range(max(1, n - 1), len(words)):
                        self.gram_counter[' '.join(words[index - n + 1: index + 1])] += 1       # (i-n+1) ~ i
                        self.context_counter[' '.join(words[index - n + 1: index])] += 1        # (i-n+1) ~ (i-1)

        for (gram, num) in self.gram_counter.items():
            self.prob[gram] = float(num) / self.context_counter[' '.join(gram.split(' ')[:-1])]

    def fit_tag(self, text, tags):
        gram_counter = Counter()
        tag_counter = Counter()
        for (line, line_tag) in zip(text, tags):
            for (word, tag) in zip(line, line_tag):
                gram = word + ' ' + tag
                gram_counter[gram] += 1
                tag_counter[tag] += 1

        for (gram, num) in gram_counter.items():
            self.prob[gram] = float(num) / tag_counter[gram.split(' ')[-1]]

    def train(self, text: str):
        self.fit(text)

    def save(self, filename: str):
        with open(filename, 'w') as file:
            file.write(str(self.n) + '\n')
            for (gram, prob) in self.prob.items():
                temp = gram + '\t' + str(prob) + '\n'
                file.write(temp)

    def load(self, filename: str):
        with open(filename, 'r') as file:
            for index, line in enumerate(file):
                if index == 0:
                    parameter = line.replace('\n', '')
                    self.n = int(parameter)
                else:
                    temp = line.split('\t')
                    self.prob[temp[0]] = float(temp[-1])

    def test(self, text: str, smoothing='linear', linear_lambda: list = []):
        self.p = defaultdict(lambda: linear_lambda[0] / V)
        if smoothing == 'linear':
            entropy = self.linear_smoothing(text, linear_lambda)

        elif smoothing == 'witten_bell':
            entropy = self.witten_bell_smoothing(text)
        else:
            return None
        return entropy

    def linear_smoothing(self, text: str, linear_lambda: list = []):
        '''
        :param text: input text
        :param linear_lambda: [l1, l2, ..., ln] where sum(linear_lambda) == 1
        :return: entropy
        '''
        if abs(1 - sum(linear_lambda)) > 1e-5:
            # print('Lambda error.')
            return -1

        entropy = 0.0
        total_length = 0.0

        lamb = linear_lambda
        for line in text.split('\n'):
            if len(line) > 0:
                words = ['<s>'] * (self.n - 1) + line.split(' ') + ['<\\s>']
                for index in range(self.n - 1, len(words)):
                    word = words[index]
                    self.p[word] = lamb[1] * self.prob[word] + lamb[0] / V
                    for n in range(2, self.n + 1):
                        gram = ' '.join(words[index - n + 1: index + 1])
                        self.p[gram] = lamb[n] * self.prob[gram] + self.p[' '.join(gram.split(' ')[1:])]
                    entropy -= math.log(self.p[word], 2)
                total_length += len(words)
        return entropy / total_length

    def witten_bell_smoothing(self, text: str, lamb_uni: float=0.95):
        '''
        :param text: input_text
        :param lamb_uni: lambda for P_ML(w_i)
        :return: entropy
        '''
        lamb = {}
        entropy = 0.0
        total_length = 0.0
        text_stoken = ''                                    # text with token '<s>' and '<\s>'
        for line in text.split('\n'):
            text_stoken = text_stoken + '<s> ' + line + ' <\\s> '

        # calculate lambda[w]
        for word in text_stoken.split(' '):
            for n in range(1, self.n):
                pattern = r'' + word + ' .+? ' * (n)
                context = re.findall(pattern, text_stoken)
                context = list(map(lambda s: s[:-1], context))
                # print(context)
                u = len(set(context))
                c = text_stoken.split(' ').count(word)
                lamb[(word, n)] = 1 - (float(u) / (u + c))

        # calculate entropy
        for line in text.split('\n'):
            words = ['<s>'] * (self.n - 1) + line.split(' ') + ['<\\s>']
            for index in range(max(1, self.n - 1), len(words)):
                word = words[index]
                self.p[word] = lamb_uni * self.prob[word] + (1 - lamb_uni) / V
                for n in range(2, self.n + 1):
                    gram = ' '.join(words[index - n + 1: index + 1])
                    self.p[gram] = lamb[(word, n - 1)] * self.prob[gram] + (1 - lamb[(word, n - 1)]) * self.p[' '.join(gram.split(' ')[1:])]
                entropy -= math.log(self.p[word], 2)
            total_length += len(words) - 1

        return entropy / total_length


if __name__ == '__main__':
    bigram = Ngram(N)
    # Train
    with open(PATH + TRAIN_FILENAME) as file:
        text = ''.join([line for line in file])
        if text[-1] == '\n':
            text = text[:-1]
        bigram.train(text)
        bigram.save(MODEL_FILENAME)

    # Test
    bigram.load(MODEL_FILENAME)
    with open(PATH + TEST_FILENAME) as file:
        text = ''.join([line for line in file])
        if text[-1] == '\n':
            text = text[:-1]
        bigram.test(text, 'linear')
        # linear smoothing
        # --------------------
        lambdas = []

        def lambda_generator(lambdas_local, num, total_sum=1.0, step=20):
            if num == 1:
                lambdas.append(lambdas_local + [1 - sum(lambdas_local)])
                return
            for l in np.linspace(0.0, 1.0, step):
                if total_sum - l >= 0:
                    lambda_generator(lambdas_local + [l],
                                     num - 1,
                                     total_sum - l,
                                     step)
                else:
                    continue

        lambda_generator([], N + 1, 1.0, 20)

        for lamb in lambdas:
            entropy = bigram.test(text, 'linear', lamb)
            if entropy != -1:
                print(lamb, entropy)
        # --------------------

        # witten_bell smoothing
        # --------------------
        entropy = bigram.test(text, 'witten_bell')
        print(entropy)
        # --------------------
