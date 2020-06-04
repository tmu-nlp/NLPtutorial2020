import math
from collections import defaultdict

from tqdm import tqdm
from pprint import pprint

class bigramLM():
    def __init__(self):
        self.freqs = defaultdict(lambda: 0)
        self.probs = defaultdict(lambda: 0)
        self.total_freqs = 0

    def train(self, file_path):
        """ train bigram language model
        
        :param unigram_to_freq: defaultdict, [word]: frequency
        :param bigram_to_freq: defaultdict, [word_i, word_i+1]: frequency

        :return bigram_to_prob: dict, [word_i, word_i+1]: probability
        """

        with open(file_path) as fp:
            for line in fp:
                words = line.split()
                # begin of sentence token
                words.insert(0, '<s>')
                words.append('</s>')
                #print(words)
                for i in range(1, len(words)):
                    word = words[i-1]
                    bigram = f'{words[i-1]} {words[i]}'
                    #print(bigram)
                    self.freqs[word] += 1
                    self.freqs[bigram] += 1
                    self.total_freqs += 1

        for ngram, freq in self.freqs.items():
            words = ngram.split(' ')
            if len(words) == 1:
                # unigram
                self.probs[ngram] = freq / self.total_freqs
            elif len(words) == 2:
                # bigram
                context_word = words[0]
                context_freq = self.freqs[context_word]
                self.probs[ngram] = freq / context_freq
            else:
                print('error')

        return self

    def test(self, file_path, lambda_1, lambda_2, vocab_size=1000000):
        """ calculate entropy

        :param lambda_1: float, smoothing parameter of unigram probability
        :param lambda_2: float, smoothing parameter of unigram probability
        :param vocab_size: int, size of vocabulary

        :param W: int, total number of words in test data
        :param H: float, sum of the log-likelihoods

        :return entropy:
        """

        W = 0
        H = 0

        with open(file_path) as fp:
            for line in fp:
                words = line.split()
                words.insert(0, '<s>')
                words.append('</s>')
                for i in range(1, len(words)):
                    tgt_word = words[i-1]
                    bigram = f'{words[i-1]} {words[i]}'

                    # unigram prob
                    prob = (1 - lambda_1) / vocab_size
                    if tgt_word in self.probs:
                        prob += lambda_1 * self.probs[tgt_word]
                    # bigram prob
                    prob *= (1 - lambda_2)
                    if bigram in self.probs:
                        prob += lambda_2 * self.probs[bigram]

                    H -= math.log(prob, 2)
                    W += 1

            # compute entropy
            entropy = H / W

        return entropy

    def grid_search(self, file_path):
        """ grid search lambda_1 and lambda_2

        :param lambda_1: float, smoothing parameter of unigram probability
        :param lambda_2: float, smoothing parameter of bigram probability

        :return lambda_1_best, lambda_2_best, entropy_best:
        """
        lambda_1_best = None
        lambda_2_best = None
        entropy_best = float('inf')
        
        print('grid search...')
        for lambda_1 in tqdm(range(5, 100, 5)):
            for lambda_2 in range(5, 100, 5):
                lambda_1 /= 100
                lambda_2 /= 100
                entropy = self.test(file_path, lambda_1=lambda_1, lambda_2=lambda_2)
                if entropy < entropy_best:
                    lambda_1_best = lambda_1
                    lambda_2_best = lambda_2
                    entropy_best = entropy

        return lambda_1_best, lambda_2_best, entropy_best


if __name__ == '__main__':
    train_file_path = './data/wiki-en-train.word'
    test_file_path = './data/wiki-en-test.word'
    biLM = bigramLM()
    biLM.train(train_file_path)
    #pprint(biLM.freqs)
    #pprint(biLM.probs)
    lambda_1_best, lambda_2_best, entropy_best = biLM.grid_search(test_file_path)
    print('lambda_1: {}\n lambda_2: {}\n entropy: {:.2f}'.format(lambda_1_best, lambda_2_best, entropy_best))

