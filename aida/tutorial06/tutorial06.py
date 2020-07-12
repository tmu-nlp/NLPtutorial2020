import random
from math import exp
from collections import defaultdict

def sign(score):
    if score >= 0:
        return 1
    else:
        return -1

class SupportVectorMachine():
    def __init__(self):
        """
        :param w: defaultdict, weight for each feature
        :param margin: minimum distance between boundary and case
        :param c: normalize parameter
        :param last: dict, use for normalize
        :param is_train: bool, training or test
        """
        self.w = defaultdict(lambda: 0)
        self.margin = 10
        self.c = 0.0001
        self.last = {}
        self.is_train = True

    def create_features(self, sentence):
        """ obtain unigram features

        :return phi: defaultdict, count features
        """
        phi = defaultdict(lambda: 0)
        words = sentence.split()
        for word in words:
            phi['UNI:{}'.format(word)] += 1
        return phi

    def update_weights(self, phi, y_ans):
        """ update weights """
        for feature, value in phi.items():
            self.w[feature] += value * y_ans
        return self
    
    def getw(self, feature, iter):
        """ obtain normalized weight

        :param feature: feature name (unigram)
        :param iter: training iteration
        :return self.w[feature]: 
        """
        last_iter = self.last.get(feature, 0)
        if iter != last_iter:
            c_size = self.c*(iter-last_iter)
            if abs(self.w[feature]) < c_size:
                self.w[feature] = 0
            else:
                self.w[feature] -= sign(self.w[feature]) * c_size
            self.last[feature] = iter
        return self.w[feature]

    def predict_one(self, phi, iter, f=sign):
        """ predict one case

        :param w:
        :param phi:
        :param score:
        :function f: compute probability
        :return y_hat: prediction 1 or -1
        """
        score = 0
        if self.is_train:
            for feature, value in phi.items():
                score += value * self.getw(feature, iter)
            y_hat = f(score)
            return y_hat, score
        else:
            for feature, value in phi.items():
                score += value * self.w[feature]
            y_hat = f(score)
            return y_hat

    def fit(self, file_path, max_epoch, seed):
        """ training all dataset

        :param file_path: a path of training dataset
        :param max_epoch: int, max epoch
        :param seed: int, random seed
        :param prev_w: defaultdict, weights before normalization
        """
        random.seed(seed)
        with open(file_path) as fp:
            lines = fp.readlines()
        for _ in range(max_epoch):
            random.shuffle(lines)
            for iter, line in enumerate(lines):
                y_ans, sentence = line.strip().split('\t')
                phi = self.create_features(sentence)
                prev_w = self.w
                _, score = self.predict_one(phi, iter)
                y_ans = int(y_ans)
                value = score * y_ans
                if value <= self.margin:
                    self.update_weights(phi, y_ans)
                else:
                    self.w = prev_w

    def predict(self, input_file):
        """ predict all sentences

        :return predictions: predicted labels
        """
        predictions = []
        self.is_train = False
        with open(input_file) as fp:
            for line in fp:
                sentence = line.strip()
                phi = self.create_features(sentence)
                y_hat = self.predict_one(phi, iter=None)
                predictions.append(y_hat)
        return predictions

if __name__ == '__main__':
    train_file = './../data/titles-en-train.labeled'
    svm = SupportVectorMachine()
    svm.fit(train_file, max_epoch=10, seed=1)
    with open('./weights.txt', 'w') as fp:
        for feature, value in svm.w.items():
            fp.write('{}\t{}\n'.format(feature, value))

    test_file = './../data/titles-en-test.word'
    predictions = svm.predict(test_file)
    with open('./my-answer.txt', 'w') as fp:
        for y in predictions:
            fp.write(f'{y}\n')

