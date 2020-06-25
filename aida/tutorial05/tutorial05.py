from collections import defaultdict
from pprint import pprint

class Perceptron():
    def __init__(self):
        """
        :param w: dict, weight for each feature
        """
        self.w = {}

    def create_features(self, sentence):
        """ obtain unigram features
        :param phi: defaultdict
        """
        phi = defaultdict(lambda: 0)
        words = sentence.split()
        for word in words:
            phi['UNI:{}'.format(word)] += 1
        return phi

    def initialize_weights(self, phi):
        """ initialize weights
        """
        for feature in phi.keys():
            if feature not in self.w:
                self.w[feature] = 0
        return self
    
    def update_weights(self, phi, ans):
        """ update weights
        """
        for feature, value in phi.items():
            self.w[feature] += value * ans
        return self

    def predict_one(self, phi):
        """ prediction
        :param w:
        :param phi:
        :param score:

        :return y_hat: prediction 1 or -1
        """
        score = 0
        for feature, value in phi.items():
            if feature in self.w:
                score += value * self.w[feature]
        if score >= 0:
            y_hat = 1
        else:
            y_hat = -1
        return y_hat

    def train_one(self, sentence, ans, n_iter):
        """ online training
        """
        for n in range(n_iter):
            phi = self.create_features(sentence)
            self.initialize_weights(phi)
            y_hat = self.predict_one(phi)
            if ans != y_hat:
                self.update_weights(phi, ans)

    def fit(self, file_path, n_iter):
        """ training all dataset
        :param file_path: a path of training dataset
        :param n_iter: int, iteration
        """
        with open(file_path) as fp:
            for line in fp:
                ans, sentence = line.strip().split('\t')
                #self.train_online(sentence, ans, n_iter)
                self.train_one(sentence, int(ans), n_iter)

    def predict(self, input_file):
        """ predict all sentences
        """
        with open(input_file) as fp:
            for line in fp:
                sentence = line.strip()
                phi = self.create_features(sentence)
                y_hat = self.predict_one(phi)
                print(y_hat)

if __name__ == '__main__':
    #train_file = './../test/03-train-input.txt'
    train_file = './../data/titles-en-train.labeled'
    perceptron = Perceptron()
    perceptron.fit(train_file, n_iter=10)
    #pprint(perceptron.w)
    test_file = './../data/titles-en-test.word'
    perceptron.predict(test_file)


