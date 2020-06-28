from collections import defaultdict

class Perceptron():
    def __init__(self):
        """
        :param w: defaultdict, weight for each feature
        """
        self.w = defaultdict(lambda: 0)

    def create_features(self, sentence):
        """ obtain unigram features
        :param phi: defaultdict, count features
        """
        phi = defaultdict(lambda: 0)
        words = sentence.split()
        for word in words:
            phi['UNI:{}'.format(word)] += 1
        return phi

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
            score += value * self.w[feature]
        if score >= 0:
            y_hat = 1
        else:
            y_hat = -1
        return y_hat

    def train_one(self, sentence, ans):
        """ training one sentence
        """
        phi = self.create_features(sentence)
        y_hat = self.predict_one(phi)
        if ans != y_hat:
            self.update_weights(phi, ans)

    def fit(self, file_path, n_iter):
        """ training all dataset
        :param file_path: a path of training dataset
        :param n_iter: int, iteration
        """
        for _ in range(n_iter):
            with open(file_path) as fp:
                for line in fp:
                    ans, sentence = line.strip().split('\t')
                    self.train_one(sentence, int(ans))

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
    with open('./weights.txt', 'w') as fp:
        for feature, value in perceptron.w.items():
            fp.write('{}\t{}\n'.format(feature, value))
    test_file = './../data/titles-en-test.word'
    perceptron.predict(test_file)

