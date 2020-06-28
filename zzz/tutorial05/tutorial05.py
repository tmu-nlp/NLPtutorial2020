import numpy as np
from collections import OrderedDict

PATH = '/Users/zz_zhang/勉強会/NLPチュートリアル/NLPtutorial2020/data/'
TRAIN_FILE = 'titles-en-train.labeled'
TEST_FILE = '03-train-answer'

class Perceptron:
    def __init__(self, shape=None):
        if shape is None:
            return -1
        else:
            self.shape = shape
        # self.weight = np.random.random_sample(self.shape)
        self.weight = np.zeros(self.shape)

    def fit(self, x, y, feature_function=lambda x: x):

        for (xx, yy) in zip(x, y):
            feature_x = feature_function(xx)
            y_hat = self.predict_one(feature_x)
            if y_hat != yy:
                self.update_weight(feature_x, yy)

    def predict_one(self, x):
        # return self.sign(np.multiply(x, self.weight))
        return self.sign(x.dot(self.weight))


    def predict(self, x, feature_function=lambda x: x):
        x_feature = feature_function(np.array(x))
        y = []
        for xx in x_feature:
            y.append(self.predict_one(xx))
        return np.array(y)

    def update_weight(self, x, y):
        self.weight = self.weight + y * x

    def sign(self, v):
        return 1 if v >= 0 else -1


class FeatureFunctions:
    def __init__(self, text=None):
        self.vocab = {}
        self.shape = None
        if text is not None:
            self.fit(text)
        else:
            return -1

    def fit(self, text: list):
        for line in text:
            for word in line:
                if ('UNI:' + word) not in self.vocab.keys():
                    self.vocab['UNI:' + word] = len(self.vocab) + 1
        self.vocab['<UNK>'] = 0
        self.shape = len(self.vocab)

    def unigram_transform(self, sentence):
        phi = [0 for _ in range(len(self.vocab))]
        for word in sentence:
            if ('UNI:' + word) in self.vocab.keys():
                phi[self.vocab['UNI:' + word]] += 1
            else:
                phi[self.vocab['<UNK>']] += 1
        return np.array(phi)

    # def unigram(self, text):
    #     if isinstance(text, list):
    #         return self.unigram_transform(text)
    #     elif isinstance(text, str):
    #         line_list = [line for line in text.split('\n')]
    #         return self.unigram_transform(line_list)


if __name__ == '__main__':
    # p = Perceptron((2, 3))
    # print(p.weight)
    with open(PATH + TRAIN_FILE) as file:
        text = []
        labels = []
        for line in file:
            label, *sentence = line.split()
            labels.append(int(label))
            text.append(sentence)
        # print(labels, text)

        f = FeatureFunctions(text)
        print(f.vocab)
        # print(f.unigram_transform(text[0]))
        # print(f.shape)
        shape = (f.shape,)

        p = Perceptron(shape)
        p.fit(text, labels, feature_function=f.unigram_transform)
        print(p.weight)