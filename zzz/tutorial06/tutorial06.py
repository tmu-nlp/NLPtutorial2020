'''
Result:
Normalization Parameter, Accuracy
1.0 51.54%
0.1 51.54%
0.01 51.54%
0.001 80.02%
0.0001 88.34%
0.0001 88.59%
'''
import numpy as np
from math import exp

from zzz.tutorial05.tutorial05 import FeatureFunctions

PATH = '/Users/zz_zhang/勉強会/NLPチュートリアル/NLPtutorial2020/data/'
TRAIN_FILE = 'titles-en-train.labeled'
TEST_FILE = 'titles-en-test.labeled'


class SupportVectorClassifier(object):
    def __init__(self, learning_rate=0.01, margin=0.1, normalization_para=0.1, create_features=lambda x: x):
        self.learning_rate = learning_rate
        self.learning_rate_para = 1 / learning_rate
        self.margin = margin
        self.create_features = create_features
        self.norma_para = normalization_para
        self.weight = None

    def fit(self, x, y, batch_size=64, epoch=1):
        self.weight = np.zeros(self.create_features(x[0]).shape)
        for _ in range(epoch):
            print('Start to train in epoch {0}.'.format(_))
            for (index, (data, label)) in enumerate(zip(x, y)):
                phi = self.create_features(data)
                val = self.weight.dot(phi) * label
                self.learning_rate = 1.0 / (self.learning_rate_para + index)
                if val <= self.margin:
                    self.update(phi, label)

                if index % 100 == 0:
                    print('{0:.2f}% trained.'.format(float(index) * 100 / len(x)))
            print('100% trained.')

    def predict_one(self, phi):
        return self.sigmoid(self.weight.dot(phi))

    def predict(self, x):
        res = map(self.predict_one, map(self.create_features, x))
        res = map(lambda v: -1 if v < 0.5 else 1, res)
        return res

    def score(self, x, y):
        res = self.predict(x)
        correct = 0
        for (y1, y2) in zip(y, res):
            if y1 == y2:
                correct += 1
        return float(correct) / len(y)

    def sigmoid(self, a):
        return 1.0 / (1 + exp(-a))

    def update(self, phi, label):
        self.weight = np.array(
            list(map(lambda v: 0.0 if abs(v) < self.norma_para else v * (1 - self.norma_para), self.weight)))

        diff = self.differential_sigmoid(phi, label)
        self.weight += self.learning_rate * diff

    def differential_sigmoid(self, phi, label):
        res = phi * (exp(self.weight.dot(phi)) / (1 + exp(self.weight.dot(phi))) ** 2)
        return res * label


if __name__ == '__main__':
    # p = Perceptron((2, 3))
    # print(p.weight)
    with open(PATH + TRAIN_FILE) as file:
        text_train = []
        labels_train = []
        for line in file:
            label, *sentence = line.split()
            labels_train.append(int(label))
            text_train.append(sentence)
        # print(labels, text)

        f = FeatureFunctions(text_train)

        shape = (f.shape,)
        print(shape)
        svc = SupportVectorClassifier(learning_rate=.1, margin=0.3, normalization_para=0.00001,
                                      create_features=f.unigram_transform)

        svc.fit(text_train, labels_train)

    with open(PATH + TEST_FILE) as file:
        text_test = []
        labels_test = []

        for line in file:
            label, *sentence = line.split()
            labels_test.append(int(label))
            text_test.append(sentence)

        labels_predict = svc.predict(text_test)
        # print(svc.score(text_test, labels_test))
        with open('tutorial06.res', 'w') as output_file:
            for (x, y) in zip(labels_predict, text_test):
                output_file.write(str(x) + '\t' + ' '.join(y) + '\n')
