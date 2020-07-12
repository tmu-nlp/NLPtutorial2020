from math import exp
import numpy as np
from collections import defaultdict
from tqdm import tqdm

def sign(score):
    if score >= 0:
        return 1
    else:
        return -1

def preprocess(train_file):
    """ obtain X_train, y_train, word_to_id before define model

    :param label: label(-1, 1)
    :param phi: frequency of each feature
    :return: X_train[len(lines), len(word_to_id)], y_train[len(lines)], word_to_id
    """
    word_to_id = defaultdict(lambda: len(word_to_id))
    with open(train_file) as fp:
        features = []
        labels = []
        for line in fp:
            label, sentence = line.strip().split('\t')
            phi = defaultdict(lambda: 0)
            words = sentence.split()
            for word in words:
                phi[word_to_id[f'UNI:{word}']] += 1
            features.append(phi)
            labels.append(int(label))
    print(len(word_to_id))
    X_train = np.zeros([len(labels), len(word_to_id)])
    y_train = np.zeros([len(labels)])
    for i in range(len(labels)):
        y_train[i] = labels[i]
        for j, v in features[i].items():
            X_train[i][j] = v

    return X_train, y_train, word_to_id

class MultiLayerPerceptron():
    def __init__(self, lr, input_size, output_size, word_to_id):
        """
        :param is_train: bool, training or test
        """
        self.lr = lr
        self.word_to_id = word_to_id
        self.win = (np.random.rand(input_size, len(word_to_id)) - 0.5) / 5
        self.wout = (np.random.rand(output_size, input_size) - 0.5) / 5
        self.bin = (np.random.rand(input_size) - 0.5) / 5
        self.bout = (np.random.rand(output_size) - 0.5) / 5
        self.is_train = True

    def forward(self, phi):
        """ inference
        :param phi1: output from input layer
        :return: output from output layer
        """
        phi1 = np.tanh(self.win@phi+self.bin)
        phi2 = np.tanh(self.wout@phi1+self.bout)
        return phi2

    def backward(self, y_ans, y_hat, phi):
        """ backprop """
        delta = np.array([y_ans - y_hat])
        delta_out = delta * (1 - y_hat**2)
        delta_in = delta_out@self.wout * (1 - np.tanh(self.win@phi+self.bin)**2)
        return delta_out, delta_in

    def update_weights(self, delta_out, delta_in, phi):
        """ update weights """
        self.wout += self.lr * np.outer(delta_out, np.tanh(self.win@phi+self.bin))
        self.win += self.lr * np.outer(delta_in, phi)
        self.bout += self.lr * delta_out.reshape(-1)
        self.bin += self.lr * delta_in.reshape(-1)
        return self
    
    def create_features(self, sentence):
        """ obtain unigram features

        :return phi: defaultdict, count features
        """
        phi = np.zeros([len(self.word_to_id)])
        words = sentence.split()
        for word in words:
            feature = f'UNI:{word}'
            if feature in self.word_to_id:
                phi[self.word_to_id[feature]] += 1
        return phi

    def fit(self, X_train, y_train, max_epoch):
        """ training all dataset

        :param X_train: features [len(lines), len(word_to_id)]
        :param y_train: labels [len(lines)]
        :param max_epoch: int, max epoch
        """
        lr_decay = self.lr / max_epoch
        for epoch in tqdm(range(max_epoch)):
            if epoch > 0:
                self.lr -= lr_decay
            N = np.random.permutation(y_train.shape[0])
            for i in N:
                phi = X_train[i]
                y_ans = y_train[i]
                y_hat = self.forward(phi)
                delta_out, delta_in = self.backward(y_ans, y_hat, phi)
                self.update_weights(delta_out, delta_in, phi)

    def predict(self, test_file):
        """ predict all sentences

        :return predictions: predicted labels
        """
        predictions = []
        self.is_train = False
        with open(test_file) as fp:
            for line in fp:
                sentence = line.strip()
                phi = self.create_features(sentence)
                y_hat = sign(self.forward(phi))
                predictions.append(y_hat)
        return predictions

if __name__ == '__main__':
    np.random.seed(1)
    train_file = './../data/titles-en-train.labeled'
    X_train, y_train, word_to_id = preprocess(train_file)
    mlp = MultiLayerPerceptron(lr=0.01, input_size=1, output_size=1, word_to_id=word_to_id)
    mlp.fit(X_train, y_train, max_epoch=10)
    test_file = './../data/titles-en-test.word'
    predictions = mlp.predict(test_file)
    with open(f'./my-answer.txt', 'w') as fp:
        for y in predictions:
            fp.write(f'{y}\n')

"""
lr=0.01, input=1, max_epoch=10
acc: 93.94%

lr=0.1, input=1, max_epoch=10
acc: 94.08%

lr=1.0, input=1, max_epoch=10
acc: 91.04%

lr=1.0, input=2, max_epoch=10
acc: 89.30%

lr=1.0, input=4, max_epoch=10
acc: 85.44%

lr=1.0, input=8, max_epoch=10
acc: 52.32%
"""

