import numpy as np

PATH = '/Users/zz_zhang/勉強会/NLPチュートリアル/NLPtutorial2020/test/'
TRAIN_FILE = '03-train-input.txt'
TEST_FILE = ''


class DenseLayer(object):
    def __init__(self, in_dim, out_dim, activation_function=None, name=''):
        self.weight = np.random.rand(in_dim, out_dim) - 0.5
        self.bias = np.random.rand(1)[0]
        self.name = ''
        self.act = activation_function

    def update(self, weight, bias):
        self.weight = weight
        self.bias = bias


class NeuralNetwork(object):

    def __init__(self, learning_rate):
        self.layers = []
        self.learning_rate = learning_rate

    def add_layer(self, in_dim, out_dim, activation_function=None, name='layer_{}'):
        if len(self.layers) == 0:
            self.layers.append(DenseLayer(in_dim + 1, out_dim, activation_function, name.format(len(self.layers))))
        else:
            self.layers.append(DenseLayer(in_dim, out_dim, activation_function, name.format(len(self.layers))))

    def fit(self, x, y):
        for (data, label) in zip(x, y):
            data = np.array([1] + list(data))
            y_hat = self.predict_one(data)
            self.update(y_hat, label)

    def update(self, y_hat, y):
        delta_prime = self.backward(y_hat, y)
        for (index, layer) in enumerate(self.layers):
            diff_w = np.outer(delta_prime[index + 1], layer.weight).T
            diff_b = delta_prime[index + 1].T

            layer.update(self.learning_rate * np.array(diff_w) + layer.weight,
                         self.learning_rate * diff_b + layer.bias)

    def backward(self, y_hat, y):
        delta = [0] * (len(self.layers)) + [y - y_hat[0]]
        delta_prime = [0] * (len(self.layers) + 1)

        for index in range(len(self.layers) - 1, -1, -1):
            delta_prime[index + 1] = delta[index + 1] * (1 - delta[index + 1] ** 2)
            w, b = self.layers[index].weight, self.layers[index].bias
            d = np.dot(w, delta_prime[index + 1])
            delta_prime[index] = d
        return np.array(delta_prime)

    def predict_one(self, x):
        temp = x
        for layer in self.layers:
            temp = np.dot(temp, layer.weight) + layer.bias
            temp = layer.act(temp)
        return temp

    def predict(self, x):
        res = []
        for data in x:
            data = np.array([1] + list(data))
            res.append(self.predict_one(data))
        return res

class BagOfWord(object):
    def __init__(self):
        self.d = {'<UNK>': 0}
        self.size = 1

    def fit(self, text):
        for sentence in text:
            for word in sentence:
                if word not in self.d:
                    self.d[word] = len(self.d)
                    self.size += 1

    def fit_transform(self, text):
        self.fit(text)
        return self.transform(text)

    def transform(self, text):
        res = []
        for sentence in text:
            temp_sent = [0] * len(self.d)
            for word in sentence:
                if word in self.d:
                    temp_sent[self.d[word]] += 1
                else:
                    temp_sent[self.d['<UNK>']] += 1
            res.append(temp_sent)
        return np.array(res)

if __name__ == '__main__':
    with open(PATH + TRAIN_FILE) as file:
        text = []
        cate = []
        for line in file:
            c, t = line.split('\t')
            cate.append(int(c))
            text.append(t.split(' '))
        bow = BagOfWord()
        text = bow.fit_transform(text)
        # print(text)
        model = NeuralNetwork(learning_rate=0.1)
        model.add_layer(bow.size, 1)
        model.fit(text, cate)