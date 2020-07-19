import numpy as np
from scipy.special import softmax
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

class RNN():
    def __init__(self, learning_rate, word_to_id, label_to_id, embed_size, hidden_size, output_size):
        self.lr = learning_rate
        self.word_to_id = word_to_id
        self.label_to_id = label_to_id

        self.embed_size = embed_size 
        self.hidden_size = hidden_size
        self.weight_h = np.random.normal(size=[embed_size+hidden_size, hidden_size]) / np.sqrt(hidden_size)
        self.b_h = np.random.normal(size=hidden_size) / np.sqrt(hidden_size)
        self.weight_o = np.random.normal(size=[hidden_size, output_size]) / np.sqrt(output_size)
        self.b_o = np.random.normal(size=output_size) / np.sqrt(output_size)

    def forward(self, inputs):
        """ predict """
        self._inputs = []
        self._hiddens = []
        self._outputs = []
        hidden = self.init_hidden()
        for x in inputs:
            x = np.concatenate((x, hidden), 0)
            hidden = np.tanh(x@self.weight_h + self.b_h + 1) 
            output = softmax(hidden@self.weight_o + self.b_o + 1)
            self._inputs.append(x)
            self._hiddens.append(hidden)
            self._outputs.append(output)
        return self._outputs
    
    def __call__(self, inputs):
        outputs = self.forward(inputs)
        return outputs

    def init_hidden(self):
        return np.random.rand(self.hidden_size)

    def backward(self, y_train):
        """ compute backprop from outputs """
        self._delta_w_h = np.zeros(self.weight_h.shape)
        self._delta_w_o = np.zeros(self.weight_o.shape)
        self._delta_b_h = np.zeros(self.b_h.shape)
        self._delta_b_o = np.zeros(self.b_o.shape)
        delta_h = np.zeros(self.hidden_size)
        for i in range(1, len(y_train)+1):
            t = len(y_train) - i
            delta_o = y_train[t] - self._outputs[t] # output_size
            self._delta_w_o += np.outer(self._hiddens[t], delta_o) # hidden_size*output_size
            self._delta_b_o += delta_o # output_size
            d = (np.dot(delta_h, self.weight_h[self.embed_size:, :]) + np.dot(self.weight_o, delta_o.T)) # hidden_size
            delta_h = d * (1 - self._hiddens[t]**2) # hidden_size
            self._delta_w_h += np.outer(self._inputs[t], delta_h) # (embed_size+hidden_size)*hidden_size
            self._delta_b_h += delta_h # hidden_size
        return self
    
    def update_weights(self, lr):
        """ update weights """
        self.weight_h += lr * self._delta_w_h
        self.weight_o += lr * self._delta_w_o
        self.b_h += lr * self._delta_b_h
        self.b_o += lr * self._delta_b_o
        return self

    def fit(self, sentences_train, labels_train, max_epoch):
        """ train model """
        lr_decay = self.lr / max_epoch
        lr_temp = self.lr
        losses = []
        for t in tqdm(range(max_epoch)):
            print(f'{t}-th epoch')
            if t > 0:
                lr_temp -= lr_decay
            shuffled_ids = np.random.permutation(len(labels_train))
            loss = 0
            for i in shuffled_ids:
                X_train = obtain_onehot(sentences_train[i], self.word_to_id)
                y_train = obtain_onehot(labels_train[i], self.label_to_id)
                y_hat = self.forward(X_train)
                #print(f'  hat: {y_hat}\n  ans: {y_train}')
                loss += criterion(y_hat, y_train)
                self.backward(y_train)
                self.update_weights(lr_temp)
            ave_loss = loss / len(shuffled_ids)
            print(f'  loss: {ave_loss}')
            losses.append(ave_loss)
        plot_loss(losses)
        return
    
    def predict(self, sentences_test):
        predictions = []
        for sent in sentences_test:
            y_test = obtain_onehot(sent, self.word_to_id)
            pred = self.forward(y_test)
            predictions.append(pred)
        return predictions


def preprocess(file_names):
    """ obtain X_train, y_train, word_to_id before define model

    :param file_names: list, files[train, test]
    :param is_train: bool
    :param word_to_id: dict, convert word to id
    :param label_to_id: dict, convert label to id
    :param label: list, labels converted to id
    :param words: list, words converted to id in each sentence
    :return: sentences[len(lines), len(line_each)], labels[len(lines), len(line_each)], word_to_id, label_to_id
    """
    UNK = '<UNK>'
    word_to_id = defaultdict(lambda: len(word_to_id))
    label_to_id = defaultdict(lambda: len(label_to_id))
    word_to_id[UNK] = 0
    file_train = file_names[0]
    with open(file_train) as fp:
        sentences_train = []
        labels_train = []
        for line in fp:
            word_labels = line.strip().split()
            words = []
            labels_each = []
            for word_label in word_labels:
                word, label = word_label.split('_')
                words.append(word_to_id[word])
                labels_each.append(label_to_id[label])
            sentences_train.append(words)
            labels_train.append(labels_each)
    file_test = file_names[1]
    with open(file_test) as fp:
        sentences_test = []
        for line in fp:
            sentence = []
            words = line.strip().split()
            for word in words:
                if word in word_to_id.keys():
                    sentence.append(word_to_id[word])
                else:
                    sentence.append(word_to_id['<UNK>'])
            sentences_test.append(sentence)
    return sentences_train, labels_train, sentences_test, word_to_id, label_to_id

def create_id2label(label_to_id):
    id_to_label = {}
    for label, id in label_to_id.items():
        id_to_label[id] = label
    return id_to_label

def obtain_onehot(sentence, token_to_id):
    """ create onehot vector from one sentence

    :return: onehot_vecs 
    """
    onehot_vecs = np.zeros([len(sentence), len(token_to_id)])
    for i, word_id in enumerate(sentence):
        onehot_vecs[i][word_id] += 1
    return onehot_vecs

def criterion(y_hat, y_ans):
    """ compute loss
    
    :param y_hat: numpy.array, predict label
    :param y_ans: numpy.array, answer label
    :return: loss
    """
    loss = 0
    for hat, ans in zip(y_hat, y_ans):
        loss += 1/2 * np.linalg.norm(hat - ans)**2
    return loss

def plot_loss(losses):
    """ plot train losses """
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(range(len(losses)), losses)
    plt.savefig('./tutorial08/loss.png')

def decode(pred_sent, id_to_label):
    """ convert id to label in prediction

    :param pred_sent: predictions in one sentnece
    :param id_to_label: dict
    :return: labels
    """
    labels = [id_to_label[np.argmax(pred)] for pred in pred_sent]
    return labels

if __name__ == '__main__':
    #file_train = './test/05-train-input.txt'
    #file_test = './test/05-test-input.txt'
    file_train = './data/wiki-en-train.norm_pos'
    file_test = './data/wiki-en-test.norm'
    sentences_train, labels_train, sentences_test, word_to_id, label_to_id = preprocess([file_train, file_test])
    id_to_label = create_id2label(label_to_id)

    # define params
    learning_rate = 0.001
    embed_size = len(word_to_id)
    hidden_size = 50
    output_size = len(label_to_id)
    max_epoch = 20

    rnn = RNN(learning_rate, word_to_id, label_to_id, embed_size, hidden_size, output_size)
    rnn.fit(sentences_train, labels_train, max_epoch)
    predictions = rnn.predict(sentences_test)

    with open('./tutorial08/my-answer.pos', 'w') as fp:
        for pred_sent in predictions:
            labels = decode(pred_sent, id_to_label)
            fp.write(f'{" ".join(labels)}\n')

