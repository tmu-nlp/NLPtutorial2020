import sys
sys.path.append("../")
import numpy as np
from common.layers import TimeEmbedding, TimeRNN, TimeAffine, TimeSoftmaxWithLoss
from common.functions import softmax
from common.optimizer import SGD

class RNNLM:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn
        embed_W = (rn(V, D) / 100).astype('f')
        rnn_Wx = (rn(D, H) / np.sqrt(D)).astype('f')
        rnn_Wh = (rn(H, H) / np.sqrt(H)).astype('f')
        rnn_b = np.zeros(H).astype('f')
        affine_W = (rn(H, V) / np.sqrt(H)).astype('f')
        affine_b = np.zeros(V).astype('f')
        self.layers = [
            TimeEmbedding(embed_W),
            TimeRNN(rnn_Wx, rnn_Wh, rnn_b, stateful=True),
            TimeAffine(affine_W, affine_b)
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.rnn_layer = self.layers[1]
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, X):
        for layer in self.layers:
             X = layer.forward(X)
        return X

    def forward(self, X, y):
        X = self.predict(X)
        loss = self.loss_layer.forward(X, y)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def reset_state(self):
        self.rnn_layer.reset_state()


def load_data(path):
    with open(path, "r") as f:
        data = f.readlines()
    X, y = [], []
    for line in data:
        X_i, y_i = [], []
        for word_pos in line.strip().split():
            word, pos = word_pos.split("_")
            X_i.append(word)
            y_i.append(pos)
        X.append(X_i)
        y.append(y_i)
    return X, y

def get_cntvec(vectorizer, X):
    X = vectorizer.transform(list(map(lambda x: " ".join(x), X)))
    return X.toarray()

def get_enc(cat, y):
    y_new = []
    for y_i in y:
        y_i_new = []
        for y_ij in y_i:
            y_ij_new = [0] * cat
            y_ij_new[y_ij-1] = 1
            y_i_new.append(y_ij_new)
        y_new.append(y_i_new)
    return y_new

def get_le(vectorizer, y):
    y = [[vectorizer.get(y_i_j, 0) if vectorizer.get(y_i_j, 0) == 0 else vectorizer.get(y_i_j)+1 for y_i_j in y_i] for y_i in y]
    return y

def zero_padding(X):
    max_len = 0
    for i in X:
        if max_len < len(i):
            max_len = len(i)
    for i in range(len(X)):
        while (len(X[i])) < max_len:
            # X[i] = np.append(X[i], -1)
            X[i].append(0)
    return np.array(X)


def train_model(model, optimizer, X, y, batch_size=32, lr=0.01, max_epoch=100):
    max_iters = len(X) // batch_size
    time_idx = 0
    total_loss = 0
    loss_count = 0
    data_size = len(X)
    for epoch in range(max_epoch):
        idx = np.random.permutation(np.arange(data_size))
        X = X[idx]
        y = y[idx]
        for iters in range(max_iters):
            batch_X = X[iters*batch_size:(iters+1)*batch_size]            
            batch_y = y[iters*batch_size:(iters+1)*batch_size]
            loss = model.forward(batch_X, batch_y)
            model.backward()
            optimizer.update(model.params, model.grads)
            total_loss += loss
            loss_count += 1
        avg_loss = total_loss / loss_count
        print("epoch {}, loss {:.3f}".format(epoch, avg_loss))
        total_loss, loss_count = 0, 0

def batch_pred(test_X, batch_size=32):
    iters = len(test_X) // batch_size
    pred_ys = []
    for i in range(iters):
        pred_y = model.predict(test_X[i*batch_size:(i+1)*batch_size])
        pred_y = pred_y.reshape(pred_y.shape[0]*pred_y.shape[1], -1)
        pred_y = softmax(pred_y).argmax(axis=1)
        pred_ys += list(pred_y)
    # if batch_size*iters < len(test_X):
    #     pred_y = model.predict(test_X[iters*batch_size:])
    #     pred_y = pred_y.reshape(pred_y.shape[0]*pred_y.shape[1], -1)
    #     pred_y = softmax(pred_y).argmax(axis=1)
    #     pred_ys += list(pred_y)
    return pred_ys
        
if __name__ == "__main__":
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    import itertools

    train = "../../data/wiki-en-train.norm_pos"
    test = "../../data/wiki-en-test.norm_pos"
    # load_data(train)
    train_X, train_y = load_data(train)
    test_X, test_y = load_data(test)
    # preprocess X
    # cv = CountVectorizer()
    # cv.fit(list(map(lambda x: " ".join(x), train_X)))
    le = LabelEncoder()
    le.fit(list(itertools.chain.from_iterable(train_X)))
    le = dict(zip(le.classes_, le.transform(le.classes_)))
    train_X, test_X = zero_padding(get_le(le, train_X)), zero_padding(get_le(le, test_X))
    # preprocess y    
    le = LabelEncoder()
    le.fit(list(itertools.chain.from_iterable(train_y)))
    le = dict(zip(le.classes_, le.transform(le.classes_)))
    train_y, test_y = zero_padding(get_le(le, train_y)), zero_padding(get_le(le, test_y))
    model = RNNLM(vocab_size=len(set(itertools.chain.from_iterable(train_X)))+1, wordvec_size=100, hidden_size=100)
    optimizer = SGD(lr=0.01)
    train_model(model, optimizer, train_X, train_y, max_epoch=30)
    # prediction
    # pred_y = model.predict(test_X[:32])
    # pred_y = pred_y.reshape(pred_y.shape[0]*pred_y.shape[1], -1)
    # test_y = test_y[:32]
    # test_y = test_y.reshape(test_y.shape[0]*test_y.shape[1])
    # pred_y = softmax(pred_y).argmax(axis=1)
    # print(test_y)
    # print(pred_y)
    pred_y = batch_pred(test_X)
    test_y = test_y.reshape(test_y.shape[0]*test_y.shape[1])[:len(pred_y)]
    print(pred_y)
    print(test_y)
    print(accuracy_score(test_y, pred_y))