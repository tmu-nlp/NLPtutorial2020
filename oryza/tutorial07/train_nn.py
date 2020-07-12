from collections import defaultdict
import numpy as np
import joblib

def create_features(features, ids):
    phi = np.zeros(len(ids))
    words = features.split(' ')
    for word in words:
        if 'UNI:' + word not in ids:
            continue
        phi[ids['UNI:' + word]] += 1 # We add "UNI:" to indicate unigrams
    return phi

def init_network(feature_size, node, layer):
    nn = []
    w_0 = 2 * np.random.rand(node, feature_size) - 1
    b_0 = np.random.rand(1, node)
    nn.append((w_0, b_0))

    while len(nn) < layer:
        w = 2 * np.random.rand(node, node) - 1
        b = np.random.rand(1, node)
        nn.append((w, b))

    w_output = 2 * np.random.rand(1, node) - 1
    b_output = np.random.rand(1, 1)
    nn.append((w_output, b_output))

    return nn

def forward_nn(nn, phi_0):
    phi = [0 for _ in range(len(nn) + 1)]
    phi[0] = phi_0

    for i in range(len(nn)):
        w, b = nn[i]
        phi[i + 1] = np.tanh(np.dot(w, phi[i]) + b).T
    return phi

def backward_nn(nn, phi, y_):
    J = len(nn)
    d = np.zeros(J+1, dtype=np.ndarray)
    d[-1] = np.array([y_ - phi[J][0]])
    d_ = np.zeros(J+1, dtype=np.ndarray)

    for i in range(J, 0, -1):
        d_[i] = d[i] * (1 - phi[i]**2).T
        (w, b) = nn[i - 1]
        d[i - 1] = np.dot(d_[i], w)
    
    return d_

def update_weights(nn, phi, d_, lr):
    for i in range(len(nn)):
        w, b = nn[i]
        w += lr * np.outer(d_[i+1], phi[i])
        b += lr * d_[i+1]

def trainNN(text, num_node, num_layer, num_epoch, lr):
    feat_lab = []
    ids = defaultdict(lambda: len(ids))

    for line in text:
        line = line.strip().split('\t')
        feature = line[1]
        label = int(line[0])
        feat_lab.append((create_features(feature, ids), label))

    network = init_network(len(ids), num_node, num_layer)
    joblib.dump(dict(ids), open(f'ids.pkl', 'wb'))

    for e in range(num_epoch):
        for (phi_0, y) in feat_lab:
            phi = forward_nn(network, phi_0)
            d = backward_nn(network, phi, y)
            update_weights(network, phi, d, lr)
    
    joblib.dump(network, open(f'network_model', 'wb'))

if __name__ == "__main__":
    text = open('titles-en-train.labeled')
    # text = open('03-train-input.txt')

    # trainNN(text, num_node, num_layer, num_epoch, lr)
    trainNN(text, 2, 1, 1, 0.1)