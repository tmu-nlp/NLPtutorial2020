from collections import defaultdict
import numpy as np
import pickle

# debug大変だった
# ndarrayを指定する
# 時々Transposeの必要あり

def create_features(sentence, ids):
    phi = np.zeros(len(ids))
    words = sentence.split()
    for word in words:
        phi[ids["UNI:" + word]] += 1
    return phi


def init_network(feature_size, node, layer):
    # Networkの初期化（ランダム）
    # np.random.rand(n, m): 0.0以上, 1.0未満のｍ要素のリストｘnのリスト

    net = []

    # 1つ目の隠れ層
    w0 = 2 * np.random.rand(node, feature_size) - 0.5  # 重みは-0.5以上0.5未満で初期化
    b0 = np.random.rand(1, node)
    net.append((w0, b0))

    # 中間層
    while len(net) < layer:
        w = 2 * np.random.rand(node, node) - 0.5
        b = np.random.rand(1, node)
        net.append((w, b))

    # 出力層
    w_o = 2 * np.random.rand(1, node) - 0.5
    b_o = np.random.rand(1, 1)

    net.append((w_o, b_o))

    return net


def forward_nn(net, phi_0):
    phi = [0 for _ in range(len(net) + 1)]   # phiはリスト, phi_0はnp.zeros
    phi[0] = phi_0
    for i in range(len(net)):
        w, b = net[i]
        phi[i + 1] = np.tanh(np.dot(w, phi[i]) + b).T 
    # Transposeしない --> ValueError: shapes (1,2) and (1,2) not aligned: 2 (dim 1) != 1 (dim 0)
    # print(phi)
    return phi


def backward_nn(net, phi, label):
    j = len(net)
    delta = np.zeros(j + 1, dtype=np.ndarray)   # dtype=np.ndarray を指定しないと動かない
    delta[-1] = np.array([label - phi[j][0]])
    delta_prime = np.zeros(j + 1, dtype=np.ndarray)
    for i in range(j, 0, -1):
        delta_prime[i] = delta[i] * (1 - phi[i] ** 2).T  
        w, _ = net[i - 1]
        delta[i - 1] = np.dot(delta_prime[i], w)
    return delta_prime


def update_weights(net, phi, delta_prime, eta):
    for i in range(len(net)):
        w, b = net[i]
        w += eta * np.outer(delta_prime[i + 1], phi[i])
        b += eta * delta_prime[i + 1]


###train###
np.random.seed(seed=0)
ids = defaultdict(lambda: len(ids))
feat_label = []


# 先にidsのkeyを用意しておいてあげないと動かない
with open("titles-en-train.labeled", "r", encoding="utf-8") as train_file:
    for line in train_file:
        label, sentence = line.strip().split("\t")
        for word in sentence.split():
            ids["UNI:" + word]

with open("titles-en-train.labeled", "r", encoding="utf-8") as train_file:
    for line in train_file:
        label, sentence = line.strip().split("\t")
        label = int(label)
        phi = create_features(sentence, ids)
        feat_label.append((phi, label))

net = init_network(len(ids), 2, 1)
#print(feat_label)
for _ in range(5):
    for phi_0, label in feat_label:
        phi = forward_nn(net, phi_0)
        delta_prime = backward_nn(net, phi, label)
        update_weights(net, phi, delta_prime, 0.1)
# print(net)
# print(ids)
with open("net", "wb") as net_file, open("ids", "wb") as ids_file:
    pickle.dump(net, net_file)
    pickle.dump(dict(ids), ids_file)
