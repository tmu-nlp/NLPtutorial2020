from collections import defaultdict
import numpy as np
import pickle


def predict_one(net, phi_0):
    phi = [0 for _ in range(len(net) + 1)]
    phi[0] = phi_0
    for i in range(len(net)):
        w, b = net[i]
        phi[i + 1] = np.tanh(np.dot(w, phi[i]) + b).T
    score = phi[len(net)][0]
    return 1 if score >= 0 else -1


def create_features_test(sentence, ids):
    # testのときはidsに含まれるwordのみカウントする
    phi = np.zeros(len(ids))
    words = sentence.split()
    for word in words:
        if "UNI:" + word not in ids:
            continue
        phi[ids["UNI:" + word]] += 1
    return phi

np.random.seed(seed=0)
with open("net", "rb") as net_file, open("ids", "rb") as ids_file:
    net = pickle.load(net_file)
    ids = pickle.load(ids_file)

with open("titles-en-test.word", "r", encoding="utf-8") as test_file, open(
    "my_result.txt", "w", encoding="utf-8"
) as ans_file:
    for line in test_file:
        phi = create_features_test(line.strip(), ids)
        predict = predict_one(net, phi)
        ans_file.write(str(predict) + "\t" + line)

# epoch = 1
# Accuracy = 52.320227% (seed = 0)
# Accuracy = 83.244775% (seed = 1)

# epoch = 5
# Accuracy = 91.073326% (seed = 0)
# Accuracy = 91.037903% (seed = 1)
