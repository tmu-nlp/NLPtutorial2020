from collections import defaultdict
from train_nn import create_features
import numpy as np
import joblib


def predict_one(network, phi_0):
    phi = [0 for _ in range(len(network) + 1)]
    phi[0] = phi_0
    for i in range(len(network)):
        w, b = network[i]
        phi[i + 1] = np.tanh(np.dot(w, phi[i]) + b).T
    score = phi[len(network)][0]
    return 1 if score >= 0 else -1

def predict_all(model_file, input_file, ids_file):
    network_model = joblib.load(model_file)
    ids = joblib.load(ids_file)
    
    res = []
    for line in input_file:
        line = line.strip()
        phi = create_features(line, ids)
        y_predict = predict_one(network_model, phi)
        res.append(str(y_predict) + '\t' + line + '\n')

    return res

if __name__ == "__main__":
    input = open('titles-en-test.word')

    with open('my_answer','w') as fout:
        prediction = predict_all('network_model', input, 'ids.pkl')
        
        for line in prediction:
            fout.write(line)

# python3 grade-prediction.py titles-en-test.labeled my_answer
# layer = 1, lr = 0.1, epoch = 1, acc = 52.320227%