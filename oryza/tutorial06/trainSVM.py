from collections import defaultdict
import numpy as np
import re

def create_features(features):
    phi = defaultdict(lambda: 0)
    words = features.split(' ')
    for word in words:
        phi['UNI:' + word] += 1 # We add "UNI:" to indicate unigrams
    return phi

def calculate_val(w, phi, y):
    val = 0
    for name, value in phi.items():
        if name in w:
            val += value * w[name] * int(y)
    return val

def update_weights(w, phi, y, c):
    for name, value in w.items():
        if abs(value) < c:
            w[name] = 0
        else:
            w[name] -= np.sign(value) * c

    for name, value in phi.items():
        w[name] += int(value) * int(y)

def train_svm(text):
    c = 0.0001
    margin = 10
    w = defaultdict(lambda: 0)
    
    for line in text:
        line = line.strip().split('\t')
        feature = line[1]
        label = int(line[0])
        phi = create_features(feature)
        val = calculate_val(w, phi, label)

        if val <= margin:
            update_weights(w, phi, label, c)

    with open('model_titles_en','w') as model_out:
        for name, value in w.items():
            model_out.write(name + '\t' + str(round(value,6)) + '\n')

if __name__ == "__main__":
    text = open('titles-en-train.labeled')
    train_svm(text)