#python train_perceptron.py ../test/03-train-input.txt
#python train_perceptron.py ../data/titles-en-train.labeled
from collections import defaultdict
import sys
import pickle
MAX_ITER = 10
#ラベルを予測
def predict_one(w, phi):
    score = 0
    for name, value in phi.items():
        if name in w:
            score += value*w[name]
        else: w[name] = 0
    if score > 0: return 1
    else: return -1

#一文中で出現した各単語の頻度をまとめた辞書(key:name, value:freqency)
def create_features(x):
    phi = defaultdict(int)
    words = x.split()
    for word in words:
        phi[f'UNI:{word}'] += 1
    return phi

#重み更新
def update_weights(w, phi, y):
    for name, value in phi.items():
        w[name] += value*y
    return w

#学習
def train(train_file, iterations):
    weights = {}
    for _ in range(iterations):
        for line in train_file:
            label_true, sentence = line.split('\t')
            #一文中で出現した各単語の頻度をまとめた辞書(key:name, value:freqency)
            phi = create_features(sentence)
            #ラベルを予測
            label_pred = predict_one(weights, phi)
            if label_true != label_pred:
                weights = update_weights(weights, phi, int(label_true))
    return weights

def main():
    with open(sys.argv[1]) as train_f \
        , open('./model/weights.pickle', mode='wb') as weight_f:
        weights = train(train_file=train_f, iterations=MAX_ITER)
        pickle.dump(weights, weight_f)
    
    '''
    for name, value in sorted(weights.items(), key=lambda x:x[0]):
        print(f"{name}\t{value}")
    '''

if __name__ == '__main__':
    main()