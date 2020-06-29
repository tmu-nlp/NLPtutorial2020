# python test_perceptron.py ../data/titles-en-train.word
# python test_perceptron.py ../data/titles-en-test.word
# python ../script/grade-prediction.py ../data/titles-en-test.labeled my_answer
import sys
import pickle
from train_perceptron import create_features, predict_one

def predict_all(model_file, input_file):
    with open(model_file, mode='rb') as model_f \
        , open(input_file) as in_f:
        weights = pickle.load(model_f)
        #入力ファイルを一行ずつsplit
        for line in in_f:
            phi = create_features(line)
            label_pred = predict_one(weights, phi)
            print(label_pred)

def main():
    model_fname = './model/weights.pickle'
    input_fname = sys.argv[1]
    predict_all(model_fname, input_fname)

if __name__ == '__main__':
    main()

'''result
Accuracy = 80.623450%
'''