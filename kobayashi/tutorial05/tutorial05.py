#python tutorial05.py ../../test/03-train-input.txt train_result.txt test_input.txt test_result.txt
#python tutorial05.py ../../data/titles-en-train.labeled train_result2.txt ../../data/titles-en-test.word test_result2.txt

import sys
import math
from collections import defaultdict

class Perceptron:

    def __init__(self):
        self.word_weight_dict = defaultdict(lambda:0)
    
    def predict_one(self, phi):
        score = 0
        for name, value in sorted(phi.items()):
            if name in self.word_weight_dict:
                score += value * self.word_weight_dict[name]
        if score >= 0:
            return 1
        else:
            return -1
        
    def create_features(self, x):
        phi = defaultdict(lambda:0)
        words = x.split()
        for word in words:
            phi["UNI:" + word] += 1
            self.word_weight_dict["UNI:" + word]
        return phi

    def update_weights(self, phi, y):
        for name, value in sorted(phi.items()):
            self.word_weight_dict[name] += value * y
    
    def online_train(self, input_file, output_file):
        with open(input_file, "r", encoding = "utf-8") as input_file, \
             open(output_file, "w", encoding = "utf-8") as output_file:
            for line in input_file:
                x = line.strip().split("\t")
                phi = self.create_features(x[1])
                y_pre = self.predict_one(phi)
                y_ans = int(x[0])
                if y_pre != y_ans:
                    self.update_weights(phi, y_ans)
            for key, value in sorted(self.word_weight_dict.items()):
                output_file.write(f'{key}\t{value}\n')

            

    def test(self, input_file, output_file):
        with open(input_file, "r", encoding = "utf-8") as input_file, \
             open(output_file, "w", encoding = "utf-8") as output_file:
            for line in input_file:
                phi = self.create_features(line)
                y_pre = self.predict_one(phi)
                output_file.write(f'{y_pre}\t{line}')


if __name__ == "__main__":
    perceptron = Perceptron()
    perceptron.online_train(sys.argv[1], sys.argv[2])
    perceptron.test(sys.argv[3], sys.argv[4])

'''
Accuracy = 90.967056%
'''