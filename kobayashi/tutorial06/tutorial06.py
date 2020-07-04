#python tutorial06.py ../../test/03-train-input.txt train_result.txt test_input.txt test_result.txt
#python tutorial06.py ../../data/titles-en-train.labeled train_result2.txt ../../data/titles-en-test.word test_result2.txt
#python evaluation.py ../../data/titles-en-test.labeled test_result2.txt

import sys
import math
from collections import defaultdict

class Perceptron:

    def __init__(self):
        self.word_weight_dict = defaultdict(lambda:0)       #重みをインスタンス変数で定義
        self.c = 0.0001
        self.margin = 1
    
    def predict_one(self, phi): 
        score = self.multiply_weight_phi(phi)
        if score >= 0:
            return 1
        else:
            return -1
    
    def multiply_weight_phi(self, phi):
        a=0
        for name, value in phi.items():
            a += value* self.word_weight_dict[name]
        return a
        
    def create_features(self, x):
        phi = defaultdict(lambda:0)
        words = x.split()
        for word in words:
            phi["UNI:" + word] += 1
            self.word_weight_dict["UNI:" + word]
        return phi

    def update_weights(self, phi, y):
        for name, value in self.word_weight_dict.items():
            if abs(value) < self.c:
                self.word_weight_dict[name]=0
            else:
                self.word_weight_dict[name]-= self.sign(value) * self.c
        a = self.multiply_weight_phi(phi)
        b = y * math.exp(a)/((math.exp(a)+1)**2)
        for name, value in phi.items():
            self.word_weight_dict[name] += value*b
    
    def sign(self, value):
        if value >= 0:
            return 1
        else:
            return -1
            
    def online_train(self, input_file, output_file):
        with open(input_file, "r", encoding = "utf-8") as input_file, \
             open(output_file, "w", encoding = "utf-8") as output_file:
            for line in input_file:
                x = line.strip().split("\t")
                phi = self.create_features(x[1])   #phiは頻度ベクトル
                y_ans = int(x[0])
                val = self.multiply_weight_phi(phi)*y_ans
                if val <= self.margin:
                    self.update_weights(phi, y_ans)
            for key, value in sorted(self.word_weight_dict.items()):
                output_file.write(f'{key}\t{value}\n')

    def test(self, input_file, output_file):
        with open(input_file, "r", encoding = "utf-8") as input_file, \
             open(output_file, "w", encoding = "utf-8") as output_file:
            for line in input_file:
                phi = self.create_features(line.strip())
                y_pre = self.predict_one(phi)
                output_file.write(f'{y_pre}\t{line}')


if __name__ == "__main__":
    perceptron = Perceptron()
    perceptron.online_train(sys.argv[1], sys.argv[2])
    perceptron.test(sys.argv[3], sys.argv[4])

'''
Accuracy:91.002480%
'''