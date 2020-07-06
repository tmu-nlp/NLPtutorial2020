import sys
from collections import defaultdict

train_data="../../data/titles-en-train.labeled"
model="./model"
test_data="../../data/titles-en-test.word"
my_ans_data = "my_answer"

class SVM():
    def __init__(self):
        self.w = defaultdict(int)
        self.margin = 1
        self.c = 0.0001

    def create_features(self, x):
        phi = defaultdict(int)
        words = x.split()
        for word in words:
            phi[f"UNI:{word}"] += 1

        return phi

    def update_weights(self, phi, y):
        for name, value in self.w.items():
            if abs(value) < self.c:
                self.w[name] = 0
            else:
                self.w[name] -= value*self.c
        for name, value in phi.items():
            self.w[name] += value*y

    def predict_one(self, phi):
        score = 0
        for name, value in phi.items():
            if name in self.w:
                score += value*self.w[name]
        if score >= 0:
            return 1
        else:
            return -1

    def predict(self, input_file, output_file):
        w = {}
        with open(input_file, encoding="utf-8") as i_f,\
                open(output_file, "w", encoding="utf-8") as o_f:
            for line in i_f:
                phi = self.create_features(line)
                y_prime = self.predict_one(phi)
                o_f.write(f"{y_prime}\n")

    def dot_weight_phi(self, phi):
        sum = 0
        for key, value in phi.items():
            sum += self.w[key]*phi[key]
        return sum

    def fit(self, file_path, iter):
        for _ in range(iter):
            with open(file_path, encoding="utf-8") as f:
                for line in f:
                    y, x = line.split("\t")
                    y = int(y)
                    phi = self.create_features(x)
                    val = self.dot_weight_phi(phi)*y
                    if val <= self.margin:
                        self.update_weights(phi, y)


if __name__ == "__main__":
    svm = SVM()
    svm.fit(train_data, 10)
    """
    with open(model, "w", encoding="utf-8") as m:
        for key, value in sorted(svm.w.items(), key=lambda x:x[0]):
            m.write(f"{key}\t{value}\n")
    """
    svm.predict(test_data, my_ans_data)

"""
margin 0 iter 1
~/Documents/GitHub/NLPtutorial2020/kondo/tutorial06$ ../../script/grade-prediction.py ../../data/titles-en-test.labeled my_answer
Accuracy = 91.144173%

magin 0
kon@seiichironoMacBook-Air:~/Documents/GitHub/NLPtutorial2020/kondo/tutorial06$ ../../script/grade-prediction.py ../../data/titles-en-test.labeled my_answer
Accuracy = 91.604676%

margin 0.5
kon@seiichironoMacBook-Air:~/Documents/GitHub/NLPtutorial2020/kondo/tutorial06$ ../../script/grade-prediction.py ../../data/titles-en-test.labeled my_answer
Accuracy = 87.885228%

margin 1
kon@seiichironoMacBook-Air:~/Documents/GitHub/NLPtutorial2020/kondo/tutorial06$ ../../script/grade-prediction.py ../../data/titles-en-test.labeled my_answer
Accuracy = 93.092455%

margin 1.5
kon@seiichironoMacBook-Air:~/Documents/GitHub/NLPtutorial2020/kondo/tutorial06$ ../../script/grade-prediction.py ../../data/titles-en-test.labeled my_answer
Accuracy = 91.958909%
"""