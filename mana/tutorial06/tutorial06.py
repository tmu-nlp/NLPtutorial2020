from collections import defaultdict
from math import sin


def predict_margin(w, phi, label):
    score = 0
    for key, value in phi.items():
        if key in w:
            score += value * w[key] * label
    return score


def predict_one(w, phi):
    score = 0
    for key, value in phi.items():
        if key in w:
            score += value * w[key]
    if score >= 0:
        return 1
    else:
        return -1


def create_features(x):
    phi = defaultdict(int)
    for word in x:
        phi["UNI:" + word] += 1
    return phi


def update_weight(w, phi, y, c):
    for key, value in phi.items():
        w[key] += value * y

    for key, value in w.items():
        if abs(value) < c:
            w[key] = 0
        else:
            w[key] -= sin(value) * c


def predict_all(model_file, input_file):
    weights = defaultdict(int)

    with open(model_file, "r") as f:
        model = f.readlines()

    for line in model:
        line = line.strip().split()
        feature = line[0]
        weight = line[1]
        weights[feature] = float(weight)

    with open(input_file, "r") as f1:
        inputFile = f1.readlines()

    output = open("my_answer.txt", "w")

    for line in inputFile:
        phi = create_features(line.strip().split())
        y_pred = predict_one(weights, phi)
        output.write(str(y_pred) + "\t" + line)

    output.close()


class svm:
    def train_svm(self, model_file, output_file, margin):
        with open(model_file) as f:
            model = f.readlines()

        weight = defaultdict(int)

        for iter in range(10):
            for line in model:
                line = line.strip().split()
                features = line[1:]
                label = int(line[0])
                phi = create_features(features)
                val = predict_margin(weight, phi, label)

                if val <= margin:
                    update_weight(weight, phi, label, 0.0001)

        ans = open(output_file, "w")
        for key, value in weight.items():
            ans.write(key + "\t" + str(value) + "\n")
        ans.close()

    def test_svm(self, model_file, test_file):
        predict_all(model_file, test_file)


svmM = svm()

print(svmM.train_svm("titles-en-train.labeled", "my_ans.txt", 100))
print(svmM.test_svm("my_ans.txt", "titles-en-test.word"))

# Results
# epoch = 1
# Accuracy = 91.746369% (svm, margin = 0)
# Accuracy = 92.454835% (svm, margin = 10)
# Accuracy = 92.490259% (svm, margin = 100)
# Accuracy = 90.967056% (perceptron)

# epoch = 10
# Accuracy = 92.242295% (svm, margin = 0)
# Accuracy = 91.888062% (svm, margin = 10)
# Accuracy = 93.517535% (svm, margin = 100)
# Accuracy = 93.446688% (perceptron)

# Accuracy = 93.552958% (perceptron, epoch = 100)
