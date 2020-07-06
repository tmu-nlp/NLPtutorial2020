from collections import defaultdict
from trainSVM import create_features

def predict_one(w, phi):
    score = 0
    for name, value in phi.items():
        if name in w:
            score += value + w[name]
    if score >= 0:
        return 1
    else:
        return -1

def predict_all(model_file, input_file):
    w = defaultdict(lambda:0)
    for line in model_file:
        line = line.strip().split('\t')
        name = line[0]
        weight = line[1]
        w[name] = float(weight)
    
    res = []
    for line in input_file:
        line = line.strip()
        phi = create_features(line)
        y_predict = predict_one(w, phi)
        res.append(str(y_predict) + '\t' + line + '\n')

    return res

if __name__ == "__main__":
    model = open('model_titles_en')
    input = open('titles-en-test.word')

    with open('my_answer','w') as fout:
        prediction = predict_all(model, input)
        
        for line in prediction:
            fout.write(line)

# python3 grade-prediction.py titles-en-test.labeled my_answer
# c = 0.001     Accuracy = 87.743535%
# c = 0.0001    Accuracy = 88.345731%
# c = 0.00001   Accuracy = 87.920652%