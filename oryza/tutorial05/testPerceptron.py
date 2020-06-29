from collections import defaultdict
from trainPerceptron import create_features, predict_one

def predict_all(model_file, input_file):
    w = defaultdict(lambda:0)
    for line in model_file:
        line = line.strip().split('\t')
        name = line[0]
        weight = line[1]
        w[name] = int(weight)
    
    res = []
    for line in input_file:
        line = line.strip()
        phi = create_features(line)
        y_predict = predict_one(w, phi)
        res.append(str(y_predict) + '\t' + line + '\n')

    return res

if __name__ == "__main__":
    input = open('titles-en-test.word')
    model = open('model_titles_en')

    with open('my_answer','w') as fout:
        prediction = predict_all(model, input)
        
        for line in prediction:
            fout.write(line)

# python3 grade-prediction.py titles-en-test.labeled my_answer
# Accuracy = 82.288346%