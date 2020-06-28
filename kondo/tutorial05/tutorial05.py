from collections import defaultdict

test = 0

if test == 1:
    train_file = "../../test/03-train-input.txt"
    model = "test_model_file.txt"
else:
    train_file = "../../data/titles-en-train.labeled"
    test_file = "../../data/titles-en-test.word"
    model = "model_file.txt"
    ans = "my_answer.pos"



def update_weights(w, phi, y):
    for name, value in phi.items():
        w[name] += value*y
    return w

def perceptron(train_file_name, model_file_name):
    w = defaultdict(int)
    with open(train_file_name, encoding="utf-8") as file_train, \
            open(model_file_name, "w", encoding="utf-8") as file_model:
        for line in file_train:
            y, x = line.split("\t")
            y = int(y)
            phi = create_features(x)
            y_prime = predict_one(w, phi)
            if y_prime != y:
                w = update_weights(w, phi, y)
        sorted_w = sorted(w.items(), key = lambda x: x[0])
        for key, value in sorted_w:
            file_model.write("{}\t{:6f}\n".format(key, value))

def predict_one(w, phi):
    score = 0
    for name, value in phi.items():
        if name in w:
            score += value*w[name]
    if score >= 0:
        return 1
    else:
        return -1

def predict_all(model_file_name, input_file_name, ans_file_name):
    w = {}
    with open(model_file_name, encoding="utf-8") as file_model,\
            open(input_file_name, encoding="utf-8") as file_input,\
            open(ans_file_name, "w", encoding="utf-8") as file_ans:

        for line in file_model:
            name, w_name = line.split()
            w_name = float(w_name)
            w[name] = w_name
        print(w)

        for line in file_input:
            phi = create_features(line)
            y_prime = predict_one(w, phi)
            file_ans.write("{}\n".format(y_prime))

def create_features(x):
    phi = defaultdict(int)
    words = x.split()
    for word in words:
        phi["UNI:"+word] += 1

    return phi

if __name__ == "__main__":
    if test == 1:
        perceptron(train_file, model)
    else:
        perceptron(train_file, model)
        predict_all(model, test_file, ans)


#diff test_model_file.txt ../../test/03-train-answer.txt 
#../../script/grade-prediction.py ../../data/titles-en-test.labeled my_answer.pos
#Accuracy = 90.967056%