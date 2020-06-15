import math
from collections import defaultdict

def wordDelimitation(model_file, text, ans):
    lambda_1 = 0.95
    V = 1000000
    model_dict = defaultdict(float)
    with open(model_file, "r", encoding="utf-8") as modelf:
        probs = modelf.readlines()
    #print(probs)

    for prob in probs:
        prob = prob.strip("\n").split(" ")
        model_dict[prob[0]] = float(prob[1])
    
    #print(model_dict)
    ans_file = open(ans, "w", encoding="utf-8")
    with open(text, "r", encoding="utf-8") as f:
        inputs = f.readlines()
    for line in inputs:
        line = line.strip()
        best_edge = {}
        best_score = {}
        best_edge[0] = None
        best_score[0] = 0
        for word_end in range(1, len(line)+1):
            best_score[word_end] = math.inf
            for word_begin in range(0, word_end):
                word = line[word_begin:word_end]
                #print(word)
                if word in model_dict or len(word) == 1:
                    prob = lambda_1 * model_dict[word] + (1-lambda_1)/V  #ここよくわからない
                    my_score = best_score[word_begin] + -math.log(prob, 2)
                    if my_score < best_score[word_end]:
                        best_score[word_end] = my_score
                        best_edge[word_end] = [word_begin, word_end]
        #print(best_edge)
        #print(best_score)
        words = []
        next_edge = best_edge[len(best_edge)-1]
        while next_edge != None:
            word = line[next_edge[0]:next_edge[1]]
            words.append(word)
            next_edge = best_edge[next_edge[0]]
        words.reverse()
        #print(" ".join(words))
        ans_file.write(" ".join(words) + "\n")
    ans_file.close()
    return None

if __name__ == "__main__":
    """
    with open("/work/test/04-model.txt", "r") as model_file:
        model_file = model_file.readlines()
    model_dict = {}
    for line in model_file:
        line = line.strip().split()
        model_dict[line[0]] = float(line[1])
    #print(model_dict)
    """
    wordDelimitation("model_file.txt", "wiki-ja-test.txt", "my_answer.txt")

"""
Sent Accuracy: 23.81% (20/84)
Word Prec: 68.93% (1861/2700)
Word Rec: 80.77% (1861/2304)
F-meas: 74.38%
Bound Accuracy: 83.25% (2683/3223)
"""
    
#動くけどあんまりしっくりこない。やっぱり最初のnodeは-1が綺麗そう。グラフの実装もうちょいどうにかならんかなー。
#</s>のprob=1はどこの話しだったっけ。
#dataのなかに.txtが全部ない気がする。
#ディレクトリ行ったり来たりする/がすぐわかんなくなる。いつ/つければいいのか。
