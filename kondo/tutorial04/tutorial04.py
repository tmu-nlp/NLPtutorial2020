from collections import defaultdict
import math

test = 0

if test == 1:
    train_file = "../../test/05-train-input.txt"
    test_file = "../../test/05-test-input.txt"
    model = "test_model_file.txt"
    ans = "test_ans_file.txt"
else:
    train_file = "../../data/wiki-en-train.norm_pos"
    test_file = "../../data/wiki-en-test.norm"
    model = "model_file.txt"
    ans = "my_answer.pos"

def train_hmm(train_file_name, model_file_name):

    emit = defaultdict(int)
    transition = defaultdict(int)
    context = defaultdict(int)

    with open(train_file_name, encoding = "utf-8") as file_data,\
            open(model_file_name, "w", encoding = "utf-8") as file_model:
        for line in file_data:
            previous = "<s>" #文頭記号
            context[previous] += 1
            wordtags = line.split()
            for wordtag in wordtags:
                word, tag = wordtag.split("_")
                transition[previous+" "+tag] += 1   #遷移を数え上げる
                context[tag] += 1                   #文脈を数え上げる
                emit[tag + " " + word] += 1         #生成を数え上げる
                previous = tag
            transition[previous+" </s>"] += 1
        #遷移確率を出力
        sorted_transition = sorted(transition.items(), key = lambda x: x[0])
        for key, value in sorted_transition:
            previous, word = key.split()
            file_model.write("T {} {:6f}\n".format(key, float(value)/float(context[previous])))
        #生成確率を生成
        sorted_emit = sorted(emit.items(), key = lambda x: x[0])
        for key, value in sorted_emit:
            tag, word = key.split()
            file_model.write("E {} {:6f}\n".format(key, float(value)/float(context[tag])))

def viterbi(test_file_name, model_file_name, ans_file_name):
    #model読み込み
    transition = {}
    emission = {}
    possible_tags = {}
    with open(model_file_name, encoding = "utf-8") as file_model:
        for line in file_model:
            typ, context, word, prob = line.split()
            possible_tags[context] = 1
            if typ == "T":
                transition["{} {}".format(context, word)] = float(prob)
            else:
                emission["{} {}".format(context, word)] = float(prob)

    #前向きステップ
    lambda_1 = 0.95
    lambda_unk = 1 - lambda_1
    V = 1000000
    with open(test_file_name, encoding = "utf-8") as file_test,\
            open(ans_file_name, "w", encoding = "utf-8") as file_ans:
        for line in file_test:
            words = line.split()
            l = len(words)
            best_score = {}
            best_edge = {}
            #文頭
            best_score["0 <s>"] = 0
            best_edge["0 <s>"] = None
            #中間
            for i in range(l):
                for prev in possible_tags:
                    for nex in possible_tags:
                        X = "{} {}".format(i, prev)
                        Y = "{} {}".format(prev, nex)
                        Z = "{} {}".format(nex, words[i])
                        if X in best_score and Y in transition:
                            if Z in emission:
                                score = best_score[X] - math.log2(transition[Y]) - math.log2(lambda_1*emission[Z] + lambda_unk/V)
                            else: score = best_score[X] - math.log2(transition[Y]) - math.log2(lambda_unk/V)
                            W = "{} {}".format(i + 1, nex)
                            if W not in best_score or best_score[W] > score:
                                best_score[W] = score
                                best_edge[W] = X
            #文末
            for tag in possible_tags:
                X = "{} {}".format(l, tag)
                Y = "{} </s>".format(tag)
                Z = "{} </s>".format(l + 1)
                if X in best_score and Y in transition:
                    score = best_score[X] - math.log2(transition[Y])
                    if Z not in best_score or best_score[Z] > score:
                        best_score[Z] = score
                        best_edge[Z] = X

            tags = []
            next_edge = best_edge["{} </s>".format(l + 1)]
            while next_edge != "0 <s>":
                position, tag = next_edge.split()
                tags.append(tag)
                next_edge  = best_edge[next_edge]
            tags.reverse()
            file_ans.write(" ".join(tags) + "\n")

if __name__ == "__main__":
    train_hmm(train_file, model)
    viterbi(test_file, model, ans)

"""
Accuracy: 90.82% (4144/4563)

Most common mistakes:
NNS --> NN      45
NN --> JJ       27
NNP --> NN      22
JJ --> DT       22
VBN --> NN      12
JJ --> NN       12
NN --> IN       11
NN --> DT       10
NNP --> JJ      8
JJ --> VBN      7
"""