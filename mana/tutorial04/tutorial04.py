from collections import defaultdict
from math import log2


def load_model(model_file):
    with open(model_file, "r") as model_file:
        model_file = model_file.readlines()
    transition = defaultdict(bool)
    emission = defaultdict(bool)
    possible_tags = {}
    for line in model_file:
        line = line.strip().split(" ")
        type = line[0]
        context = line[1]
        word = line[2]
        prob = float(line[3])
        possible_tags[context] = 1  # 可能なタグとして保存

        if type == "T":
            transition[(context, word)] = prob
        else:
            emission[(context, word)] = prob

    return [transition, emission, possible_tags]


class hmm:
    def test_hmm(self, model_file, test_file):
        model = load_model(model_file)
        transition = model[0]
        emission = model[1]
        possible_tags = model[2]

        with open(test_file, "r") as f:
            test_file = f.readlines()

        ans = open("my_answer.txt", "w")

        for line in test_file:
            words = line.strip().split()
            words.append("</s>")  # 重要
            I = len(words)
            best_score = {}
            best_edge = {}
            best_score[(0, "<s>")] = 0  # <s> から始まる
            best_edge[(0, "<s>")] = None

            for i in range(I):
                for prev in possible_tags:
                    # print(prev)
                    for next in possible_tags:
                        # print(next)
                        # print(best_score[(i, prev)], transition[(prev,next)])
                        if (i, prev) in best_score and transition[
                            (prev, next)
                        ] != False:
                            # print("Yes")
                            # print(transition[(prev, next)])
                            # print(words[i], next)
                            # print(emission[(next, words[i])])
                            score = (
                                best_score[(i, prev)]
                                - log2(transition[(prev, next)])
                                - log2(
                                    (0.95 * emission[(next, words[i])] + 0.05 / 1000000)
                                )
                            )
                            # print(score)
                            if (i + 1, next) not in best_score or best_score[
                                (i + 1, next)
                            ] > score:
                                best_score[(i + 1, next)] = score
                                best_edge[(i + 1, next)] = (i, prev)
                    if (i, prev) in best_score and transition[(prev, "</s>")] != False:
                        score = (
                            best_score[(i, prev)]
                            + log2(transition[(prev, "</s>")])
                            + log2(
                                (0.95 * emission[("</s>", words[i])] + 0.05 / 1000000)
                            )
                        )
                        if (i + 1, "</s>") not in best_score or best_score[
                            i + 1, "</s>"
                        ] > score:
                            best_score[(i + 1, "</s>")] = score
                            best_edge[(i + 1, "</s>")] = (i, prev)
            # print(best_score)
            # print(best_edge)
            tags = []
            next_edge = best_edge[(I, "</s>")]
            # print(best_edge[(I, "</s>")])

            while next_edge not in {(0, "<s>"), False}:  # このエッジの品詞を出力に追加
                # position = next_edge[0]
                tag = next_edge[1]
                tags.append(tag)
                next_edge = best_edge[next_edge]
                # print(next_edge)

            tags.reverse()
            ans.write(" ".join(tags) + "\n")
        ans.close()

    def train_hmm(self, train_file, output):
        emit = defaultdict(int)
        transition = defaultdict(int)
        context = defaultdict(int)

        with open(train_file, "r") as f:
            train_file = f.readlines()

        for line in train_file:
            previous = "<s>"  # 文頭記号
            context[previous] += 1
            wordtags = line.strip().split()
            # print(wordtags)
            for wordtag in wordtags:
                word, tag = wordtag.split("_")
                transition[previous + " " + tag] += 1  # 遷移を数え上げる
                context[tag] += 1  # 文脈を数え上げる
                emit[tag + " " + word] += 1  # 生成を数え上げる
                previous = tag
            transition[previous + " </s>"] += 1
        # 遷移確率を出力

        output_file = open(output, "w")
        for key in transition:
            previous, word = key.split()
            output_file.write(
                "T " + key + " " + str(transition[key] / context[previous]) + "\n"
            )
        # 生成確率を出力
        for key in emit:
            previous, word = key.split()
            output_file.write(
                "E " + key + " " + str(emit[key] / context[previous]) + "\n"
            )
        output_file.close()


HmmML = hmm()
# HmmML.train_hmm("wiki-en-train.norm_pos", "HmmModel.txt")
# HmmML.test_hmm("05-train-answer.txt", "05-test-input.txt")
HmmML.test_hmm("HmmModel.txt", "wiki-en-test.norm")

"""
perl gradepos.pl wiki-en-test.pos my_answer.txt
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
