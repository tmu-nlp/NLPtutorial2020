from collections import defaultdict
import math

test = 0

if test == 1:
    inp = "../../test/04-input.txt"
    model = "../../test/04-model.txt"
else:
    data = "../../data/wiki-ja-train.word"
    model = "model_file.txt"
    inp = "../../data/wiki-ja-test.word"

def modeling_unigram(file1, file2):
    tot_cnt = 0
    cnts = defaultdict(int)
    prob_dic = {}
    with open(file1, encoding="utf-8") as trn_file:
        line = trn_file.readline()
        while(line):
            words = line.split()
            words.append("</s>")
            for word in words:
                cnts[word] += 1
                tot_cnt += 1
            line = trn_file.readline()

    with open(file2, "w", encoding="utf-8") as mdl_file:
        for word, cnt in sorted(cnts.items()):
            prb = cnts[word]/tot_cnt
            prob_dic[word] = prb
    return prob_dic

def Viterbi(target, prob_dic):
    lambda_1 = 0.95
    lambda_unk = 1 - lambda_1
    V = 1000000
    with open(target, encoding = "utf-8") as tar:
        for line in tar:
            line = line.replace(" ", "")
            best_edge = {}
            best_score = {}
            best_edge[0] = None
            best_score[0] = 0
            for word_end in range(1, len(line)):
                best_score[word_end] = 10000000000
                for word_begin in range(0, len(line)-1):
                    word = line[word_begin:word_end]
                    if word in prob_dic or len(word) == 1:
                        if word in prob_dic:
                            prob = lambda_1*float(prob_dic[word]) + lambda_unk/V
                        else:
                            prob = lambda_unk/V
                        my_score = best_score[word_begin] - math.log(prob)
                        if my_score < best_score[word_end]:
                            best_score[word_end] = my_score
                            best_edge[word_end] = (word_begin, word_end)

            words = []
            next_edge = best_edge[len(best_edge) - 1]
            while next_edge != None:
                word = line[next_edge[0]:next_edge[1]]
                words.append(word)
                next_edge = best_edge[next_edge[0]]
            words.reverse()
            print(" ".join(words))

if __name__ == "__main__":
    if test == 1:
        prob_dic = {}
        with open(model, encoding = "utf-8") as mod:
            for line in mod:
                word, prob = line.split()
                prob_dic[word] = prob
    else:
        prob_dic = modeling_unigram(data, model)

    Viterbi(inp, prob_dic)