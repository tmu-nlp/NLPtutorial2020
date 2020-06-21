#python tutorial04.py ../../test/05-train-input.txt trainResult.txt ../../test/05-test-input.txt testResult.txt
#python tutorial04.py ../../data/wiki-en-train.norm_pos trainResult2.txt ../../data/wiki-en-test.norm testResult2.txt
import sys
from collections import defaultdict
import math
class Hmm:
    def __init__(self):
        self.emit = defaultdict(lambda:0)
        self.transition = defaultdict(lambda:0)
        self.context = defaultdict(lambda:0)
        self.lambda_1 = 0.95
        self.lambda_unk = 1 - self.lambda_1
        self.N = 1e6

    def train(self, in_file, out_file):
        with open(in_file, "r", encoding = "utf-8") as in_file:
            for line in in_file:
                previous = "<s>"
                self.context[previous] += 1
                wordtags = line.strip().split(" ")

                for wordtag in wordtags:
                    word, tag = wordtag.split("_")
                    self.transition[previous + " " + tag] += 1
                    self.context[tag] += 1
                    self.emit[tag + " " + word] += 1
                    previous = tag
                self.transition[previous + " " + "</s>"] += 1  
        with open(out_file, "w", encoding = "utf-8") as out_file:
            for key, value in sorted(self.transition.items()):
                previous, word = key.split(" ")
                out_file.write("T" + " " + key + " " + f'{value/self.context[previous]:.06f}' + "\n")
            for key, value in sorted(self.emit.items()):
                tag, word = key.split(" ")
                out_file.write("E" + " " + key + " " + f'{value/self.context[tag]:.06f}' + "\n")
        
    def test(self, model_file, test_file, out_file):
        with open(model_file, "r", encoding="utf-8") as file:
            transition = defaultdict(lambda:0)
            emission = defaultdict(lambda:0)
            possible_tags = defaultdict(lambda:0)
            with open(out_file, "w", encoding = "utf-8") as out_file:
                for line in file:
                    word_type, context, word, prob = line.strip().split(" ")
                    possible_tags[context] = 1
                    if word_type == "T":
                        transition[context + " " + word] = float(prob)
                    else:
                        emission[context + " " + word] = float(prob)
                with open(test_file, "r", encoding="utf-8")as file:
                    for line in file:
                        best_score = {}
                        best_edge = {}
                        best_score["0 <s>"] = 0
                        best_edge["0 <s>"] = ""
                        words = line.strip().split()
                        word_length = len(words)
                        #forward
                        for i in range(word_length):
                            for prev_word in possible_tags:
                                for next_word in possible_tags:
                                    a = f'{i} {prev_word}'
                                    b = f'{i+1} {next_word}'
                                    c = f'{prev_word} {next_word}'
                                    if a in best_score and  c in transition:
                                        prob_emi = self.lambda_unk / self.N
                                        if f'{next_word} {words[i]}' in emission:
                                            prob_emi += self.lambda_1 * emission[f'{next_word} {words[i]}']
                                        score = best_score[a] + -math.log2(transition[c]) + -math.log2(prob_emi)
                                        if  (b not in best_score) or(best_score[b] > score) :
                                            best_score[b] = score
                                            best_edge[b] = a
                        #</s>をadd
                        for prev_word in possible_tags:
                            a = f'{word_length} {prev_word}'
                            b = f'{word_length+1} </s>'
                            c = f'{prev_word} </s>'
                            if a in best_score and c in transition:
                                prob_emi = self.lambda_unk / self.N
                                if f'{next_word} {words[i]}' in emission:
                                    prob_emi += self.lambda_1 * emission[f'{next_word} {words[i]}']
                                score = best_score[a] + -math.log2(transition[c]) + -math.log2(prob_emi)
                                if (b not in best_score) or (best_score[b] > score) :
                                    best_score[b] = score
                                    best_edge[b] = a
                        #backward
                        tags = []
                        next_edge = best_edge[f'{(word_length+1)} </s>']
                        while next_edge != "0 <s>":
                            position, tag = next_edge.split()
                            tags.append(tag)
                            next_edge = best_edge[next_edge]
                        tags.reverse()
                        out_file.write(" ".join(tags))
                        out_file.write("\n")

if __name__ == "__main__":
    hmm = Hmm()
    hmm.train(sys.argv[1],sys.argv[2])
    hmm.test(sys.argv[2],sys.argv[3],sys.argv[4])