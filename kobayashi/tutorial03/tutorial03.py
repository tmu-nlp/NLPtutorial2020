# 実行コード
# python tutorial03.py input.txt ../../test/04-input.txt result.txt
# python tutorial03.py ../../data/wiki-ja-train.word ../../data/wiki-ja-test.word result2.txt

import sys
import math
from collections import defaultdict

class WordDivision:
    def __init__(self):
        self.unigram_probs_dict = defaultdict(lambda:0)
        self.unigram_total_count = 0
        self.lambda_1 = 0.95
        self.lambda_unk = 1 - self.lambda_1

    def calc_unigram_probs(self, filename):
        training_file = open(filename, "r", encoding = "utf-8")
        for line in training_file:
            line = line.strip()         
            words = line.split()
            words.append("</s>")       
            for word in words:
                self.unigram_probs_dict[word] += 1
                self.unigram_total_count += 1
        for i in self.unigram_probs_dict:
            self.unigram_probs_dict[i] /= self.unigram_total_count
    
    def solve(self, in_filename, out_filename):
        out_file = open(out_filename, "w", encoding = "utf-8")
        with open(in_filename, "r", encoding = "utf-8") as file:
            for line in file:

                #forward_step
                words_list = line.strip().split()
                words = "".join(words_list)
                best_score = list(range(len(words)+1))
                best_edge = list(range(len(words)+1))
                best_edge[0] = [0,0]
                best_score[0] = 0
                for word_end in range(1, len(words)+1, 1):
                    best_score[word_end] = 1e10
                    for word_begin in range(0, word_end, 1):
                        word = words[word_begin:word_end]
                        if word in self.unigram_probs_dict or len(word) == 1:
                            prob = self.lambda_1 * self.unigram_probs_dict[word] + self.lambda_unk / self.unigram_total_count
                            my_score = best_score[word_begin] + -math.log2(prob)
                            if my_score < best_score[word_end]:
                                best_score[word_end] = my_score
                                best_edge[word_end] = [word_begin, word_end]

                #backward_step
                best_path = []
                next_edge = best_edge[len(words)]
                while next_edge != [0, 0]:
                    best_path.append(next_edge[1])
                    next_edge = best_edge[next_edge[0]]
                best_path.reverse()
                j = 0
                str_1 = ""
                for i in best_path:
                    str_1 = f"{str_1}{words[j:i]} "
                    j = i
                out_file.write(str_1[:-1] + "\n")
        out_file.close()

if __name__ == "__main__":
    word_division = WordDivision()
    word_division.calc_unigram_probs(sys.argv[1])
    word_division.solve(sys.argv[2], sys.argv[3])