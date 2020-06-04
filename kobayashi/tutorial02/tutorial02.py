#↓実行コード
#python tutorial02.py ../../data/wiki-en-train.word model_file 

import sys
import math
from collections import defaultdict

class BigramModel():
    def __init__(self, lambda_x):
        self.total_count = 0
        self.bigram_numer_counts = defaultdict(lambda:0)
        self.bigram_denom_counts = defaultdict(lambda:0)
        self.unigram_numer_counts = defaultdict(lambda:0)
        self.unigram_denom_count = 0
        self.lambda_1 = 0.95
        self.lambda_2 = lambda_x / 100
        self.V = 1e6
        self.W = 0
        self.H = 0
    
    def train(self, in_filename, out_filename):
        with open(in_filename,encoding='utf-8') as file:
            for line in file:
                line.lower()
                line.strip()
                words = line.split()
                words.insert(0, "<s>")
                words.append("</s>")
                for i in range(1,len(words)-1):
                    a =" ".join([words[i],words[i+1]])
                    self.bigram_numer_counts[a] += 1
                    self.bigram_denom_counts[words[i]] += 1
                    self.unigram_numer_counts[words[i+1]] += 1
                    self.unigram_denom_count += 1

        out_file = open(out_filename,"w",encoding='utf-8')
        for ngram, count in self.bigram_numer_counts.items():
            bigram = ngram.split()
            probability_of_bigram = count / self.bigram_denom_counts[bigram[0]]
            probability_of_unigram = self.unigram_numer_counts[bigram[1]] / self.unigram_denom_count
            out_file.write(bigram[0] + " " + bigram[1] + " " + str(probability_of_bigram) + " " + str(probability_of_unigram) + "\n")
            
    def test(self, filename):
        with open(filename,"r",encoding = 'utf-8') as file:
            for line in file:
                line.strip()
                words = line.split()
                P1 = self.lambda_1 * float(words[3]) + (1-self.lambda_1) / self.V
                P2 = self.lambda_2 * float(words[2]) + (1 - self.lambda_2) * P1
                self.H += -math.log(P2,2)
                self.W += 1
            
            print(f"lambda_2 = {self.lambda_2} : entropy =  {self.H / self.W}")

    
if __name__ == "__main__":
    for i in range(5,100,5):
        bigram_model = BigramModel(i)
        trained_data_filename = sys.argv[2]
        bigram_model.train(sys.argv[1], trained_data_filename)
        bigram_model.test(trained_data_filename)
    #lambda_2は大きいほどエントロピーは小さくなった