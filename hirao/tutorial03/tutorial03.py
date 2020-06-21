import os
import sys
from math import log2
from typing import List
sys.path.append("../")
from tutorial01.tutorial01 import UnigramLanguageModel
INF = 10**10

class WordSeparator(UnigramLanguageModel):
   def __init__(self):
      super().__init__()
      self.probs = dict()

   def viterbi(self, sentence: str):
      N = len(sentence) + 1; V = 1e6
      best_edges = [[-1, INF] for _ in range(N)]
      best_edges[0] = [None, 0]

      # 前向きステップ
      for word_end in range(1, N):
         for word_begin in range(word_end):
            word = sentence[word_begin:word_end]
            if word in self.probs or len(word) == 1:
               prob = (1 - self.lambda_1)/V
               if word in self.probs:
                  prob += self.lambda_1*self.probs[word]
               cur_score = best_edges[word_begin][1] - log2(prob)
               if cur_score < best_edges[word_end][1]:
                  best_edges[word_end][1] = cur_score
                  best_edges[word_end][0] = word_begin
      # 後ろ向きステップ
      words = []
      cur_edge = N-1
      pre_edge = best_edges[cur_edge][0]
      while pre_edge != None:
         words.append(sentence[pre_edge:cur_edge])
         cur_edge = pre_edge
         pre_edge = best_edges[cur_edge][0]
      return words[::-1]

   def test_separator(self, filename: str, output_file: str):
      assert os.path.exists(filename), f"{filename} does not exist."
      out = []
      with open(filename, encoding='utf-8') as f:
         for line in f:
            line = line.strip()
            res = self.viterbi(line)
            out.append(" ".join(res))
      with open(filename, encoding='utf-8', mode='w') as f:
         for line in out:
            f.write(line)

if __name__ == "__main__":
   tokenizer = WordSeparator()

   # テスト用
   # tokenizer.load_txt("../../test/04-model.txt")
   # tokenizer.test_separator("../../test/04-input.txt")

   train_file = "../../data/wiki-ja-train.word"
   tokenizer.train(train_file)

   test_file = "../../data/wiki-ja-test.txt"
   output_file = "my_answer.word"
   tokenizer.test_separator(test_file, output_file)

   """
   Sent Accuracy: 0.00% (/84)
   Word Prec: 68.93% (1861/2700)
   Word Rec: 80.77% (1861/2304)
   F-meas: 74.38%
   Bound Accuracy: 83.25% (2683/3223)
   """
   # Sent Accuracyが何故か0になる問題、去年もあった気がしたけど忘れた