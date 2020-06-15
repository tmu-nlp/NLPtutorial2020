import os
import joblib
from math import log2
from typing import List
from collections import defaultdict, OrderedDict

# TODO: 3以上のN-gramの拡張
class NgramLanguageModel:
   def __init__(self, n_gram):
      self.n = n_gram
      self.cnts = defaultdict(lambda: 0)
      self.cnt_u = defaultdict(lambda: 0)
      self.context_cnts = defaultdict(lambda: 0)
      self.probs = defaultdict(lambda: 0)
      assert self.n >= 1, "N-gram must be positive integer."

   def attach_tokens(self, words: List[str]):
      start_tokens = ["<s>"] * (self.n - 1)
      end_tokens = ["</s>"] * (self.n - 1)
      return start_tokens + words + end_tokens

   def witten_bell_smoothing(self, word: str) -> int:
      if word in self.cnts:
         c = self.cnts[word]
         u = self.cnt_u[word]
         return (1 - (u / (u + c)))
      else:
         return 0.05

   def train(self, filename: str):
      # Validate
      assert os.path.exists(filename), f"{filename} does not exist."
      # Read text
      with open(filename) as f:
         for line in f:
            words = line.split()
            words = self.attach_tokens(words)

            # Count total word and individual words
            for i in range(len(words) - (self.n - 1)):
               if f"{words[i]}_{words[i+1]}" not in self.cnts:
                  self.cnt_u[words[i]] += 1
               self.cnts[f"{words[i]}_{words[i+1]}"] += 1
               self.context_cnts[words[i]] += 1
               self.cnts[words[i+1]] += 1
               self.context_cnts[""] += 1
      # Calculate probability
      for n_gram, cnt in self.cnts.items():
         words = n_gram.split("_")
         if len(words) < self.n:
            words = [""] * (self.n - len(words)) + words
         self.probs[n_gram] = self.cnts[n_gram] / self.context_cnts[words[0]]

   def test(self, filename: str):
      V = 1e6; W = 0; H = 0
      lambda_1 = 0.5; lambda_2 = 0.5
      assert os.path.exists(filename), f"{filename} does not exist."
      with open(filename) as f:
         for line in f:
            words = line.split()
            words = self.attach_tokens(words)
            for i in range(len(words) - (self.n - 1)):
               lambda_2 = self.witten_bell_smoothing(words[i])
               p1 = lambda_1 * self.probs[words[i+1]] + (1 - lambda_1) / V
               p2 = lambda_2 * self.probs[f"{words[i]}_{words[i+1]}"] + (1 - lambda_2) * p1
               H += -log2(p2)
               W += 1
      print(f"Entropy   : {H/W:.6f}")
      print(f"Perplexity: {2**(H/W):.6f}")
      # print(f"Coverage  : {(W-unk_cnt)/W:.6f}")
   def save_txt(self, filename: str):
      with open(filename, mode='w') as f:
         for word, w_prob in sorted(self.probs.items()):
               f.write(f"{word} {w_prob}\n")

   def load_txt(self, filename: str):
      with open(filename) as f:
         for word, w_prob in self.probs.items():
               for line in f:
                  word, prob = line.split()
                  self.probs[word] = float(prob)

if __name__ == "__main__":
   NgramLM = NgramLanguageModel(2)
   txt_file = "tutorial01.txt"
   # train_file = "../../test/01-train-input.txt"
   # test_file = "../../test/01-test-input.txt"
   """
   Entropy   : 6.612325
   Perplexity: 97.838146
   """

   train_file = "../../data/wiki-en-train.word"
   test_file = "../../data/wiki-en-test.word"
   """
   Entropy   : 9.850008
   Perplexity: 922.885321
   """
   NgramLM.train(train_file)
   NgramLM.save_txt(txt_file)
   NgramLM.load_txt(txt_file)
   NgramLM.test(test_file)