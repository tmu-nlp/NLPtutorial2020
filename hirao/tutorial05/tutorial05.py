import numpy as np
from typing import List, Dict
from sklearn.metrics import classification_report, accuracy_score
from collections import defaultdict


class Perceptron():
   def __init__(self):
      self.w = defaultdict(lambda: 0)

   def create_feat(self, sentence: str) -> Dict[str, int]:
      feat = defaultdict(lambda: 0)
      words = sentence.split()
      for word in words:
         feat[f"UNI:{word}"] += 1
      return feat

   def predict_one(self, feat: Dict[str, int]) -> int:
      score = 0
      for word, v in feat.items():
         if word in self.w:
            score += v * self.w[word]
      if score >= 0:
         return 1
      else:
         return -1

   def predict_all(self, input_file: str) -> List[int]:
      res = []
      with open(input_file, mode='r', encoding='utf-8') as f:
         for line in f:
            feat = self.create_feat(line)
            y = self.predict_one(feat)
            res.append(y)
      return res

   def update_w(self, feat: Dict[str, int], sign: int):
      for word, cnt in feat.items():
         self.w[word] += sign * cnt

   def train(self, num_iterations: int, input_file: str, val_file: str=None, val_ans_file: str=None):
      with open(input_file, mode='r', encoding='utf-8') as f:
         for i in range(num_iterations):
            preds = self.predict_all(val_file)
            check_score(val_ans_file, preds)
            for line in f:
               y, x = line.split("\t")
               y = int(y)
               feat = self.create_feat(x)
               y_pred = self.predict_one(feat)
               if y != y_pred:
                  self.update_w(feat, y)

def check_score(gold_file: str, pred: List[int], detail: bool=False):
   gold = []
   with open(gold_file, mode='r', encoding='utf-8') as f:
      for line in f:
         label = int(line.split("\t")[0])
         gold.append(label)
   gold = np.array(gold)
   pred = np.array(pred)
   if detail:
      print(classification_report(gold, pred))
   print(f"accuracy: {accuracy_score(gold, pred)}")


def main():
   perceptron = Perceptron()
   perceptron.train(10, "../../data/titles-en-train.labeled", "../../data/titles-en-test.word", "../../data/titles-en-test.labeled")
   results = perceptron.predict_all("../../data/titles-en-test.word")
   # print("\n".join(list(map(str, results))))
   check_score("../../data/titles-en-test.labeled", results, True)

if __name__ == "__main__":
   main()