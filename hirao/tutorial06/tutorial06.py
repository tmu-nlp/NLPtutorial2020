import numpy as np
from typing import List, Dict
from sklearn.metrics import classification_report, accuracy_score
from collections import defaultdict


class SupportVectorMachine():
   def __init__(self):
      self.w = defaultdict(lambda: 0)
      self.last = defaultdict(lambda: 0)
      self.c = 0.0001

   def create_feat(self, sentence: str) -> Dict[str, int]:
      feat = defaultdict(lambda: 0)
      words = sentence.split()
      for word in words:
         feat[f"UNI:{word}"] += 1
      return feat

   def sign(self, w: int) -> int:
      if w > 0:
         return 1
      elif w == 0:
         return 0
      else:
         return -1

   def get_w(self, feat_name: str, iter_num: int) -> int:
      if iter_num > self.last[feat_name]:
         c_size = self.c * (iter_num - self.last[feat_name])
         if abs(self.w[feat_name]) <= c_size:
            self.w[feat_name] = 0
         else:
            self.w[feat_name] -= self.sign(self.w[feat_name]) * c_size
         self.last[feat_name] = iter_num
      return self.w[feat_name]

   def predict_one(self, feat: Dict[str, int], iter_num: int = 0) -> int:
      score = 0
      for word, v in feat.items():
         if word in self.w:
            score += v * self.get_w(word, iter_num)
      if score >= 0:
         return 1
      else:
         return -1

   def predict_all(self, input_file: str, iter_num: int = 0) -> List[int]:
      res = []
      with open(input_file, mode='r', encoding='utf-8') as f:
         for line in f:
            feat = self.create_feat(line)
            y = self.predict_one(feat, iter_num)
            res.append(y)
      return res

   def update_w(self, feat: Dict[str, int], sign: int):
      for word, cnt in feat.items():
         self.w[word] += sign * cnt

   def train(self, num_iterations: int, input_file: str, val_file: str=None, val_ans_file: str=None):
      with open(input_file, mode='r', encoding='utf-8') as f:
         for i in range(num_iterations):
            # preds = self.predict_all(val_file, i)
            # check_score(val_ans_file, preds)
            for line in f:
               y, x = line.split("\t")
               y = int(y)
               feat = self.create_feat(x)
               y_pred = self.predict_one(feat, i)
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
   SVM = SupportVectorMachine()
   SVM.train(100, "../../data/titles-en-train.labeled", "../../data/titles-en-test.word", "../../data/titles-en-test.labeled")
   results = SVM.predict_all("../../data/titles-en-test.word")
   # print("\n".join(list(map(str, results))))
   check_score("../../data/titles-en-test.labeled", results, True)

if __name__ == "__main__":
   main()

"""
              precision    recall  f1-score   support

          -1       0.88      0.95      0.92      1477
           1       0.94      0.86      0.90      1346

    accuracy                           0.91      2823
   macro avg       0.91      0.91      0.91      2823
weighted avg       0.91      0.91      0.91      2823

accuracy: 0.9096705632306057
"""