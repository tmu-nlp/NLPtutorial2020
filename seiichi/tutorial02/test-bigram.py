import os, sys
import math
import dill
import numpy as np
if not os.path.exists("./model"): os.mkdir("./model")

from bigram import BigramModel


if __name__ == "__main__":
    # train_path = sys.argv[1]
    # train_path = "../../data/wiki-en-train.word"
    test_path = "../../data/wiki-en-test.word"
    model = BigramModel()
    model.load_model("./model/wikien.model")
    ent = model.calc_entropy(test_path)
    print("entoropy: ", ent)
