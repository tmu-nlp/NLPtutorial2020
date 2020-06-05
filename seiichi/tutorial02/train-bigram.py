# train 1-gram model
import os, sys
import math
import dill
import numpy as np
if not os.path.exists("./model"): os.mkdir("./model")

from bigram import BigramModel


if __name__ == "__main__":
    # train_path = sys.argv[1]
    # train_path = "../../test/02-train-input.txt"
    # test_path = "../../test/02-test-input.txt"
    train_path = "../../data/wiki-en-train.word"
    model = BigramModel()
    model.train(train_path)
    model.save_model("./model/wikien.model")
    # model.out_word_prob(smoothing=True)

""" bigram word probs of `test/02-train-input.txt`
<s> a 1.000000
_ </s> 0.250000
_ a 0.250000
_ b 0.250000
_ c 0.125000
_ d 0.125000
a b 1.000000
b c 0.500000
b d 0.500000
c </s> 1.000000
d </s> 1.000000
"""