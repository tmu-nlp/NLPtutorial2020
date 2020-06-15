# train 1-gram model
import os, sys
import math
import pickle
import numpy as np
if not os.path.exists("./model"): os.mkdir("./model")

from unigram import UnigramModel


if __name__ == "__main__":
    # train_path = sys.argv[1]
    # train_path = "../../test/01-train-input.txt"
    # test_path = "../../test/01-test-input.txt"
    train_path = "../../data/wiki-en-train.word"
    model = UnigramModel()
    model.train(train_path)
    model.save_model("./model/wikien.model")
    # model.word_probabilities()