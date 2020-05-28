from collections import defaultdict

tot_cnt = 0
cnts = defaultdict(int)

"""
file1 = "01-train-input.txt"
file2 = "train-input-model.txt"
"""

file1 = "wiki-en-train.word"
file2 = "model_file.txt"


with open(file1, encoding="utf-8") as trn_file:
    line = trn_file.readline()
    while(line):
        words = line.split()
        words.append("</s>")
        for word in words:
            cnts[word] += 1
            tot_cnt += 1
        line = trn_file.readline()

with open(file2, "w", encoding="utf-8") as mdl_file:
    for word, cnt in sorted(cnts.items()):
        prb = cnts[word]/tot_cnt
        mdl_file.write("{}\t{:.6f}\n".format(word, prb))
