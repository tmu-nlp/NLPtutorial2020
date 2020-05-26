from collections import defaultdict
import sys
total_cnt = 0
dic = defaultdict(lambda: 0)

with open(sys.argv[1]) as training_file\
    , open(sys.argv[2], mode='w') as model_file:
    for line in training_file:
        words = line.split()
        words.append('</s>')
        for word in words:
            dic[word] += 1
            total_cnt += 1

    for word, cnt in sorted(dic.items(), key=lambda x:x[0]):
        proba = float(cnt / total_cnt)
        model_file.write(f'{word}\t{proba:.6f}\n')

