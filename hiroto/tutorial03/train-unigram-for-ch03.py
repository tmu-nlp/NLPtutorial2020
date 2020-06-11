# トレーニング
# python train-unigram-for-ch03.py ../data/wiki-ja-train.word
import sys
import pickle
total_cnt = 0
dic_cnt = {}
dic_prob = {}

with open(sys.argv[1]) as training_file\
    , open('../model/ch03_dic.pickle', mode='wb') as model_file:
    for line in training_file:
        words = line.split()
        words.append('</s>')
        for word in words:
            if word not in dic_cnt.keys():
                dic_cnt[word] = 0
            dic_cnt[word] += 1
            total_cnt += 1
    for word, cnt in sorted(dic_cnt.items(), key=lambda x:x[0]):
        prob = float(cnt / total_cnt)
        dic_prob[word] = prob
        print(f'{word}\t{prob:.6f}')

    pickle.dump(dic_prob, model_file)