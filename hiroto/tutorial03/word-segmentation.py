# トレーニングファイル
# python word-segmentation.py ../data/wiki-ja-train.txt > my_answer_train.word
# ../script/gradews.pl ../data/wiki-ja-train.word my_answer_train.word
# テストファイル
# python word-segmentation.py ../data/wiki-ja-test.txt > my_answer.word
# ../script/gradews.pl ../data/wiki-ja-test.word my_answer.word
# big-ws-modelを使った時
# python word-segmentation.py ../data/wiki-ja-test.txt > my_answer_big.word
# ../script/gradews.pl ../data/wiki-ja-test.word my_answer_big.word
import sys
import pickle
import math
#確率を保存する辞書
dic = {}
#未知語を計算する時に使う．語彙数N=10の6乗
N = 1000000
lambda1 = 0.95

'''
#テスト用のファイル"test/04-model.txt"や，"data/big-ws-model.txt"を読み込む時，
fname1 = "../test/04-model.txt"
fname2 = "../data/big-ws-model.txt"
with open(fname2) as infile:
    for line in infile:
        cols = line.split()
        dic[cols[0]] = float(cols[1].strip())
'''

#辞書を読み込む
with open('./model/ch03_dic.pickle', mode='rb') as modelfile:
    dic = pickle.load(modelfile)


with open(sys.argv[1]) as infile:
    for line in infile:
        line = line.strip()
        #前向きステップ
        best_edge = {}
        best_score = {}
        best_edge[0] = None
        best_score[0] = 0
        for word_end in range(1, len(line)+1):
            best_score[word_end] = float('inf')
            for word_begin in range(0, word_end):
                word = line[word_begin : word_end]
                if word in dic.keys() or len(word) == 1:
                    if word in dic.keys():
                        prob = lambda1*dic[word] + (1-lambda1)/N
                    #未知語に対しての確率
                    else: prob = (1-lambda1)/N
                    my_score = best_score[word_begin] + (-math.log(prob))
                    if my_score < best_score[word_end]:
                        best_score[word_end] = my_score
                        best_edge[word_end] = (word_begin, word_end)
        #後ろ向きステップ
        words = []
        next_edge = best_edge[len(best_edge)-1]
        while next_edge != None:
            #このエッジの部分文字列を追加
            word = line[next_edge[0]:next_edge[1]]
            words.append(word)
            next_edge = best_edge[next_edge[0]]
        words.reverse()
        print(' '.join(words))
