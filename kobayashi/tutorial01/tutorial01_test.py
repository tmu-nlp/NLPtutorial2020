import sys
import math

#学習したモデルの読み込み

model_file = open(sys.argv[1],"r")
probabilities = {}      #最尤推定で求めた確率（学習した結果のモデル）を入れる

for line in model_file:
    a = line.split()        #aは仮の変数
    probabilities[a[0]] = float(a[1])


#評価と結果表示

test_file = open(sys.argv[2],"r") 
lambda_1 = 0.95                  #既知語である確率
lambda_unk = 1 - lambda_1        #未知語である確率
V = 1000000                      #未知語を含む語彙数
W = 0                            #test_fileの合計文字数のカウント用
H = 0                            #-log2(P)のsum用（H/W でエントロピー求まる）
unkwon_word_count = 0            #未知語のカウント用（重複もカウントするよ）

for line in test_file:  
    words = line.split()
    words.append("</s>")
    for word in words:
        W += 1
        P = lambda_unk / V      #lambda_unk * (1 / V) : 未知語のための調整項
        if word in probabilities:
            P += lambda_1 * probabilities[word]
        else:
            unkwon_word_count += 1
        H += -math.log(P, 2)    #底は２つ目の引数

print("entropy = " + str((H / W)))
print("coverage = " + str(((W - unkwon_word_count) / W)))