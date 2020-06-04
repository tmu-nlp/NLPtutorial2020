import sys
from collections import defaultdict

my_dict = defaultdict(lambda:0)
total_count = 0           #総単語数カウント用
training_file = open(sys.argv[1],"r")

for line in training_file:
    line = line.strip()         #行末記号(\n)の削除
    words = line.split()
    words.append("</s>")        #文末記号(\s)の追加
    for word in words:
        my_dict[word] += 1
        total_count += 1

#ファイルへ書き出し

model_file = open(sys.argv[2],"w")

for word, count in sorted(my_dict.items()):
    probability = count/total_count
    model_file.write(word+"\t"+str(probability)+"\n") 

model_file.close()
training_file.close()
