import sys

text_file = open(sys.argv[1], encoding='utf-8')
text_data = text_file.read()
text_file.close()

word_dic = {}

for line in text_data:
    words = line.split()
    for word in words:
        if not(word in word_dic):
            word_dic[word] = int(1)
        else:
            word_dic[word] += 1

#辞書型をソートするとリスト型で帰ってくる
sorted_word_list = sorted(word_dic.items(), key=lambda x: x[0])

for (key, value) in sorted_word_list:
    print("{}\t{}".format(key, value))

