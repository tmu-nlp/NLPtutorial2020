import sys
dict = {}

my_file = open(sys.argv[1])
for line in my_file:
    #改行文字とか両端から消す
    line = line.strip()
    #単語分割
    words = line.split()

    for word in words:
        if word in dict.keys():
            dict[word] += 1
        else:
            dict[word] = 1

for item in dict.items():
    print(f"{item[0]}\t{item[1]}")

my_file.close()