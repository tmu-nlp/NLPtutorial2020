import sys
from collections import defaultdict

my_dict = defaultdict(lambda:1)
my_file = open(sys.argv[1], "r")

for line in my_file:
    line = line.strip()
    words = line.split(" ")
    for word in words:
        if word in my_dict:
            my_dict[word] += 1
        else:
            my_dict[word]

for key, val in sorted(my_dict.items()):
    print("%s %r" % (key, val))

my_file.close()
