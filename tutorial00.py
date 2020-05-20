import sys
from collections import defaultdict

counts = {}
my_file = open(sys.argv[1])

words = []
for line in my_file:
    words = line.split()
    
    for w in words:
        if w in counts:
            counts[w] +=1
        else:
            counts[w] = 1

for x, y in sorted(counts.items()):
    print ('%s: %r' % (x, y))

print('\nNumber of Unique Words: ' + str(len(counts)))