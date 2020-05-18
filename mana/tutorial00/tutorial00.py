#!/usr/bin/python3
import sys

counts = {}

testFile = open(sys.argv[1], 'r')
testLines = [line.strip() for line in testFile.readlines()]

for line in testLines:
    line = line.split(' ')
    for word in line:
        if word in counts:
            counts[word] = counts[word]+1
        else:
            counts[word] = 1
 
#no. of unique words
print("the number of unique words in a given text is " + str(len(counts)))

#word frequency in alphabetical order
print("a list of words in a given text by frequency is")
for key, value in sorted(counts.items()):
    print('%s --> %r' % (key, value))      #homage to Graham sensei's style using arrows in the slide