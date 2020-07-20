#!/usr/bin/python

import sys
import math

ref = list()
test = list()

# Load the reference file
ref_file = open(sys.argv[1], "r",encoding = 'utf-8')
for line in ref_file:
    line = line.strip()
    columns = line.split("\t")
    ref.append(int(columns[0]))
ref_file.close()

# Load the testing file
test_file = open(sys.argv[2], "r",encoding = 'utf-8')
for line in test_file:
    line = line.strip()
    columns = line.split("\t")
    test.append(int(columns[0]))
test_file.close()

# Check to make sure that both are the same length
if len(test) != len(ref):
    sys.exit(1)

total = 0
correct = 0
for i in range(0, len(ref)):
    total += 1
    if ref[i] == test[i]:
        correct += 1

print (f'Accuracy:{float(float(correct)/float(total)*100.0):.06f}%')
