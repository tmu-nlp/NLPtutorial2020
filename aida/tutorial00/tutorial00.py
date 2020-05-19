import sys

file_path = sys.argv[1]
word_to_freq = {}

with open(file_path) as fp:
    for line in fp:
        words = line.split()
        #print(words)
        for word in words:
            if word in word_to_freq:
                word_to_freq[word] += 1
            else:
                word_to_freq[word] = 1

#print(word_to_freq)

sorted_word_to_freq = sorted(word_to_freq.items(), key=lambda x:x[0])
#print(sorted_word_to_freq)
for w_f in sorted_word_to_freq:
    word = w_f[0]
    freq = w_f[1]
    print(f'{word}\t{freq}')

