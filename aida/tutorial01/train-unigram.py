import sys
import pickle
from collections import defaultdict

def train(file_path):
    """ train unigram language model
    
    :param word_to_freq: defaultdict, [word]: frequency
    :param total_freq: int, sum of all frequencies

    :return word_to_prob: dict, [word]: probability
    """
    word_to_freq = defaultdict(lambda: 0)

    with open(file_path) as fp:
        for line in fp:
            words = line.split()
            # begin of sentence token
            words.append('/s')
            for word in words:
                word_to_freq[word] += 1

    word_to_prob = {}
    total_freq = sum([freq for freq in word_to_freq.values()])
    for word, freq in word_to_freq.items():
        word_to_prob[word] = freq / total_freq

    return word_to_prob


if __name__ == '__main__':
    file_path = sys.argv[1]
    word_to_prob = train(file_path)
    # save dictionary
    fp = open('word_to_prob.pkl', 'wb')
    pickle.dump(word_to_prob, fp)

