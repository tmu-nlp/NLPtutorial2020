import sys
import pickle
import math

def test(file_path, vocab_size=1000000, unknown_prob=0.05):
    """ calculate entropy and coverage

    :param word_to_prob: dict, [word]: probability
    :param vocab_size: int, consider unknown words
    :param unknown_prob: float, probability of unknown words

    :param unknown_freq: int, frequency of unknown words
    :param total_words: list, total words in test dataset
    :param log_likelihoods: list, store the log-likelihood

    :return entropy, perplexity, coverage:
    """
    # load dictionary
    fp = open('word_to_prob.pkl', 'rb')
    word_to_prob = pickle.load(fp)

    # settings
    unknown_freq = 0
    total_words = []
    log_likelihoods = []

    # read file
    with open(file_path) as fp:
        for line in fp:
            log_likelihood = 0
            words = line.split()
            # begin of sentence token
            words.append('/s')

            total_words.extend(words)

            for word in words:
                prob = unknown_prob / vocab_size
                if word in word_to_prob:
                    prob += (1 - unknown_prob) * word_to_prob[word]
                else:
                    unknown_freq += 1
                log_likelihood -= math.log(prob, 2)
            log_likelihoods.append(log_likelihood)

    # compute entropy, perplexity and coverage
    entropy = sum(log_likelihoods) / len(total_words)
    perplexity = 2**entropy
    coverage = (len(total_words) - unknown_freq) / len(total_words)

    return entropy, perplexity, coverage


if __name__ == '__main__':
    file_path = sys.argv[1]
    entropy, perplexity, coverage = test(file_path)
    print('entropy: {}\nperplexity: {}\ncoverage: {}'.format(entropy, perplexity, coverage))


