import math
import string
from collections import defaultdict

import numpy as np
from nltk.corpus import stopwords
from tqdm import tqdm


class LDA():
    def __init__(self, num_top, doc, vocab, alpha=0.5, beta=0.5):
        self.num_top = num_top
        self.doc = doc
        self.vocab = vocab
        self.alpha = alpha
        self.beta = beta
        self.topics_doc = []
        self.xcounts = defaultdict(lambda: 0)
        self.ycounts = defaultdict(lambda: 0)

    def add_count(self, word, topic, docid, amount):
        self.xcounts[topic] += amount
        self.xcounts[f'{word}|{topic}'] += amount
        self.ycounts[docid] += amount
        self.ycounts[f'{topic}|{docid}'] += amount

    def initialize(self):
        """ initialize topics

        :param file_path: path of file
        """
        for docid, words in enumerate(self.doc):
            topics = []
            for word in words:
                topic = np.random.randint(self.num_top)
                topics.append(topic)
                self.add_count(word, topic, docid, 1)
            self.topics_doc.append(topics)
                
    def prob_word_given_topic(self, word, topic):
        """ compute P(word|topic)

        :return: p_w_k
        """
        Nx = len(self.vocab)
        p_w_k = self.xcounts[f'{word}|{topic}'] + self.alpha
        p_w_k /= self.xcounts[topic] + self.alpha * Nx
        return p_w_k

    def prob_topic_given_topics(self, topic, docid):
        """ compute P(topic|topics)
        
        :return: p_k_y
        """
        Ny = self.num_top
        p_k_y = self.ycounts[f'{topic}|{docid}'] + self.beta
        p_k_y /= self.ycounts[docid] + self.beta * Ny
        return p_k_y

    def sample_one(self, probs):
        """ sample one label
        
        :param probs: list, probabilities
        :return: label sampled
        """
        z = sum(probs)
        remaining = np.random.uniform(0, z)
        for i in range(len(probs)):
            remaining -= probs[i]
            if remaining <= 0:
                return i
    
    def sampling(self, max_iter):
        """ sampling

        :param max_iter: max iteration
        """
        for _ in tqdm(range(max_iter)):
            for i in range(len(self.doc)):
                ll = 0
                for j in range(len(self.doc[i])):
                    word = self.doc[i][j]
                    topic = self.topics_doc[i][j]
                    self.add_count(word, topic, i, -1)
                    probs = []
                    for k in range(self.num_top):
                        p_w_k = self.prob_word_given_topic(word, k)
                        p_k_y = self.prob_topic_given_topics(k, i)
                        probs.append(p_w_k*p_k_y)
                    new_topic = self.sample_one(probs)
                    ll += math.log(probs[new_topic])
                    self.add_count(word, new_topic, i, 1)
                    self.topics_doc[i][j] = new_topic
            #print(ll)


def preprocess(file_path, stopwords=[]):
    """ file => doc
    :param file_path: path of file
    :param stopwords: stopwords
    :return: doc, vocab 
    """
    doc = []
    vocab = set()
    with open(file_path) as fp:
        for line in fp:
            sent = []
            words = line.strip().split()
            for word in words:
                if word not in stopwords:
                    vocab.add(word)
                    sent.append(word)
            doc.append(sent)
    return doc, vocab

def sort_each_topic(xcounts, topic, vocab, topk):
    """ in each topic

    :return: sorted_words
    """
    word_freq = {}
    for word in vocab:
        freq = xcounts[f'{word}|{topic}']
        if freq > 0:
            word_freq[word] = freq
    sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1])
    sorted_words = [w_f[0] for w_f in sorted_word_freq]
    return sorted_words[:topk]

if __name__ == '__main__':
    np.random.seed(1)
    #file_test = './test/07-train.txt'
    file_train = './data/wiki-en-documents.word'

    # define stopwords
    punctuations = set(string.punctuation)
    stop_words = set(stopwords.words('english'))
    stop_words = stop_words.union(punctuations)
    #from pprint import pprint
    #pprint(stop_words)
    #exit()

    # preprocessing
    #doc, vocab = preprocess(file_test)
    doc, vocab = preprocess(file_train, stopwords=stop_words)

    # define LDA
    NUM_TOP = 5
    MAX_ITER = 50
    lda = LDA(num_top=NUM_TOP, doc=doc, vocab=vocab)
    lda.initialize()

    # sampling
    lda.sampling(max_iter=MAX_ITER)

    for topic in range(NUM_TOP):
        print(f'topic {topic}')
        words = sort_each_topic(lda.xcounts, topic, vocab, topk=100)
        print(' '.join(words))
        print()

