import sys, os
import numpy as np
from collections import defaultdict
from scipy.special import digamma
# from numba import jitclass, f8, i8, b1, u1

# spec = [
#     ("documents", u1[:,:]),
#     ("num_topic", i8),
#     ("num_docs", i8),
#     ("num_words", i8),
#     ("total_words", i8),
#     ("alpha", f8[:]),
#     ("beta", f8),
#     ("Z", i8[:,:]),
#     ("n_d_k", i8[:,:]),
#     ("n_k_w", i8[:,:]),
#     ("n_k", i8[:]),
# ]

# @jitclass(spec)
class LDA(object):
    def __init__(self, documents, num_words=None, num_topic=10, alpha=0.1, beta=0.1):
        self.documents = documents
        self.num_topic = num_topic
        self.num_docs = len(documents)
        if num_words == None:
            self.num_words = self._cnt_words(documents)
        else:
            self.num_words = num_words
        self.total_words = sum([len(doc) for doc in self.documents])
        self.alpha = np.fromiter((1.0 / self.num_topic for i in range(self.num_topic)), dtype=float, count=self.num_topic)
        self.beta = beta
        self.Z = list(map(lambda x: np.zeros(shape=(len(x)), dtype=int), documents))
        self.n_d_k = np.zeros(shape=(self.num_docs, self.num_topic), dtype=int) # count of topic k in document d
        self.n_k_w = np.zeros(shape=(self.num_topic, self.num_words), dtype=int) # count of word w in topic k
        self.n_k = np.zeros(shape=(self.num_topic, ), dtype=int) # count of topic k
        return 

    def _cnt_words(self, documents):
        return len(set([word for doc in documents for word in doc]))

    def init(self):
        for i in range(self.num_docs):
            for j in range(len(self.documents[i])):
                word_id = self.documents[i][j]
                # randomly allocate topic for first time
                topic_id = np.random.choice(range(self.num_topic), 1)[0]
                self.Z[i][j] = topic_id
                self.n_k_w[topic_id, word_id] += 1
                self.n_k[topic_id] += 1
            for k in range(self.num_topic):
                self.n_d_k[i, k] = (self.Z[i]==k).sum()
        return self
    
    def get_prob(self, doc_id, topic_id, word_id):
        return (self.n_d_k[doc_id, topic_id] + self.alpha[topic_id]) * (self.n_k_w[topic_id, word_id] + self.beta) \
            / (self.alpha[topic_id] * self.num_topic + sum(self.n_d_k[doc_id])) / (self.n_k[topic_id] + self.beta * self.num_words)

    def get_topic_prob_dist(self, doc_id, word_id):
        probs = np.zeros(self.num_topic)
        for t in range(self.num_topic):
            probs[t] = self.get_prob(doc_id, t, word_id)
        probs /= probs.sum()
        return probs

    def sampling(self):
        for i in range(self.num_docs):
            for j in range(len(self.documents[i])):
                word_id = self.documents[i][j]
                topic_id = self.Z[i][j]
                # prepare statistics w/o w_{word_id}
                self.n_d_k[i, topic_id] -= 1
                self.n_k_w[topic_id, word_id] -= 1
                self.n_k[topic_id] -= 1
                probs = self.get_topic_prob_dist(i, word_id)
                new_topic_id = np.random.multinomial(1, probs).argmax()
                # update statistics
                self.Z[i][j] = new_topic_id
                self.n_d_k[i, new_topic_id] += 1
                self.n_k_w[new_topic_id, word_id] += 1
                self.n_k[new_topic_id] += 1
        return self
        
    def calc_log_likelihood(self):
        logL = 0
        for i in range(self.num_docs):
            for j in range(self.num_words):
                for k in range(self.num_topic):
                    logL += self.get_prob(i, k, j)
        return np.log(logL)
    
    def calc_perplexity(self):
        return np.exp(-self.calc_log_likelihood() / self.total_words)
    
    def calc_new_alpha_k(self, k):
        denom, numer= 0, 0
        for i in range(self.num_docs):
            denom += digamma(sum(self.n_d_k[i]) + sum(self.alpha)) - digamma(sum(self.alpha))
        for i in range(self.num_docs):
            numer += (digamma(self.n_d_k[i, k]) - digamma(self.alpha[k])) * self.alpha[k]
        return numer / denom

    # fixed-point iteration; ref: https://www.coronasha.co.jp/np/isbn/9784339027587/
    def update_alpha(self):
        for k in range(self.num_topic):
            self.alpha[k] = self.calc_new_alpha_k(k)
        return self


def load_data(path):
    vocab = set()
    with open(path, "r") as f:
        tmp = f.readlines()
    data = []
    for doc in tmp:
        tar = doc.strip().split()
        for word in tar:
            vocab.add(word)
        data.append(tar)
    word2id = defaultdict(lambda: int)
    for i, word in enumerate(list(vocab)):
        word2id[word] = i
    data_new = []
    for doc in data:
        doc_new = []
        for word in doc:
            doc_new.append(word2id.get(word, 0))
        data_new.append(doc_new)
    return data_new, word2id

if __name__ == "__main__":
    documents, word2id = load_data("../../data/wiki-en-documents.word")
    lda = LDA(documents=documents,num_words=len(word2id.items()),num_topic=10)
    lda.init()
    print(lda.total_words)
    for i in range(10):
        lda.sampling().update_alpha()
        # p = lda.calc_log_likelihood()
        # print("lod_likelihood:", p)
        ppl = lda.calc_perplexity()
        print("perplexity:", ppl)