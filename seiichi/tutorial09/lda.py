import sys, os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from scipy.special import digamma

class LDA(object):
    def __init__(self, documents, num_words=None, num_topic=10, alpha=0.01, beta=0.02):
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
        self.init()
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
        for i in tqdm(range(self.num_docs), postfix="sampling"):
            for j in range(len(self.documents[i])):
                word_id = self.documents[i][j]
                topic_id = self.Z[i][j]
                # prepare statistics w/o w_{word_id}
                self.n_d_k[i, topic_id] -= 1
                self.n_k_w[topic_id, word_id] -= 1
                self.n_k[topic_id] -= 1
                probs = self.get_topic_prob_dist(i, word_id)
                if any(np.isnan(probs)):
                    raise Exception("probs contain nan")
                new_topic_id = np.random.multinomial(1, probs).argmax()
                # update statistics
                self.Z[i][j] = new_topic_id
                self.n_d_k[i, new_topic_id] += 1
                self.n_k_w[new_topic_id, word_id] += 1
                self.n_k[new_topic_id] += 1
        return self
    
    def calc_sample_log_likelihood(self, doc_id):
        logL = 0
        for j in range(len(self.documents[doc_id])):
            tmp = 0
            for k in range(self.num_topic):
                tmp += self.get_prob(doc_id, k, self.documents[doc_id][j])
            logL += np.log(tmp)
        return logL

    def calc_log_likelihood(self):
        logL = 0
        for i in range(self.num_docs):
            for j in range(len(self.documents[i])):
                tmp = 0
                for k in range(self.num_topic):
                    tmp += self.get_prob(i, k, self.documents[i][j])
                logL += np.log(tmp)
        return logL
    
    def calc_perplexity(self):
        return np.exp(-self.calc_log_likelihood() / self.total_words)
    
    def calc_sample_perplexity(self):
        # sample one document randomly
        doc_id = np.random.choice(range(self.num_docs), 1)[0]
        total_words = len(self.documents[doc_id])
        return np.exp(-self.calc_sample_log_likelihood(doc_id) / total_words)

    def calc_new_alpha_k(self, k):
        denom, numer= 0, 0
        for i in range(self.num_docs):
            denom += digamma(sum(self.n_d_k[i]) + sum(self.alpha)) - digamma(sum(self.alpha))
        for i in range(self.num_docs):
            numer += (digamma(self.n_d_k[i, k] + self.alpha[k]) - digamma(self.alpha[k]))
        return numer / denom * self.alpha[k]

    # fixed-point iteration; ref: https://www.coronasha.co.jp/np/isbn/9784339027587/
    def update_alpha(self):
        for k in tqdm(range(self.num_topic), postfix="alpha_update"):
            self.alpha[k] = self.calc_new_alpha_k(k)
        return self

    def get_topic(self, doc_id, word_id):
        topic_dist = self.get_topic_prob_dist(doc_id, word_id)
        topic = topic_dist.argmax()
        return topic
    
    def get_high_freq_words_in_topic(self, topic_id, id2word, num_words=20):
        tmp = self.n_k_w[topic_id].copy()
        indices = self.n_k_w[topic_id].argsort()[::-1][:num_words]
        ranking = []
        for index in indices:
            ranking.append((tmp[index], id2word[index]))
        return ranking


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

def look_topic(model, documents, id2word, path="./result/sample.txt"):
    with open(path, "w") as f:
        for doc_id, doc in enumerate(documents):
            f.write("document: {}\n".format(doc_id))
            for word in doc:
                f.write("{}: {}, ".format(id2word.get(word), model.get_topic(doc_id, word)))
            f.write("\n")

def ranking(model, id2word, path="./result/topic.txt"):
    with open(path, "w") as f:
        for i in range(model.num_topic):
            ranking = model.get_high_freq_words_in_topic(i, id2word)
            f.write("topic #{}\n".format(i))
            for cnt, word in ranking:
                f.write("{}: {}, ".format(word, cnt))
            f.write("\n")

if __name__ == "__main__":
    documents, word2id = load_data("../../data/wiki-en-documents.word")
    id2word = {v: k for k, v in word2id.items()}
    lda = LDA(documents=documents,num_words=len(word2id.items()),num_topic=10)
    for i in range(30):
        lda.sampling().update_alpha()
        ppl = lda.calc_sample_perplexity()
        print("perplexity:", ppl)
    look_topic(lda, documents[:10], id2word)
    ranking(lda, id2word)


"""iteration log
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 732/732 [02:50<00:00,  4.30it/s, sampling]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 78.50it/s, alpha_update]
perplexity: 1166.988994728392
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 732/732 [02:48<00:00,  4.36it/s, sampling]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 78.68it/s, alpha_update]
perplexity: 994.0577595864557
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 732/732 [02:46<00:00,  4.41it/s, sampling]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 72.84it/s, alpha_update]
perplexity: 947.418630475421
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 732/732 [02:47<00:00,  4.38it/s, sampling]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 79.71it/s, alpha_update]
perplexity: 932.8883766403045
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 732/732 [02:44<00:00,  4.44it/s, sampling]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 82.90it/s, alpha_update]
perplexity: 908.5961145207184
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 732/732 [02:47<00:00,  4.38it/s, sampling]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 76.95it/s, alpha_update]
perplexity: 758.986268424211
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 732/732 [02:45<00:00,  4.43it/s, sampling]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 81.88it/s, alpha_update]
perplexity: 820.5312918255563
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 732/732 [02:48<00:00,  4.36it/s, sampling]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 79.73it/s, alpha_update]
perplexity: 1075.965528626552
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 732/732 [02:46<00:00,  4.39it/s, sampling]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 78.29it/s, alpha_update]
perplexity: 776.1885113068798
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 732/732 [02:46<00:00,  4.41it/s, sampling]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 68.07it/s, alpha_update]
perplexity: 847.9171188713785
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 732/732 [02:54<00:00,  4.20it/s, sampling]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 77.27it/s, alpha_update]
perplexity: 777.9359387822318
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 732/732 [02:52<00:00,  4.25it/s, sampling]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 79.92it/s, alpha_update]
perplexity: 800.207625538823
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 732/732 [02:57<00:00,  4.12it/s, sampling]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 77.51it/s, alpha_update]
perplexity: 556.6963490725749
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 732/732 [02:41<00:00,  4.53it/s, sampling]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 62.40it/s, alpha_update]
perplexity: 755.4142276586833
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 732/732 [02:41<00:00,  4.54it/s, sampling]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 66.73it/s, alpha_update]
perplexity: 798.6026605115497
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 732/732 [02:45<00:00,  4.44it/s, sampling]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 80.46it/s, alpha_update]
perplexity: 797.7586164535369
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 732/732 [02:44<00:00,  4.44it/s, sampling]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 80.62it/s, alpha_update]
perplexity: 704.8385052677107
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 732/732 [02:41<00:00,  4.52it/s, sampling]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 61.97it/s, alpha_update]
perplexity: 738.1032888434299
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 732/732 [02:43<00:00,  4.48it/s, sampling]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 63.63it/s, alpha_update]
perplexity: 701.080319128154
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 732/732 [02:41<00:00,  4.53it/s, sampling]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 79.07it/s, alpha_update]
perplexity: 572.9729081947808
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 732/732 [02:41<00:00,  4.52it/s, sampling]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 81.79it/s, alpha_update]
perplexity: 514.05828773003
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 732/732 [02:41<00:00,  4.54it/s, sampling]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 74.11it/s, alpha_update]
perplexity: 1094.7367376973168
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 732/732 [02:42<00:00,  4.51it/s, sampling]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 77.00it/s, alpha_update]
perplexity: 591.0496263177358
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 732/732 [02:43<00:00,  4.48it/s, sampling]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 82.73it/s, alpha_update]
perplexity: 684.8272985893551
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 732/732 [03:03<00:00,  3.98it/s, sampling]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 79.15it/s, alpha_update]
perplexity: 768.5730093507523
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 732/732 [02:50<00:00,  4.29it/s, sampling]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 81.42it/s, alpha_update]
perplexity: 691.1470993985747
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 732/732 [02:47<00:00,  4.37it/s, sampling]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 80.39it/s, alpha_update]
perplexity: 518.4390994845262
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 732/732 [02:43<00:00,  4.48it/s, sampling]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 77.07it/s, alpha_update]
perplexity: 1080.7930393377271
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 732/732 [02:56<00:00,  4.16it/s, sampling]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 19.40it/s, alpha_update]
perplexity: 681.6418951477493
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 732/732 [02:45<00:00,  4.42it/s, sampling]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 85.66it/s, alpha_update]
perplexity: 808.3028704650784
"""
    