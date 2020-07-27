import random as r
import numpy as np
import math
from collections import defaultdict

def sample_one(probs):
    z = sum(probs)
    remaining = r.random() * z

    for i in range(len(probs)):
        remaining -= probs[i]
        if remaining <= 0:
            return i
    raise Exception('Error at sample one')

def add_counts(xcounts, ycounts, word, topic, docid, amount):
    xcounts[topic] += amount
    xcounts[f'{word}|{topic}'] += amount

    ycounts[docid] += amount
    ycounts[f'{topic}|{docid}'] += amount
    
def train(train_file, num_topics, epoch, alpha, beta):
    xcorpus = []
    ycorpus = []
    xcounts = defaultdict(lambda: 0)
    ycounts = defaultdict(lambda: 0)
    x_size = len(xcorpus)
    y_size = len(ycorpus)

    # initialization
    for line in train_file:
        docid = len(xcorpus)
        words = line.strip().split(' ')
        topics = []
        for w in words:
            topic = r.randint(0, num_topics - 1)
            topics.append(topic)
            add_counts(xcounts, ycounts, w, topic, docid, 1)
        xcorpus.append(words)
        ycorpus.append(topics)

    # sampling
    for e in range(epoch):
        ll = 0
        for i in range(len(xcorpus)):
            for j in range(len(xcorpus[i])):
                x = xcorpus[i][j]
                y = ycorpus[i][j]
                add_counts(xcounts, ycounts, x, y, i, -1) # subtract the counts (hence -1)
                probs = []
                for k in range(num_topics):
                    P_x_K = (xcounts[f'{x}|{k}'] + alpha) / (xcounts[k] + alpha * x_size)
                    P_y_K = (ycounts[f'{k}|{y}'] + beta) / (ycounts[y] + beta * y_size)
                    probs.append(P_x_K * P_y_K) # prob of topic k
                new_y = sample_one(probs)
                ll += math.log(probs[new_y]) # calculate the log likelihood
                add_counts(xcounts, ycounts, x, new_y, i, 1) # add the counts
                ycorpus[i][j] = new_y
    
    for i in range(len(xcorpus)):
        for j in range(len(xcorpus[i])):
            x = xcorpus[i][j]
            y = ycorpus[i][j]
            print(f'{x}_{y}', end=' ')
        print()

if __name__ == "__main__":
    # train_file = open('07-train.txt')
    train_file = open('wiki-en-documents.word')
    alpha = 0.02
    beta = 0.02
    # num_topics = 2
    num_topics = 20
    epoch = 5

    train(train_file, num_topics, epoch, alpha, beta)