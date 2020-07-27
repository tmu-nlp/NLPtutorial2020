from collections import defaultdict
import random
import sys
import math

random.seed(1)

NUM_TOPICS = 2
xcorpus = []
ycorpus = []
xcounts = defaultdict(int)
ycounts = defaultdict(int)
x_words = set()
y_topics = set()

def initialize(path):
    global xcorpus
    global ycorpus
    global xcounts
    global ycounts
    global x_words
    global y_topics
    with open(path, encoding="utf-8") as f:
        for line in f:
            docid = len(xcorpus)
            words = line.strip().split()
            topics = []
            for word in words:
                topic = random.randrange(NUM_TOPICS)
                topics.append(topic)
                addcounts(word, topic, docid, 1)
                x_words.add(word)
                y_topics.add(topic)
            xcorpus.append(words)
            ycorpus.append(topics)

def addcounts(word, topic, docid, amount):
    global xcounts
    global ycounts
    xcounts[f"{topic}"] += amount
    if xcounts[f"{topic}"] < 0:
        print(f"xcounts[{topic}] < 0")
        sys.exit()

    xcounts[f"{word}|{topic}"] += amount
    if xcounts[f"{word}|{topic}"] < 0:
        print(f"xcounts[{word}|{topic}] < 0")
        sys.exit()

    ycounts[f"{docid}"] += amount
    if ycounts[f"{docid}"] < 0:
        print(f"ycounts[{docid}] < 0")
        sys.exit()

    ycounts[f"{topic}|{docid}"] += amount
    if xcounts[f"{topic}|{docid}"] < 0:
        print(f"ycounts[{topic}|{docid}] < 0")
        sys.exit()

def sampleone(probs):
    z = sum(probs)
    remaining = random.random()*z
    for i in range(len(probs)):
        remaining -= probs[i]
        if remaining <= 0:
            return i
    print("sampleone error")
    sys.exit()

def sampling(iter=100, alpha=1e-6, beta=1e-6):
    global xcorpus
    global ycorpus
    global x_words
    global y_topics
    N_x = len(x_words)
    N_y = len(y_topics)
    for _ in range(iter):
        ll = 0
        for i in range(len(xcorpus)):
            for j in range(len(xcorpus[i])):
                x = xcorpus[i][j]
                y = ycorpus[i][j]
                addcounts(x, y, i, -1)
                probs = []
                for k in range(NUM_TOPICS):
                    p_xk = (xcounts[f"{x}|{k}"] + alpha)/(ycounts[f"{k}"] + alpha*N_x)
                    p_ky = (ycounts[f"{k}|{y}"] + beta)/(ycounts[f"{y}"] + beta*N_y)
                    probs.append(p_xk*p_ky)
                new_y = sampleone(probs)
                ll += math.log(probs[new_y])
                addcounts(x, new_y, i, 1)
                ycorpus[i][j] = new_y
        print(ll)

if __name__ == "__main__":
    #path = "../../test/07-train.txt"
    path = "../../data/wiki-en-documents.word"
    initialize(path)
    sampling(1, alpha=1e-2, beta=1e-2)
    
    with open("my_ans", "w", encoding="utf-8") as f:
        for i in range(len(xcorpus)):
            for j in range(len(xcorpus[i])):
                f.write(f"{xcorpus[i][j]}_{ycorpus[i][j]}\n")