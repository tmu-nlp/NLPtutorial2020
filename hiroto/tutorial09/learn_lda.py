import random, sys
from collections import defaultdict
from math import log
from tqdm import tqdm
NUM_TOP =8
MAX_ITER = 50
ALPHA = 0.01
BETA = 0.02

def sample_one(probs):
    z = sum(probs)
    remaining = random.uniform(0, z)
    for i in range(len(probs)):
        remaining -= probs[i]
        if remaining <= 0: return i

def add_counts(xcounts, ycounts, word, topic, doc_id, amount):
    xcounts[f"{topic}"] += amount
    xcounts[f"{word}|{topic}"] += amount
    ycounts[f"{doc_id}"] += amount
    ycounts[f"{topic}|{doc_id}"] += amount
    return xcounts, ycounts

if __name__ == '__main__':
    #初期化
    xcorpus, ycorpus = [], []
    xcounts, ycounts = defaultdict(lambda:0), defaultdict(lambda:0)
    word_type = defaultdict(lambda:0)
    with open(sys.argv[1]) as f:
        for doc_id, line in enumerate(f):
            words = line.strip().split()
            topics = []
            for word in words:
                word_type[word]
                topic = random.randint(0, NUM_TOP-1)
                topics.append(topic)
                xcounts, ycounts = add_counts(xcounts, ycounts, word, topic, doc_id, 1)
            xcorpus.append(words)
            ycorpus.append(topics)
    #サンプリング
    ll = 0
    N_x = len(word_type.keys())
    N_y = NUM_TOP
    for _ in tqdm(range(MAX_ITER)):
        for i in range(len(xcorpus)):
            for j in range(len(xcorpus[i])):
                #print(j)
                #print(len(xcorpus[i]))
                x = xcorpus[i][j]
                y = ycorpus[i][j]
                xcounts, ycounts = add_counts(xcounts, ycounts, x, y, i, -1)
                probs = []
                for k in range(NUM_TOP):
                    #p_x_k = xcounts[f"{xcorpus[i][j]}|{ycorpus[i][j]}"] / xcounts[f"{y}"]#xcounts[f"{ycorpus[i][j]}"]
                    #p_k_y = ycounts[f"{ycorpus[i][j]}|{i}"] / ycounts[f"{i}"]
                    p_x_k = (xcounts[f"{x}|{k}"] + ALPHA) / (xcounts[f"{k}"] + ALPHA*N_x)
                    p_k_y = (ycounts[f"{k}|{i}"] + BETA) / (ycounts[f"{i}"] + BETA*N_y)
                    probs.append(p_x_k * p_k_y)
                new_y = sample_one(probs)
                ll += log(probs[new_y])
                xcounts, ycounts = add_counts(xcounts, ycounts, x, new_y, i, 1)
                ycorpus[i][j] = new_y
    
    #topic = defaultdict(lambda: 0)
    topic_0 = defaultdict(lambda: 0)
    topic_1 = defaultdict(lambda: 0)
    topic_2 = defaultdict(lambda: 0)
    topic_3 = defaultdict(lambda: 0)
    topic_4 = defaultdict(lambda: 0)
    topic_5 = defaultdict(lambda: 0)
    topic_6 = defaultdict(lambda: 0)
    topic_7 = defaultdict(lambda: 0)
    dic = {0:topic_0, 1:topic_1, 2:topic_2, 3:topic_3,
            4:topic_4, 5:topic_5, 6:topic_6, 7:topic_7}
    for i in range(len(xcorpus)):
        for j in range(len(xcorpus[i])):
            topic = dic[ycorpus[i][j]]
            topic[f"{xcorpus[i][j]}"] += 1
    topics = [topic_0, topic_1, topic_2, topic_3, topic_4, topic_5, topic_6, topic_7]

    with open("result.txt", mode="w") as of:
        for idx, topic in enumerate(topics):
            of.write("****************************************************\n")
            of.write(f"topic {idx}:\n")
            of.write("****************************************************\n")
            of.write(str(sorted(topic.items(), key=lambda x:x[1], reverse=True)) + '\n')