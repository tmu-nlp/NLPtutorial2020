#python tutorial09.py ../../data/wiki-en-documents.word
import sys
from collections import defaultdict
import numpy as np

class topicModel():
    def __init__(self, num_topics):
        self.num_topics = num_topics
        self.xcorpus = []
        self.ycorpus = []
        self.xcounts = defaultdict(lambda:0)
        self.ycounts = defaultdict(lambda:0)
        self.p = defaultdict(lambda:0)
        self.α = 0.01
        self.β = 0.01
    
    def sample_one(self, probs):
        z = np.sum(probs)
        remaining = np.random.rand()*z
        for i in range(len(probs)):
            remaining -= probs[i]
            if remaining <= 0 :
                return i

    def initialization(self, input_file):
        file = open(input_file, "r", encoding = "utf-8")
        for line in file:
            doc_id = len(self.xcorpus)
            words = line.strip().split()
            topics = []
            for word in words:
                topic = np.random.randint(0, self.num_topics)
                topics.append(topic)
                self.add_counts(word, topic, doc_id, 1)
            self.xcorpus.append(words)
            self.ycorpus.append(topics)
        file.close()

    def add_counts(self, word, topic, doc_id, amount):
        self.xcounts[f"{topic}"] += amount
        self.xcounts[f"{word}|{topic}"] += amount
        self.ycounts[f"{doc_id}"] += amount
        self.ycounts[f"{topic}|{doc_id}"] += amount

    def culc_prob(self, word, topic, doc_id, N_x, N_y):
        self.p[f"{word}|{topic}"] = (self.xcounts[f"{word}|{topic}"]+self.α)/(self.xcounts[f"{topic}"]+ N_x*self.α)
        self.p[f"{topic}|{doc_id}"] = (self.ycounts[f"{topic}|{doc_id}"]+self.β)/(self.ycounts[f"{doc_id}"] + N_y*self.β)

    def sampling(self, iterations):
        for iter in range(iterations):
            for i in range(len(self.xcorpus)):
                for j in range(len(self.xcorpus[i])):
                    x = self.xcorpus[i][j]
                    y = self.ycorpus[i][j]
                    self.add_counts(x, y, i, -1)
                    self.culc_prob(x, y, i, len(self.xcorpus[i]), self.num_topics)
                    probs = []
                    for k in range(self.num_topics):
                        probs.append(self.p[f"{x}|{k}"] * self.p[f"{k}|{y}"])

                    new_y = self.sample_one(probs)
                    self.add_counts(x, new_y, i, 1)
                    self.ycorpus[i][j] = new_y
    
    def output_x(self, output_file):
        file = open(output_file, "w", encoding = "utf-8")
        for i,j in self.xcounts.items():
            file.write(f"c({i}):{j} ")
        file.close()
    
    def output_y(self, output_file):
        file = open(output_file, "w", encoding = "utf-8")
        for i,j in self.ycounts.items():
            file.write(f"c({i}):{j} ")
        file.close()
    
    def check(self):
        count0 = defaultdict(lambda:0)
        count1 = defaultdict(lambda:0)
        for i,j in self.xcounts.items():
            if "|" in i:
                x = i.split("|")
                if x[1] == "0":
                    count0[x[0]] += j
                else:
                    count1[x[0]] += j

        print("topic0")
        print(sorted(count0.items(), key = lambda x:x[1], reverse=True)[:50])
        print("\ntopic1")
        print(sorted(count1.items(), key = lambda x:x[1], reverse=True)[:50])

if __name__ == "__main__":
    topic_model = topicModel(num_topics = 2)
    topic_model.initialization(sys.argv[1])
    topic_model.sampling(iterations = 1)
    topic_model.output_x("result_x.txt")
    topic_model.output_y("result_y.txt")
    topic_model.check()
'''
topic0　お寺
[('the', 36268), (',', 31888), ('of', 18705), ('.', 16924), ('and', 13600), ('in', 11729), ('(', 11348), (')', 11315), ('to', 10124), ('a', 9345), ('was', 6910), ('&quot;', 5563), ('is', 5553), ('as', 5122), ('&apos;', 4886), ('that', 3620), ('by', 3384), ('it', 3322), ('on', 2952), ('he', 2859), ('&apos;s', 2824), ('for', 2766), ('from', 2693), ('with', 2642), ('his', 2396), ('emperor', 2327), ('no', 2123), ('at', 2078), ('which', 1914), ('kyoto', 1848), (':', 1844), ('were', 1841), ('are', 1795), ('this', 1710), ('an', 1653), ('who', 1640), ('temple', 1565), ('family', 1558), ('city', 1522), ('or', 1495), ('but', 1470), ('period', 1442), ('be', 1392), (';', 1349), ('after', 1339), ('there', 1317), ('had', 1260), ('imperial', 1211), ('also', 1172), ('not', 1106)]

topic1　電車
[('the', 44060), (',', 34523), ('of', 22374), ('.', 20212), ('and', 16773), ('in', 12659), (')', 12519), ('(', 12510), ('to', 11870), ('was', 8793), ('a', 8691), ('&apos;', 5649), ('is', 5247), ('as', 5179), ('station', 5083), ('&quot;', 4897), ('for', 3894), ('on', 3839), ('that', 3834), (':', 3819), ('kyoto', 3701), ('by', 3642), ('it', 3622), ('from', 3261), ('with', 2989), ('line', 2873), ('at', 2798), ('&apos;s', 2745), ('were', 2544), ('no', 2308), ('school', 2219), ('he', 2133), ('temple', 2093), ('which', 2009), ('are', 1937), ('this', 1855), ('his', 1783), ('city', 1739), ('an', 1658), ('but', 1611), ('university', 1502), ('who', 1493), ('or', 1473), ('between', 1464), ('-', 1373), ('be', 1358), ('also', 1304), ('shrine', 1289), ('trains', 1254), ('emperor', 1253)]
'''