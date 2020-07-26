import random
import math
from tqdm import tqdm
from collections import defaultdict
import re
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))

class lda():
    def __init__(self, num_topics = 2):
        self.num_topics = num_topics
        self.xcorpus = []
        self.ycorpus = []
        self.vocab = defaultdict(int)
        self.xcounts = defaultdict(int)
        self.ycounts = defaultdict(int)

    def initialize(self, datapath):
        with open(datapath, "r") as f:
            file = f.readlines()
        for line in file:
            words = preprocess(line.strip()) #stopwordの除去
            topics = []
            for word in words:
                topic = random.randint(1, self.num_topics)
                topics.append(topic)
                self.vocab[word] += self.vocab[word] + 1   #平滑化のための単語異なり数
                self.AddCounts(word, topic, len(self.xcorpus), 1)
            self.xcorpus.append(words)
            self.ycorpus.append(topics)

    def AddCounts(self, word, topic, docid, amount):
        self.xcounts[topic] += amount
        self.xcounts[word + "|" + str(topic)] += amount
        self.ycounts[docid] += amount
        self.xcounts[str(topic) + "|" + str(docid)] += amount

    def TrainLda(self, epoch, alpha=0.1, beta=0.1):
        for _ in range(epoch):
            ll = 0
            for i in tqdm(range(len(self.xcorpus))):
                for j in range(len(self.xcorpus[i])):
                    x = self.xcorpus[i][j]
                    y = self.ycorpus[i][j]
                    self.AddCounts(x, y, i, -1)
                    probs = []
                    for k in range(self.num_topics):
                        p_x_k = (self.xcounts[x +"|"+ str(k)]+alpha)/(self.ycounts[k]+alpha*len(self.vocab))
                        p_k_y = (self.ycounts[str(k) +"|"+ str(y)]+beta)/(self.ycounts[y]+beta*self.num_topics)
                        probs.append(p_x_k*p_k_y)

                    new_y = SampleOne(probs)
                    ll += math.log(probs[new_y])
                    self.AddCounts(x, new_y, i, 1)
                    self.ycorpus[i][j] = new_y


def SampleOne(probs):
    z = sum(probs)
    remaining = random.uniform(0, z)
    for i in range(len(probs)):
        remaining = remaining - probs[i]
        if remaining <= 0:
            return i

def preprocess(text):
  cleaned = re.sub('[^a-zA-Z]+',' ', text) # remove non-alphbetic
  cleaned = [i for i in cleaned.split() if not i in stop_words] # remove stopwords
  return cleaned

#モデル
#datapath = "/work/test/07-train.txt"
datapath = "/work/data/wiki-en-documents.word"
ldaModel = lda()
ldaModel.initialize(datapath)   #初期化
ldaModel.TrainLda(5)

#結果
topicA = defaultdict(int)
topicB = defaultdict(int)
for words, topics in zip(ldaModel.xcorpus, ldaModel.ycorpus):
    for word, topic in zip(words, topics):
        if topic == 0:
            topicA[word] += 1
        else:
            topicB[word] += 1

print("TopicA")
print(sorted(topicA.items(), key=lambda x: x[1], reverse=True)[:100])
print("TopicB")
print(sorted(topicB.items(), key=lambda x: x[1], reverse=True)[:100])

"""
頻度の上から100番を出力。もっと良い可視化の方法がありそう。
TopicA
[('lineage', 70), ('financial', 70), ('nobunaga', 59), ('leave', 55), ('kyo', 51), ('koen', 49), ('kyotanabe', 47), ('enter', 47), ('promotion', 46), ('ancestor', 44), ('water', 43), ('room', 43), ('courts', 42), ('offered', 41), ('elementary', 41), ('located', 40), ('sects', 39), ('local', 39), ('according', 38), ('refers', 37), ('powerful', 37), ('association', 36), ('old', 36), ('philosophy', 36), ('fu', 36), ('college', 36), ('list', 35), ('branches', 34), ('numbers', 34), ('weapon', 34), ('wanted', 34), ('easy', 33), ('split', 33), ('sakai', 33), ('approximately', 33), ('teishi', 33), ('battles', 31), ('media', 31), ('life', 30), ('agreement', 30), ('team', 30), ('tengu', 30), ('rich', 29), ('deities', 29), ('prefectures', 28), ('crown', 28), ('rivers', 28), ('passed', 27), ('seiryu', 27), ('pagoda', 27), ('sadaie', 27), ('grandfather', 27), ('compilation', 26), ('based', 26), ('greatest', 26), ('note', 26), ('tea', 26), ('enemies', 26), ('provide', 25), ('onmyo', 25), ('overall', 25), ('director', 24), ('practical', 24), ('hirakatashi', 24), ('ages', 23), ('posthumously', 23), ('exile', 23), ('ability', 23), ('luck', 23), ('appropriate', 23), ('enjoyed', 23), ('executive', 23), ('necessary', 23), ('less', 23), ('yabusame', 23), ('april', 23), ('funai', 23), ('paradise', 22), ('th', 22), ('suggests', 22), ('dainichinyorai', 22), ('achieve', 22), ('hidetada', 22), ('holder', 22), ('tokaido', 22), ('diaries', 22), ('hitotsubashi', 22), ('northeast', 21), ('portrait', 21), ('overseas', 21), ('resign', 21), ('sections', 21), ('address', 21), ('designation', 21), ('michi', 21), ('talent', 21), ('injured', 21), ('ogi', 21), ('served', 21), ('addition', 21)]
TopicB
[('apos', 16760), ('quot', 10460), ('station', 5680), ('kyoto', 5662), ('temple', 3692), ('emperor', 3648), ('line', 3566), ('city', 3298), ('school', 2721), ('period', 2669), ('ji', 2504), ('also', 2476), ('imperial', 2445), ('family', 2253), ('one', 2066), ('prefecture', 1885), ('called', 1870), ('japan', 1764), ('became', 1717), ('shrine', 1715), ('clan', 1704), ('university', 1632), ('name', 1565), ('time', 1544), ('however', 1416), ('japanese', 1376), ('used', 1366), ('trains', 1355), ('first', 1338), ('railway', 1338), ('court', 1328), ('province', 1233), ('established', 1208), ('fujiwara', 1200), ('government', 1180), ('main', 1173), ('train', 1167), ('section', 1143), ('two', 1136), ('year', 1100), ('national', 1047), ('later', 1034), ('system', 1023), ('prince', 1022), ('many', 1021), ('son', 1019), ('made', 1012), ('said', 1001), ('express', 1000), ('people', 998), ('area', 993), ('street', 993), ('new', 927), ('osaka', 916), ('old', 908), ('since', 882), ('ward', 879), ('three', 860), ('edo', 848), ('castle', 843), ('order', 833), ('war', 830), ('cultural', 822), ('years', 804), ('hall', 803), ('dori', 780), ('shogun', 769), ('west', 768), ('th', 765), ('day', 759), ('around', 752), ('style', 749), ('cho', 745), ('kamakura', 744), ('known', 740), ('april', 736), ('well', 735), ('high', 733), ('sect', 729), ('head', 723), ('go', 718), ('shogunate', 708), ('keihan', 707), ('jr', 706), ('rank', 704), ('maizuru', 696), ('river', 693), ('second', 691), ('domain', 691), ('side', 677), ('power', 669), ('jinja', 666), ('local', 664), ('ashikaga', 663), ('buddhist', 661), ('written', 656), ('department', 653), ('built', 645), ('important', 645), ('among', 639)]
"""