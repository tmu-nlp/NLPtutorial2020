from collections import defaultdict

counts = defaultdict(int)
context_counts = defaultdict(int)

train = "../../test/02-train-input.txt"
#train = "../../data/wiki-en-train.word"

model = "./model_file.txt"


def train_bigram(file):
    with open(file, encoding="utf-8") as tra_file:
        line = tra_file.readline()
        while(line):
            words = line.split()
            words.append("</s>")
            words.insert(0, "<s>")
            for i in range(len(words) - 1):
                counts[words[i]+" "+words[i+1]] += 1
                context_counts[words[i]] += 1
                counts[words[i+1]] += 1
                context_counts[""] += 1
            line = tra_file.readline()

def culc_prob(file):
    with open(file, "w", encoding="utf-8") as model:
        for ngram, count in counts:
            words = ngram.split()
            words[-1] = ""
            prob = count/context_counts[words[0]]
            model.write("{}\t{:.6f}\n".format(ngram, prob))

if __name__ == "__main__":
    train_bigram(train)
    counts = sorted(counts.items())
    culc_prob(model)