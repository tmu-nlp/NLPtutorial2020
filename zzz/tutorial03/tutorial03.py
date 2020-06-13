import math

from zzz.tutorial02.tutorial02 import Ngram

PATH = '/Users/zz_zhang/勉強会/NLPチュートリアル/NLPtutorial2020/data/'
TRAIN_FILENAME = 'wiki-ja-train.word'
TEST_FILENAME = 'wiki-ja-test.txt'
TRAIN_ANSWER_FILENAME = 'wiki-ja-result.word'
MODEL_FILENAME = '/Users/zz_zhang/勉強会/NLPチュートリアル/NLPtutorial2020/data/big-ws-model.txt'
N = 1


class WordSplitter:
    def __init__(self, model: Ngram):
        self.model = model

    def viterbi(self, text: str):
        result = []
        for line in text.split('\n'):
            best_score = [0]
            best_edge = [None]
            for (index_e, word_end) in enumerate(line):
                best_score.append(0x3fffffff)
                best_edge.append(None)
                for (index_b, word_begin) in enumerate(line[:index_e + 1]):
                    word = line[index_b: index_e + 1]
                    if word in self.model.p or len(word) == 1:
                        temp_score = best_score[index_b] - math.log(self.model.p[word], 2)
                        if temp_score < best_score[-1]:
                            best_score[-1] = temp_score
                            best_edge[-1] = (index_b, index_e)

            # print(best_edge)

            next_edge = best_edge[-1]
            sub_res = []
            while next_edge is not None:
                start_index = next_edge[0]
                end_index = next_edge[1]
                sub_res.append(line[start_index: end_index + 1])
                next_edge = best_edge[start_index]
            sub_res.reverse()
            # print(sub_res)
            result.append(sub_res)
        return result


if __name__ == '__main__':
    uni_gram = Ngram(1)
    bi_gram = Ngram(2)
    splitter = WordSplitter(uni_gram)

    uni_gram.load(MODEL_FILENAME)

    with open(PATH + TRAIN_FILENAME) as file:
        text = ''
        text = ''.join([line for line in file])
        # print(text)
        # uni_gram.train(text)
        # bi_gram.train(text)

        uni_gram.test(text, smoothing='linear', linear_lambda=[0.95, 0.05])
        # bi_gram.test(text, smoothing='witten_bell', linear_lambda=[0.05, 0.1, 0.85])

    with open(PATH + TEST_FILENAME) as file:
        text = ''
        text = ''.join([line for line in file])

        # for (word, prob) in bi_gram.p.items():
        #     print(word, prob)

        splitted = splitter.viterbi(text)

        with open(TRAIN_ANSWER_FILENAME, 'w') as answer_file:
            for line in splitted:
                answer_file.write(' '.join(line) + '\n')

"""
Evaluation:
    =====01=====
    Model: uni_gram
        Training text: wiki-ja-train.word
        Smoothing method: linear
        Smoothing lambdas: [0.05, 0.95]
    Result:
        Sent Accuracy: 23.81% (20/84)
        Word Prec: 71.88% (1943/2703)
        Word Rec: 84.22% (1943/2307)
        F-meas: 77.56%
        Bound Accuracy: 86.30% (2784/3226)

    =====02=====
    Model: uni_gram
        Training text: wiki-ja-train.word
        Smoothing method: linear
        Smoothing lambdas: [0.5, 0.5]
    Result:
        Sent Accuracy: 23.81% (20/84)
        Word Prec: 71.95% (1944/2702)
        Word Rec: 84.27% (1944/2307)
        F-meas: 77.62%
        Bound Accuracy: 86.33% (2785/3226)
        
    =====03=====
    Model: uni_gram
        Training text: wiki-ja-train.word
        Smoothing method: linear
        Smoothing lambdas: [0.95, 0.05]
    Result:
        Sent Accuracy: 21.43% (18/84)
        Word Prec: 71.25% (1908/2678)
        Word Rec: 82.70% (1908/2307)
        F-meas: 76.55%
        Bound Accuracy: 85.83% (2769/3226)

    =====04=====
    Model: bi_gram
        Training text: wiki-ja-train.word
        Smoothing method: linear
        Smoothing lambdas: [0.05, 0.1, 0.85]
    Result:
        Sent Accuracy: 21.43% (18/84)
        Word Prec: 71.25% (1908/2678)
        Word Rec: 82.70% (1908/2307)
        F-meas: 76.55%
        Bound Accuracy: 85.83% (2769/3226)
        
    =====05=====
    Model: bi_gram
        Training text: wiki-ja-train.word
        Smoothing method: witten-bell
        Smoothing lambdas: None
    Result:
        Sent Accuracy: 23.81% (20/84)
    Word Prec: 71.88% (1943/2703)
    Word Rec: 84.22% (1943/2307)
    F-meas: 77.56%
    Bound Accuracy: 86.30% (2784/3226)

    =====06=====
    Model: uni_gram
        Model: big-ws-model.txt
        Smoothing method: linear
        Smoothing lambdas: [0.05, 0.95]
    Result:
        Sent Accuracy: 21.43% (18/84)
        Word Prec: 71.25% (1908/2678)
        Word Rec: 82.70% (1908/2307)
        F-meas: 76.55%
        Bound Accuracy: 85.83% (2769/3226)
"""