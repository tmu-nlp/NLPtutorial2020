import sys
import collections
import math


STATUS = 'train'
PATH = '/Users/zz_zhang/勉強会/NLPチュートリアル/NLPtutorial2020/test/'
INPUT_FILE_NAME = '01-train-input.txt'
ANSWER_FILE_NAME = '01-train-answer.txt'
MODEL_FILE_NAME = 'tutorial01.model'
LAMBDA_1 = 0.95
LAMBDA_UNK = 1 - LAMBDA_1
V = 1000000
W = 0
H = 0

def train_unigram(train_file):

    gram_counter = collections.Counter()
    total_length = 0
    for line in train_file:
        temp = line.replace('\n', '').split(' ') + ['</s>']
        gram_counter.update(temp)
        total_length += len(temp)
    # print(total_counter, gram_counter)
    gram_counter = dict(sorted(gram_counter.items()))

    with open(MODEL_FILE_NAME, 'w') as model_file:
        res = ''
        for word, count in gram_counter.items():
            temp = word + ' ' + str(float(count) / total_length) + '\n'
            res += temp
            model_file.write(temp)

        print('Model training finished, written in {}.'.format(MODEL_FILE_NAME))
        return res
    print('Model training failed.')
    return -1


def test_unigram(test_file):
    with open(MODEL_FILE_NAME) as model_file:
        model = {}
        for line in model_file:
            temp = line.replace('\n', '').split(' ')
            model[temp[0]] = float(temp[-1])
        model = collections.Counter(model)

        entropy = 0.0
        coverage = 0.0
        total_length = 0
        for line in test_file:
            words = line.replace('\n', '').split(' ') + ['</s>']
            total_length += len(words)

            # calc entropy
            for word in words:
                p = LAMBDA_1 * model[word] + LAMBDA_UNK / V
                entropy -= math.log(p, 2)

                coverage += 1 if model[word] > 0 else 0

        entropy /= total_length
        coverage /= total_length
        # print(entropy, coverage)
        return 'entropy = ' + str(entropy) + '\n' + 'coverage = ' + str(coverage) + '\n'


def test(result, answer):
    detla = 1e-5
    items_r = result.replace('\n', ' ').split(' ')
    items_a = answer.replace('\n', ' ').replace('\t', ' ').split(' ')

    if len(items_a) != len(items_r):
        return False
    for item_r, item_a in zip(items_r, items_a):
        try:
            r = float(item_r)
            a = float(item_a)
            if abs(r - a) > detla:
                return False
        except:
            if item_r != item_a:
                return False
    return True


if __name__ == '__main__':
    if len(sys.argv) >= 2:
        STATUS = sys.argv[1]

    # STATUS = 'test'
    print(STATUS, 'mode.')
    if STATUS == 'train':
        func = train_unigram
    else:
        func = test_unigram
        INPUT_FILE_NAME = '01-test-input.txt'
        ANSWER_FILE_NAME = '01-test-answer.txt'

    with open(PATH + INPUT_FILE_NAME) as file:
        res = func(file)

        ans = ''
        with open(PATH + ANSWER_FILE_NAME) as ans_file:
            for line in ans_file:
                if len(line) > 1 and '#' not in line:
                    ans += line
            if test(res, ans):
                print('Accept')
            else:
                print('Wrong Answer')

