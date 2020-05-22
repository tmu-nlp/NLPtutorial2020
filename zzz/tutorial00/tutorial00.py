def count_words(text):
    words = text.split(' ')
    counter = {}
    for word in words:
        if word not in counter:
            counter[word] = 1
        else:
            counter[word] += 1
    counter = sorted(counter.items(), key=lambda x: x[0])
    return counter

def test(result, answer):
    for (r, a) in zip(result, answer):
        if r != a:
            return False
    return True

if __name__ == '__main__':
    input_fileame = 'data/wiki-en-train.word'
    # test_fileame = 'test/00-answer.txt'
    with open(input_fileame, 'r') as file:
        text = ' '.join([line.replace('\n', '') for line in file])
        # print(text)
        counter = count_words(text)

        for word, count in counter:
            print(word, count)

        # with open(test_fileame, 'r') as ans_file:
        #     answer = ''.join([line for line in file])
        #     result = '\n'.join([str(word) + ' ' + str(count) for word, count in counter])
        #
        #     if test(result, answer):
        #         print('Accepted.')
        #     else:
        #         print('Wrong answer.')