from collections import defaultdict
import numpy as np
import dill
import random

def create_one_hot(id_, size):
    vec = np.zeros(size)
    vec[id_] = 1

    return vec

def find_best(p):
    # find index of y with highest probability distribution
    y = 0
    for i in range(1, len(p)):
        if p[i] > p[y]:
            y = i

    return y


def initialize_net_randomly(input_size, output_size, hidden_size):
    # randomly initialize network
    w_rx = (np.random.rand(hidden_size, input_size) - 0.5)/5 # weight - input & hidden
    b_r = (np.random.rand(hidden_size) - 0.5)/5 # bias - hidden
    w_rh = (np.random.rand(hidden_size, hidden_size) - 0.5)/5 # weight - last hidden & current hidden
    w_oh = (np.random.rand(output_size, hidden_size) - 0.5)/5 # weight - hidden & output
    b_o = (np.random.rand(output_size) - 0.5)/5 # bias - output

    return w_rx, b_r, w_rh, w_oh, b_o


def initialize_delta(input_size, output_size, hidden_size):
    # randomly initialize delta
    dw_rx = np.zeros((hidden_size, input_size))
    dw_rh = np.zeros((hidden_size, hidden_size))
    db_r = np.zeros(hidden_size)
    dw_oh = np.zeros((output_size, hidden_size))
    db_o = np.zeros(output_size)

    return dw_rx, dw_rh, db_r, dw_oh, db_o


def forward_rnn(net, x):
    input_size = len(x)
    h = [np.ndarray for _ in range(input_size)]  # hidden layers (at time t)
    p = [np.ndarray for _ in range(input_size)]  # output probability distributions (at time t)
    y = [np.ndarray for _ in range(input_size)]  # # output values (at time t)
    
    w_rx, b_r, w_rh, w_oh, b_o = net # load initialized network
    
    for t in range(input_size):
        if t > 0:
            h[t] = np.tanh(np.dot(w_rx, x[t]) + np.dot(w_rh, h[t-1]) + b_r) # hidden size for each time step
        else:
            h[t] = np.tanh(np.dot(w_rx, x[t]) + b_r)
        p[t] = np.tanh(np.dot(w_oh, h[t]) + b_o) # calculating probability distribution
        y[t] = find_best(p[t]) # get y with highest probability

    return h, p, y


def gradient_rnn(net, x, h, p, y, input_size, output_size, hidden_node):
    delta = initialize_delta(input_size, output_size, hidden_node)
    dw_rx, dw_rh, db_r, dw_oh, db_o = delta
    w_rx, b_r, w_rh, w_oh, b_o = net

    delta_r_ = np.zeros(len(b_r))  # error from the following time step
    for t in range(len(x))[::-1]:
        delta_o_ = y[t] - p[t]  # output error
        
        # output gradient
        dw_oh += np.outer(delta_o_, h[t])
        db_o += delta_o_

        if t == len(x) - 1:
            delta_r = np.dot(delta_o_, w_oh)
        else:
            delta_r = np.dot(delta_r_, w_rh) + np.dot(delta_o_, w_oh)  # backprop
        delta_r_ = delta_r * (1 - h[t]**2)  # tanh gradient

        # hidden gradient
        dw_rx += np.outer(delta_r_, x[t])
        db_r += delta_r_

        if t != 0:
            dw_rh += np.outer(delta_r_, h[t-1])

    return dw_rx, dw_rh, db_r, dw_oh, db_o

def update_weights(net, delta, lam):
    dw_rx, dw_rh, db_r, dw_oh, db_o = delta
    w_rx, b_r, w_rh, w_oh, b_o = net

    w_rx += lam * dw_rx
    w_rh += lam * dw_rh
    b_r += lam * db_r
    w_oh += lam * dw_oh
    b_o += lam * db_o


if __name__ == '__main__':
    # epoch = 10
    # hidden_node = 64
    lam = 0.006 
    epoch = 10
    hidden_node = 256

    train_file = 'wiki-en-train.norm_pos'
    # train_file = '05-train-input.txt'

    word_ids = defaultdict(lambda: len(word_ids))
    tag_ids = defaultdict(lambda: len(tag_ids))

    # create dictionary for words and tags
    for line in open(train_file, 'r'):
        line = line.rstrip()
        word_tags = line.split(' ')
        for word_tag in word_tags:
            word, tag = word_tag.split('_')
            word_ids[word]
            tag_ids[tag]
    
    input_size = len(word_ids)
    output_size = len(tag_ids)

    # one-hot vector for feature (x) and label (y) pair
    feat_label = []  
    for line in open(train_file, 'r'):
        words = []  # initialize one-hot vector for words
        tags = []  # initialize one-hot vector for tags

        line = line.rstrip()
        word_tags = line.split(' ')
        for word_tag in word_tags:
            word, tag = word_tag.split('_')
            words.append((create_one_hot(word_ids[word], input_size)))
            tags.append((create_one_hot(tag_ids[tag], output_size)))
        feat_label.append((words, tags))

    # network initialization
    net = initialize_net_randomly(input_size, output_size, hidden_node)

    # train model
    for _ in range(epoch):
        for words, tags in feat_label:
            h, p, tags_predict = forward_rnn(net, words)
            delta = gradient_rnn(net, words, h, p, tags, input_size, output_size, hidden_node)
            update_weights(net, delta, lam)

    # save model
    dill.dump(net, open('network_weight_2', 'wb'))
    dill.dump(word_ids, open('word2ids_2', 'wb'))
    dill.dump(tag_ids, open('tag2ids_2', 'wb'))