import sys
import numpy as np

from collections import defaultdict

def count_features(ids, input_file):
    with open(input_file, "r", encoding="utf-8") as input_file:
        for line in input_file:
            _, words = line.strip().split("\t")
            words = words.split(" ")
            for word in words:
                ids["UNI:"+word]

class NeuralNetwork:
    def __init__(self, input_nodes, lr):
        self.inodes = input_nodes
        self.hnodes = 2
        self.onodes = 1
        self.lr = lr

        self.w_ih = np.random.rand(self.hnodes, self.inodes)
        self.w_ho = np.random.rand(self.onodes, self.hnodes)
        self.b_ih = np.random.rand(self.hnodes)
        self.b_ho = np.random.rand(self.onodes)

        self.phi = [0,0,0]
        self.delta_ih = [0,0]
        self.delta_ho = [0]

        self.ids = defaultdict(lambda:len(self.ids))

    def count_features(self, input_file):
        for line in input_file:
            _, words = line.strip().split("\t")
            words = words.split(" ")
            for word in words:
                self.ids["UNI:"+word]

    def forward_nn(self, phi_0):
        self.phi[0] = phi_0
        self.phi[1] = np.tanh(np.dot(self.w_ih,self.phi[0]) + self.b_ih)
        self.phi[2] = np.tanh(np.dot(self.w_ho,self.phi[1]) + self.b_ho)

    def backward_nn(self, y_ans):
        self.delta_ho[0] = (y_ans - self.phi[2]) * (1-self.phi[2]**2)
        a = np.dot(self.delta_ho[0],self.w_ho)
        self.delta_ih[0] = a[0] * (1 - self.phi[1][0]**2)
        self.delta_ih[1] = a[1] * (1 - self.phi[1][1]**2)

    def predict_one(self, phi):
        score = np.dot(self.w, phi)
        if score[0] >= 0:
            return 1
        else:
            return -1

    def create_features(self, x):
        phi = np.zeros(self.inodes)
        words = x.split()
        for word in words:
            phi[self.ids["UNI:" + word]] += 1
        return phi
    
    def create_features_for_test(self, x):
        phi = np.zeros(self.inodes)
        words = x.split()
        for word in words:
            if("UNI:" + word in self.ids):
                phi[self.ids["UNI:" + word]] += 1
        return phi
    
    def update_weights(self):
        self.w_ih += self.lr * np.outer(self.delta_ih[0],self.phi[0])
        self.b_ih += self.lr * self.delta_ih[0]
        self.w_ih += self.lr * np.outer(self.delta_ih[1], self.phi[0])
        self.b_ih += self.lr * self.delta_ih[1]
        self.w_ho += self.lr * np.outer(self.delta_ho[0],self.phi[1])
        self.b_ho += self.lr * self.delta_ho[0]

    def train(self,input_file,output_file):
        with open(input_file, "r", encoding="utf-8") as input_file:
                for line in input_file:
                    label, words = line.strip().split("\t")
                    label = int(label)
                    p_0 = self.create_features(words)
                    self.forward_nn(p_0)
                    self.backward_nn(label)
                    self.update_weights()
    
    def test(self,input_file,output_file):
        with open(input_file, "r", encoding="utf-8") as input_file,\
             open(output_file, "w", encoding="utf-8") as output_file:
                for line in input_file:
                    words = line.strip()
                    p_0 = self.create_features_for_test(words)
                    self.forward_nn(p_0)
                    a=0
                    print(self.phi[2])
                    if(self.phi[2]<0):
                        a = -1
                    else:
                        a = 1
                    output_file.write(f'{a}\t{line}')
        
if __name__ == "__main__" :
    ids = defaultdict(lambda:len(ids))
    count_features(ids,sys.argv[1])
    NN = NeuralNetwork(len(ids),0.01)
    NN.train(sys.argv[1],sys.argv[2])
    NN.test(sys.argv[3],sys.argv[4])