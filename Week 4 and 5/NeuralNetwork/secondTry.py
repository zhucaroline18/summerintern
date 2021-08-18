from csv import reader
from matplotlib import pyplot as plt
import numpy as np

def load_csv(filename, max_rows):
    with open(filename) as file:
        csv_reader = reader(file)
        ret = list()
        count = 0

        for row in csv_reader:
            if row is None:
                continue
            ret.append(row)
            count +=1
            if max_rows>0 and count>=max_rows:
                break
        return ret

def load_dataset(filename, max_rows):
    if max_rows>0:
        max_rows+=1

    csv_data = load_csv(filename, max_rows)

    csv_data = csv_data[1:]

    dataset = list()
    labels = list()

    for rawrow in csv_data:
        label = int(rawrow[0])
        labels.append(label)

        row = [int(col)/255 for col in rawrow[1:]]
        dataset.append(row)

    return dataset, labels

def ReLu(Z):
    return np.maximum(Z,0)

def ReLu_deriv(Z):
    return Z>0

def softmax(Z):
    exp = np.exp(Z)
    sum = np.sum(exp)
    return exp/sum

class NeuralNetwork:
    def __init__(self, inputCount, hidden1Count, hidden2Count, outputCount):
        self.inputCount = inputCount
        self.hiddenCount = hidden1Count
        self.outputCount = outputCount

        self.W1 = np.random.rand(hidden1Count, inputCount)*2 - 1
        self.b1 = np.random.rand(hidden1Count, 1)*2-1
        self.W2 = np.random.rand(hidden2Count, hidden1Count)*2-1
        self.b2 = np.random.rand(hidden2Count, 1)*2-1
        self.W3 = np.random.rand(outputCount, hidden2Count)*2-1
        self.b3 = np.random.rand(outputCount,1)*2-1

        self.learning_rate = 0.01
    
    def feed_forward(self,X):
        Z1 = self.W1.dot(X)+self.b1
        A1 = ReLu(Z1)
        Z2 = self.W2.dot(A1)+self.b2
        A2 = ReLu(Z2)
        Z3 = self.W2.dot(A2)+self.b2
        A3 = softmax(Z3)

        return Z1, A1, Z2, A2, Z3, A3
    
    def one_hot(self, Y):
        ret = [0 for x in range(self.outputCount)]
        ret[Y]=1
        return np.array(ret).reshape(-1,1)

    def backward_prop(self, X, Z1, A1, Z2, A2, Z3, A3, Y):
        one_hot_Y = self.one_hot(Y)

        dZ3 = A3 - one_hot_Y # matrix of diferences between result and actual 
        dW3 = dZ3.dot(A2.T) # transpose A1 and dot product with the difference to shift the weights
        db3 = dZ3 #change in bias
        dZ2 = self.W2.T.dot(dZ3)*ReLu_deriv(Z2) #function for the difference and after transposing the weights, dot product with difference in outcome
        dW2 = dZ2.dot(X.T) #transpose inputs and dot product with difference in results to find how much to adjust the weight
        db2 = dZ2
        dZ1 = self.W1.T.dot(dZ2)*ReLu_deriv(Z1)
        dW1 = dZ1.dot(X.T)
        db1 = dZ1

        self.W1 = self.W1 - self.learning_rate * dW1
        self.b1 = self.b1 - self.learning_rate * db1
        self.W2 = self.W2 - self.learning_rate * dW2
        self.b2 = self.b2 - self.learning_rate * db2
        self.W3 = self.W3 - self.learning_rate * dW3
        self.b3 = self.b3 - self.learning_rate * db3

    def train(self, X, Y):
        Z1, A1, Z2, A2, Z3, A3 = self.feed_forward(X)

        self.backward_prop(X, Z1, A1, Z2, A2, Z3, A3, Y)

    def predict(self, X):
        Z1, A1, Z2, A2, Z3, A3 = self.feed_forward(X)

        index = np.argmax(A2, 0)
        return index


def train_and_predict():
    train_count = 5000
    test_count = 1000

    dataset, labels = load_dataset('mnist_train.csv', train_count + test_count)

    brain = NeuralNetwork(28*28, 16, 16, 10)
    for i in range(train_count):
        X = np.array(dataset[i]).reshape(-1,1) 
        Y = labels[i]

        brain.train(X,Y)

    correct_count = 0

    for i in range(train_count, train_count + test_count):
        X = np.array(dataset[i]).reshape(-1,1)
        Y = labels[i]

        prediction = brain.predict(X)

        if prediction == Y:
            correct_count +=1
        else:
            print(f'incorrect prediction for data at index {i}. Prediction = {prediction}, Answer = {Y}')

    print(f'{correct_count} of {test_count} are correct')

if __name__=="__main__":
    train_and_predict()