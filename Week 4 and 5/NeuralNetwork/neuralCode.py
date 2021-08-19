######################################
# CSV CODE 
######################################

from csv import reader
from matplotlib import pyplot as plt
import numpy as np

def load_csv(filename, max_rows):

    ##creating array of the data as strings, each row is an element
    ##create a list with max_rows elements. 

    with open(filename) as file:
        csv_reader = reader(file)
        ret = list()
        count = 0
        for row in csv_reader:

            if row is None:
                continue
            ret.append(row)
            count+=1
            if max_rows>0 and count>=max_rows:
                break
        return ret

def load_dataset(filename, max_rows):
    # using the csv list, create a list of labels (first element)
    ## as well as the list of datasets

    #First load the csv
    if max_rows>0:
        max_rows +=1
    csv_data = load_csv(filename, max_rows)

    # get rid of the first row because it's headers 
    csv_data = csv_data[1:]

    dataset = list()
    labels = list()

    # the label is the first element. parse into an int value
    # add to labels list 
    for raw_row in csv_data:
        label = int(raw_row[0])
        labels.append(label)

        # for the rest, create a list where you 
        # divide each element in csv by 255 to turn into grayscale value
        # add onto the dataset 
        # ???? does this append as a list?
        row = [int(col)/255.0 for col in raw_row[1:]]
        dataset.append(row)

    return dataset, labels

def show_image(dataset, labels, index):
    label = labels[index]

    # each index of dataset is a list with 28x28 elements
    # reshaping that list into a 28x28 matrix
    image = np.array(dataset[index]).reshape((28,28))

    # multiply each value by 255 to get grayscale 
    image = image*255

    #print the label so you know what it is 
    print(f'label={label}')

    #plot it in grayscale in the python drawing thing
    #first plot it in memory then show it on the screen 
    plt.gray()
    plt.imshow(image, interpolation='nearest')
    plt.show()

"""
if __name__=="__main__":

    #doing everything for specific index (row) of the file 
    dataset, labels = load_dataset('mnist_train.csv',10)
    index = 4
    print(labels[index])
    print(dataset[index])
    show_image(dataset, labels, index)
"""

######################################
#       NEURAL NETWORKING CODE       # 
######################################


def ReLu(Z):
    return np.maximum(Z,0)

def ReLu_deriv(Z):
    return Z>0

def softmax(Z):
    exp = np.exp(Z)
    sum = np.sum(exp)
    return exp/sum

class NeuralNetwork:
    def __init__(self, inputCount, hiddenCount, outputCount):
        # creating the nodes 
        self.inputCount = inputCount
        self.hiddenCount = hiddenCount
        self.outputCount = outputCount

        # initializing weights and biases randomly to between -1,1
        self.W1 = np.random.rand(hiddenCount, inputCount) - 0.5
        self.b1 = np.random.rand(hiddenCount,1)-0.5
        self.W2 = np.random.rand(outputCount, hiddenCount) - 0.5
        self.b2 = np.random.rand(outputCount, 1)- 0.5
        self.learning_rate = 0.02

    def feed_forward(self, X):
        #weighted sum + bias
        Z1 = self.W1.dot(X)+self.b1
        #convert to activation function 
        A1 = ReLu(Z1)
        #weighted Sum + bias for next layer 
        Z2 = self.W2.dot(A1) + self.b2

        #activation for final output (results of the forward prop)
        A2 = softmax(Z2)

        return Z1, A1, Z2, A2

    def one_hot(self, Y):
        #initialize list with outputCount elements and set to 0
        ret = [0 for x in range(self.outputCount)]

        #set specific y index to 1 (probably to say what it actually is)
        ret[Y]=1

        #shaping it into a vertical matrix form
        return np.array(ret).reshape(-1,1)

    def backward_prop(self, X, Z1, A1, Z2, A2, Y):
        one_hot_Y = self.one_hot(Y)

        dZ2 = A2 - one_hot_Y # matrix of diferences between result and actual 
        dW2 = dZ2.dot(A1.T) # transpose A1 and dot product with the difference to shift the weights
        db2 = dZ2 #change in bias
        dZ1 = self.W2.T.dot(dZ2)*ReLu_deriv(Z1) #function for the difference and after transposing the weights, dot product with difference in outcome
        dW1 = dZ1.dot(X.T) #transpose inputs and dot product with difference in results to find how much to adjust the weight
        db1 = dZ1

        #update weights and biases based on differences 

        self.W1 = self.W1 - self.learning_rate*dW1
        self.b1 = self.b1 - self.learning_rate*db1
        self.W2 = self.W2 - self.learning_rate * dW2
        self.b2 = self.b2 - self.learning_rate * db2


    #training using inputs X and expected Y 
    def train(self, X, Y):
        Z1, A1, Z2, A2 = self.feed_forward(X)
        self.backward_prop(X, Z1, A1, Z2, A2, Y)

    #predict answer for the input 
    def predict(self, X):
        Z1, A1, Z2, A2 = self.feed_forward(X)
        #A2 is matrix with 1 column, find index with max value and that's the prediction

        index = np.argmax(A2, 0)
        return index 

def train_and_predict():
    #only want to read small number of rows 
    train_count = 5000
    test_count = 1000

    #getting the dataset and labels enough for traincount and testcount
    dataset, labels = load_dataset('mnist_train.csv', train_count + test_count)

    brain = NeuralNetwork(28*28, 16, 10) #28*28 inputs, 10 hidden nodes, 10 ouputs
    for i in range(train_count):
        X = np.array(dataset[i]).reshape(-1,1) #get array and reshape into column 
        Y = labels[i] #expected value 

        #using the first train_count rows, train the brain, shifting the weights and biases
        brain.train(X,Y)

    correct_count = 0

    #counting how many are correct using only test data
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