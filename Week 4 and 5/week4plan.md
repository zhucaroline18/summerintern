In Week4, we will create a Neural Network using MNIST dataset to recognize images. In Week5, you will use your Neural Network on the real dataset from the uncle's company.

## Neural Network using MNIST dataset
The YouTube video Building a neural network FROM SCRATCH and Samson's blog explained how to build a Neural Network using MNIST dataset. However, there is an error in the formula used by the above YouTube.com video and the blog as it did not have correct derivative_of_loss_over_w2. The correct formula is in Samson's jupyter notebook. When we do implementation, please use the correct formula from Samson's jupyter notebook.

### NumPy Tricks

Before you implement the NeuralNetwork, please get more familiar with NumPy. Especially some tricks that people will easily get unexpected results.

```python
# Please pay attention when we mix Matrix and Vector operation using numpy as there could be unexpected result
import numpy as np
m1 = np.array([[1,2,3],[4,5,6]])
v1 = np.array([1,2,3])
r1 = m1.dot(v1)
print(r1)

v2 = np.array([10,12])
print(r1 + v2)

r2 = m1.dot(v1.reshape(-1,1))
print(r2)

# (r2 + v2) is supposed to be same as (r1 + v2), however, they are different 
print(r2 + v2)
```
To avoid unexpected results, it's better to always convert a vector to a matrix for the operations. For example, we could use

```python
# convert list [1,2,3] to a matrix with one column.
np.array([1,2,3]).reshape(-1,1)
```

## Neural Network
We will build the Neural Network on top of the code that reads MNIST.
```python 
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
            count += 1
            if max_rows > 0 and count >= max_rows:
                break

        return ret

def load_dataset(filename, max_rows):
    # as we will skip the first row, we need to load one more row from the CSV file
    if max_rows > 0:
        max_rows += 1
    csv_data = load_csv(filename, max_rows)

    # skip the first row as it just has column headers
    csv_data = csv_data[1:]

    # Create empty list
    dataset = list()
    labels = list()
    for raw_row in csv_data:
        # call int(s) to convert a string to int
        label = int(raw_row[0])
        labels.append(label)

        # for the column index from 1 to 784, convert the string to integer,
        # then divide it by 255 to get float number represent the scale of the gray in range 0 to 1
        row = [int(col) / 255.0 for col in raw_row[1:]]
        dataset.append(row)
    return dataset, labels

# show image for the data at index
def show_image(dataset, labels, index):
    label = labels[index]

    # reshape data to 28 x 28 matrix
    image = np.array(dataset[index]).reshape((28, 28))
    # convert 0 to 1 scale to 0 to 255
    image = image * 255
    print(f'label={label}')

    # now we want to plot a gray image in memory
    plt.gray()
    plt.imshow(image, interpolation='nearest')
    # finally show it on the screen
    plt.show()

def ReLU(Z):
    return np.maximum(Z, 0)

def ReLU_deriv(Z):
    return Z > 0

def softmax(Z):
    exp = np.exp(Z)
    sum = np.sum(exp)
    return exp/sum


class NeuralNetwork:
    def __init__(self, inputCount, hiddenCount, outputCount):
        self.inputCount = inputCount
        self.hiddenCount = hiddenCount
        self.outputCount = outputCount
        
        # initialize
        self.W1 = np.random.rand(hiddenCount, inputCount) - 0.5
        self.b1 = np.random.rand(hiddenCount, 1) - 0.5
        self.W2 = np.random.rand(outputCount, hiddenCount) - 0.5
        self.b2 = np.random.rand(outputCount, 1) - 0.5
        self.learning_rate = 0.01
    
    def feed_forward(self, X):
        Z1 = self.W1.dot(X) + self.b1
        A1 = ReLU(Z1)
        Z2 = self.W2.dot(A1) + self.b2
        A2 = softmax(Z2)
        return Z1, A1, Z2, A2
    
    def one_hot(self, Y):
        ret = [0 for x in range(self.outputCount)]
        ret[Y] = 1
        # return matrix with 1 column
        return np.array(ret).reshape(-1,1)
    
    def backward_prop(self, X, Z1, A1, Z2, A2, Y):
        one_hot_Y = self.one_hot(Y)

        dZ2 = A2 - one_hot_Y
        dW2 = dZ2.dot(A1.T)
        db2 = dZ2
        dZ1 = self.W2.T.dot(dZ2) * ReLU_deriv(Z1)
        dW1 = dZ1.dot(X.T)
        db1 = dZ1

        # update the weights and bias
        self.W1 = self.W1 - self.learning_rate * dW1
        self.b1 = self.b1 - self.learning_rate * db1
        self.W2 = self.W2 - self.learning_rate * dW2
        self.b2 = self.b2 - self.learning_rate * db2


    # train the network with inputs X and the expected answer Y
    def train(self, X, Y):
        Z1, A1, Z2, A2 = self.feed_forward(X)
        self.backward_prop(X, Z1, A1, Z2, A2, Y)

    # predict the answer for the input
    def predict(self, X):
        Z1, A1, Z2, A2 = self.feed_forward(X)
        # A2 is a matrix with 1 column, find the index with max value for axis 0
        index = np.argmax(A2, 0)
        return index


def train_and_predict():
    # As there are a lot of rows, we only want to try to read small number of rows, such as 10
    train_count = 5000
    test_count = 1000
    dataset, labels = load_dataset('mnist_train.csv', train_count + test_count)

    brain = NeuralNetwork(28 * 28, 10, 10)
    for i in range(train_count):
        # input is a matrix with 1 column
        X = np.array(dataset[i]).reshape(-1,1)
        # Y is expected value
        Y = labels[i]
        brain.train(X, Y)
    
    correct_count = 0
    for i in range(train_count, train_count + test_count):
        # input is a matrix with 1 column
        X = np.array(dataset[i]).reshape(-1,1)
        Y = labels[i]
        prediction = brain.predict(X)
        if prediction == Y:
            correct_count += 1
        else:
            print(f'incorrect prediction for data at index={i}. Prediction={prediction}, Answer={Y}')
    
    print(f'{correct_count} of {test_count} are correct')

if __name__ == "__main__":
    train_and_predict()
```

The core concept is feed_forward and backward_propagation. The activation function determines the derivative, which is used to adjust the weights and bias.

You may wondering why ReLU_deriv is defined as
```python
def ReLU_deriv(Z):
    return Z > 0
```
It will return a matrix or vector with boolean value instead of number value. You could write a more easy-to-understand way to do it. But Python developers always mix the boolean values with numbers. The True is 1 and False is 0. Derivative of ReLU is 0 if x is less than 0, otherwise, it's 1. A more easy-to-understand way is something like
```python
def ReLU_deriv(Z):
    ret = list()
    for row in Z:
        new_row = list()
        for col in row:
            new_col = 1 if col > 0 else 0
            new_row.append(new_col)
        ret.append(new_row)
    return ret
```
The Python developer wrote it in Z > 0 way only because Python treats True as 1 and False as 0 in math operations.

The above code will report some pictures that it did not predict correctly. You could write another program to display those pictures so that you could check it. For example
```python
# assume that the above code is in the mnistwork2.py, you could import it
import mnistwork2

dataset, labels = mnistwork2.load_dataset('mnist_train.csv', 10000)

while True:
    index_str = input("Type Index:")
    if index_str is None or len(index_str) == 0:
        break
    index = int(index_str)
    mnistwork2.show_image(dataset, labels, index)
```
By this way, you could manually check those images that your code did not predict correctly. You will see that some images are very tricky.