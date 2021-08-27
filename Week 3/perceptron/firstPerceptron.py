import random 

# for perceptron, the machine will guess whether a point given to it is above or below a line. That's it. Very simple.
# you will have training data with inputs and target outputs 
# you will have testing data with inputs and target outputs 
# for this perceptron, just the simple line x = y

class Perceptron:
    def __init__(self, inputNodeCount):  ## how many nodes do you want for the input

        self.weights = [random.uniform(-1,1) for x in range (inputNodeCount)] #setting random weights the machine will learn to adjust it 
        self.bias = random.uniform(-1,1) #setting a random bias the machine will learn to adjust it 
        self.learning_rate = 0.04 #setting a learning rate, can adjust yourself to see what it does 

    def guess(self, inputs): 
        sum = self.bias 
        for i in range (len(self.weights)):
            sum += self.weights[i]*inputs[i] #sum is going to be the sum of all weights * (what's in the inputs)

        #make a guess as to if it's above or below the line 
        # -1 if below 
        # 1 if above 
        if (sum<0):
            return -1
        return 1 

    def train(self, inputsAndTargetList):
        sum_error = 0
        for inputsAndTarget in inputsAndTargetList:

            inputs = inputsAndTarget[0: len(self.weights)]
            target = inputsAndTarget[-1]
            prediction = self.guess(inputs)
            err = target-prediction
            sum_error += err*err
            self.bias = self.bias + self.learning_rate*err

            for i in range(len(self.weights)):

                #confusing formula    

                self.weights[i] = self.weights[i] + self.learning_rate * err * inputs[i]
        print(f"error = {sum_error}")

class Point:

    #labeling some data. it can label itself since this is a simple function of whether it's below the line or not.

    def __init__(self, x = None, y = None):
        if (x is None):
            x = random.uniform(0,300)
        if y is None:
            y = random.uniform(0,300)
        
        self.x = x
        self.y = y

        #if x>y, it is above the line x=y and returns 1
        if x>y:
            self.label = 1
        #otherwise, it is below the line x=y and returns -1
        else:
            self.label = -1


def train_and_predict():

    #what's this 

    trainingPoints = [Point() for i in range(100)]
    inputsAndTargetList = [[p.x, p.y, p.label] for p in trainingPoints]

    #create a brain, who only has 2 inputs

    brain = Perceptron(2)
    for i in range(10):
        brain.train(inputsAndTargetList)

    print(f"bias = {brain.bias}, weights={brain.weights}")

    testingPoints = [Point() for i in range(50)]
    correctCount = 0

    for pt in testingPoints:
        prediction = brain.guess([pt.x,pt.y])
        if (prediction != pt.label):
            print(f"Wrong prediction for ({pt.x}, {pt.y}")
        else:
            correctCount += 1

    print (f"{correctCount} out of {len(testingPoints)} are correct")

if __name__ == "__main__":
    train_and_predict()