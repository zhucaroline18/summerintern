import random 

class Perceptron:
    def __init__(self, inputNodeCount):

        #question what is random uniform

        self.weights = [random.uniform(-1,1) for x in range (inputNodeCount)]
        self.bias = random.uniform(-1,1)
        self.learning_rate = 0.04

    def guess(self, inputs):
        sum = self.bias
        for i in range (len(self.weights)):
            sum += self.weights[i]*inputs[i]

        if (sum<0):
            return -1
        return 1

    def train(self, inputsAndTargetList):
        sum_error = 0
        for inputsAndTarget in inputsAndTargetList:

            ##question 

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

    #why set it to none 

    def __init__(self, x = None, y = None):
        if (x is None):
            x = random.uniform(0,300)
        if y is None:
            y = random.uniform(0,300)
        
        self.x = x
        self.y = y
        if x>y:
            self.label = 1
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