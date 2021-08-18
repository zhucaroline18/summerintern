import firstPerceptron

def setup():
    p = firstPerceptron.Perceptron()
    inputs = [-1,0.5]
    guess = firstPerceptron.guess(inputs)
    print(guess)

if __name__=="__main__":
    setup()