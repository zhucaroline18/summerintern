# Summer Intern

## WEEK 1- Basics and Preparation 
### Overview
| Day | Done |
|---|---|
|1 | read "Learning Python the Hard Way" |
|2 | learned markdown and git|
|3 | learned markdown|
|4 | learned git|
|5 | learned command line|
|6 | learned Python the Hard Way|
|7| learned Python the Hard Way|

### Resources 
* learning python with the book [Learn Python 3 the Hard Way](https://www.amazon.com/Learn-Python-Hard-Way-Introduction/dp/0134692888) 
* learning [Markdown](https://www.markdowntutorial.com/), here is [overview](https://www.markdownguide.org/basic-syntax/#paragraphs-1)
* learning [git](https://www.codecademy.com/learn/learn-git) (7 day free trial on Codecademy)
* learning basic command line functions with [videos](https://www.youtube.com/watch?v=MBBWVgE0ewk) and [codecademy](https://www.codecademy.com/learn/learn-the-command-line) (free trial)

### Findings/Notes
**Syntax things**
* Learn command line first 
* Use the free codecademy 7-day trial 
* Git tutorial is good but mainly take first 2, basic git workflow and how to backtrack in git ~ 3+ hours 
* Markdowntutorial ~ 15-20 minutes

**Python book**
* Finish exercises 1-44 (pages 1-188)
* Pay attention to modules

**Other** 
* Create a [github](https://github.com/) account 
* Create schedule similar to overview above to keep track of progress
* Practice git skills and push the schedule to your github

## WEEK 2 - Finish Python, Understanding Neural Networking
### Overview
| Day | Done |
|---|---|
|1 | Learned Python the Hard Way|
|2| Python the Hard Way and first and second video|
|3| Learn more Python, third and fourth video|
|4| Learn more Python, read websites|
|5| Learn SingleLinkedList and DoubleLinkedList in Python|
|6| Learn Dictionaries and start videos|
|7| Watch second video|

### Resources 
* Using [Learn More Python 3 the Hard Way](https://www.amazon.com/Learn-More-Python-Hard-Way/dp/0134123484)
* [3 Blue 1 Brown](https://www.youtube.com/channel/UCYO_jab_esuFRV4b17AJtAw) Video Resources 
    * [Introduction](https://www.youtube.com/watch?v=aircAruvnKk) (20 minutes)
    * [Gradient Descent, how NN learn](https://www.youtube.com/watch?v=IHZwWFHWa-w&list=RDCMUCYO_jab_esuFRV4b17AJtAw&index=3) (21 minutes)
    * [Backpropogation](https://www.youtube.com/watch?v=IHZwWFHWa-w&list=RDCMUCYO_jab_esuFRV4b17AJtAw&index=3) (13 minutes)
    * [Backpropogation Calculus](https://www.youtube.com/watch?v=tIeHLnjs5U8&list=RDCMUCYO_jab_esuFRV4b17AJtAw&index=4) (10 minutes)

* Supplemental Resources 
    * [Basics of Image Classification with PyTorch](https://heartbeat.fritz.ai/basics-of-image-classification-with-pytorch-2f8973c51864)
    * [Pytorch image classification on github](https://github.com/bentrevett/pytorch-image-classification)
    * [NN for beginners by beginners](https://towardsdatascience.com/neural-networks-for-beginners-by-beginners-6bfc002e13a2)

### Findings/Notes
Python Book 
* exercises 1-16 (pages 1-71) (technically not necessary for project but good to know)
* difficult to understand at first, exercises are longer and more advanced 

Videos 
* watch a few times, don't worry if don't really understand, just get basic general understanding 
* all about linear algebra and matrices
* keywords: weights, biases, backpropogation

## WEEK 3 - Start First Simple NN- Perceptron 
| Day | Done |
|---|---|
|1 | Perceptron coding train and matrix NumPy|
|2| Creating own perceptron|
|3| Watched 10.4 and 10.5 NN videos |
|4| break|
|5| coding train matrix video|
|6| coding train forward and backpropogation videos|
|7| starting Samson turorial neural network from scratch|

### Resources 
* [Coding Train Playlist](https://www.youtube.com/playlist?list=PLRqwX-V7Uu6aCibgK1PTWWu9by6XFdCfh)
* Week three [folder](./Week%3)
* [Samson Perceptron tutorial w/out pytorch](https://www.youtube.com/watch?v=w8yWXqWQYmU) (30 minutes)

### Findings/Notes

Coding Train Playlist 
* watch 10.1-10.5 for perceptron 
* watch 10.6-10.7 for matrix math which you will implement yourself with python 
* for actual nn, you can watch 10.12-10.15+ supplementally 
* use these videos to understand the perceptron and then implement yourself
* look at my perceptron for python version and learn by doing 

Samson 
* excellent tutorial of learning by doing  
* starting/beginning implementing own real neural network of recognizing mnist dataset (handwritten numbers) without pytorch, just understanding it.
* or follow my code implementation without jupiter but still learn by doing 

Rundown of perceptron 
* using the line x = y, determine whether a given point is above or below the line 
* outputting 1 for above, outputting -1 for below 


## WEEK 4
### Overview
| Day | Done |
|---|---|
|1 | building own neural network and meeting|
|2 | own neural network again|
|3 | trying to learn pytorch|
|4 | trying to learn more pytorch|
|5 | break|
|6 | pytorch videos|
|7 | pytorch videos|

### Resources 
* [MNIST Dataset](https://www.kaggle.com/oddrationale/mnist-in-csv)
* [Samson Perceptron tutorial w/out pytorch](https://www.youtube.com/watch?v=w8yWXqWQYmU) (30 minutes)
* [My MNIST code](./Week%204%20and%205/NeuralNetwork/neuralCode.py) w/out pytorch 
* [PyTorch tutorial Playlist](https://www.youtube.com/playlist?list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4)
* meeting powerpoint/recording? 

## Findings/Notes
* Samson tutorial uses jupiter notebook- can be confusing, instead can look at [my code](./Week%204%20and%205/NeuralNetwork/neuralCode.py)
* download pytorch

MNIST without pytorch summary
* 2 layers 
* use CSV data portion to load the data and test showing the image
* initialize weights and biases randomly 
* use ReLu loss function in feed forward algorithm 
* get prediction of current model with feedforward
* use softmax to see which number 1-10 it is most likely to be 
* in one_hot, set specific index to 1 to signify it is the right answer (labels)
* use backpropogation calculus to see how much to adjust the weights and biases of each layer
* then change the weights and biases accordingly 
* in train_and_predict, train the data with feedforward and back_prop functions, then use predict function to predict the answer
* see how many you got correct in the dataset using for loop

project 
* using convolutional neural network to identify faults in products 

## WEEK 5
### Overview 
| Day | Done |
|---|---|
|1 | Learning PyTorch|
|2 | Watching PyTorch videos and Learning more PyTorch|
|3 | javacert and more PyTorch Videos|
|4 | javacert and PyTorch|
|5 | PyTorch MNIST|
|6 | PyTorch MNIST and intro to project|
|7 | watching convolutional nn videos |

### Resources
* MNIST Dataset [download link](https://www.kaggle.com/oddrationale/mnist-in-csv)
* [getting familiar with MNIST with numpy](./Week%204%20and%205/MNIST.md)
* [PyTorch tutorial Playlist](https://www.youtube.com/playlist?list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4)
* MNIST with pytorch [code](./Week%204%20and%205/NeuralNetwork/mnist_with_torch.py) 

### Findings/Notes 

## WEEK 6
| Day | Done |
|---|---|
|1 | watch more convolutional nn videos|
|2 | questions on convolutional nn|
|3 | questions on convolutional nn|
|4 | break |
|5 | data loading |
|6 | data loading |
|7 | training |

### Resources 
* MNIST with pytorch code 
* Convolution nn videos 
* [Real Dataset](https://drive.google.com/file/d/1KVBfMk5IBij2u2JGyE-WKG6WkcX_oqOA/view?usp=sharing)

## WEEK 7
| Day | Done |
|---|---|
|1 | convolutional network lookover|
|2 | convolutional network questions answered|
|3 | testing the data and training|
|4 | getting accuracy |
|5 | fixing things and improving|
|6 | testing different|
|7 | testing |

## WEEK 8
|Day | Done|
|---|---|
|1 | documentation and organization|
