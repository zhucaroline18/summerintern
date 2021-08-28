# Summer Intern

## Abstract 
The final purpose of this project was to create a convolutional neural network that could identify damage on a product given training images and labels and testing images and labels. First, I had to learn basics, then getting into actual neural networking and image identification neural network before I could get into convolutional neural networking. Though it was very intimidating at first, I started by watching neural networking videos, then slowly, it started to make sense when I actually tried to implement my very first perceptron. 

Basically, a neural network has a number of layers with a number of neurons which can be adjusted and through each layer, there are weights and baises which, after given feedback from the training data, will adjust accordingly to get the output closer to the actual label as possible. This is done through forward propogation, essentially using linear algebra and matrix multiplication with the weights and biases of each layer, as well as backpropogation, which is essentially keeping track of derivatives so you can use gradient decent calculus to find out how much you need to adjust the weights and biases by. As it turns out, machine learning it is much more machine than learning, and really just has a lot of math involved and using coding to do the brute fore of it. Neural Networking is based off the brain and its neurons and how it learns, and the computer is trying to mimic that kind of learning. 

After getting the basics of a neural network, I had to try to implement it myself, and that starts with the perceptron, the most simple neural network, which is just trying to predict if a given point is above or below a certain line, such as y = x. After that, I moved on to the MNIST dataset, a dataset of 27x27 pixel images of handrawn numbers. The goal is to classify each image a number 0-9. It was difficult using only numpy to do this because it was difficult to calculate the backpropogation and weight and bias adjustments when you have to do the calculus of gradient descent yourself. That's where PyTorch comes in. 

Pytorch was difficult to learn but in essence, it helps with the gradient descent and having to calculate how much to step the weights and biases with backpropogation because it has tensors which can store derivatives and make the steps for you. So I then learned how to implement the MNIST number classification with pytorch

A Convolutional Neural Network has quite a few other things to look at, as you have to classify each pixel as a background, or part of something. This can be done using filters and max pooling and a number of other things, but with the module used, it is actually not overly necessary to understand how it works, though certainly knowing how it works will help you understand what you are doing better. When using the deeplabv3_resnet50 it helps you to implement 50 layers and helps with the convolutional part of the neural network. The main thing really was dealing with data and how to pass it in and format it correctly. The rest is actually quite similar. I learned that you have to train the data quite a bit though to get accurate results and additionally, to try different loss functions or learning rates, and if it taks a long time, you can let it run overnight. 


## WEEK 1- Basics and Preparation 
### Overview
| Day | Done |
|---|---|
|1 | Read "Learning Python the Hard Way" |
|2 | Learned markdown and git|
|3 | Learned markdown|
|4 | Learned git|
|5 | Learned command line|
|6 | Learned Python the Hard Way|
|7 | Learned Python the Hard Way|

### Resources 
* Learning python with the book [Learn Python 3 the Hard Way](https://www.amazon.com/Learn-Python-Hard-Way-Introduction/dp/0134692888) 
* Learning [Markdown](https://www.markdowntutorial.com/), here is [overview guide](https://www.markdownguide.org/basic-syntax/#paragraphs-1)
* Learning [git](https://www.codecademy.com/learn/learn-git) (7 day free trial on Codecademy)
* Learning basic command line functions with [videos](https://www.youtube.com/watch?v=MBBWVgE0ewk) and [codecademy](https://www.codecademy.com/learn/learn-the-command-line) (free trial)
* Week 1 and 2 [folder](./Week%201%20and%202)

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
* Week 1 and 2 [folder](./Week%201%20and%202)

* Supplemental Resources 
    * [Basics of Image Classification with PyTorch](https://heartbeat.fritz.ai/basics-of-image-classification-with-pytorch-2f8973c51864)
    * [Pytorch image classification on github](https://github.com/bentrevett/pytorch-image-classification)
    * [NN for beginners by beginners](https://towardsdatascience.com/neural-networks-for-beginners-by-beginners-6bfc002e13a2)

### Findings/Notes
**Python Book** 
* Exercises 1-16 (pages 1-71) (technically not necessary for project but good to know)
* Difficult to understand at first, exercises are longer and more advanced than previous book 

**Videos** 
* Watch a few times, don't worry if don't really understand, just get basic general understanding 
* All about linear algebra and matrices
* Keywords: weights, biases, backpropogation



## WEEK 3 - Start First Simple NN- Perceptron 
| Day | Done |
|---|---|
|1 | Perceptron coding train and matrix NumPy|
|2| Creating own perceptron|
|3| Watched 10.4 and 10.5 NN videos |
|4| Break|
|5| Coding train matrix video|
|6| Coding train forward and backpropogation videos|
|7| Starting Samson turorial neural network from scratch|

### Resources 
* [Coding Train Playlist](https://www.youtube.com/playlist?list=PLRqwX-V7Uu6aCibgK1PTWWu9by6XFdCfh)
* [Samson NeuralNetwork tutorial w/out pytorch](https://www.youtube.com/watch?v=w8yWXqWQYmU) (30 minutes)
* Code: 
    * Week three [folder](./Week%203)
    * [Perceptron](./Week%203/perceptron/firstPerceptron.py)
    * [Self-matrix](./Week%203/perceptron/selfMatrices.py)

### Findings/Notes

**Coding Train Playlist**
* Watch 10.1-10.5 for perceptron 
* Watch 10.6-10.7 for matrix math which you will implement yourself with python 
* For actual nn (neural network), you can watch 10.12-10.15+ supplementally 
* Use these videos to understand the perceptron and then implement yourself
* Look at my perceptron for python version and learn by doing 

**Rundown of perceptron**
* using the line `y = x`, determine whether a given point is above or below the line 
* outputting 1 for above, outputting -1 for below 

**Samson**
* don't have to finish/start this week- mostly for next week 
* Excellent tutorial for MNIST neural network of learning by doing  
* Starting/beginning implementing own real neural network of recognizing mnist dataset (handwritten numbers) without pytorch, just understanding it.
* Or follow my code implementation without jupiter but still learn by doing 



## WEEK 4- Build MNIST NN without Pytorch then Learn Pytorch 
### Overview
| Day | Done |
|---|---|
|1 | Building own neural network and meeting|
|2 | Own neural network again|
|3 | Trying to learn pytorch|
|4 | Trying to learn more pytorch|
|5 | Break|
|6 | Pytorch videos|
|7 | Pytorch videos|

### Resources 
* [Week 4 plan](./Week%204%20and%205/week4plan.md)
* [MNIST Dataset](https://www.kaggle.com/oddrationale/mnist-in-csv)
* MNIST Dataset [download link](https://www.kaggle.com/oddrationale/mnist-in-csv)
* [Samson Perceptron tutorial w/out pytorch](https://www.youtube.com/watch?v=w8yWXqWQYmU) (30 minutes)
* [PyTorch tutorial Playlist](https://www.youtube.com/playlist?list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4)
* [Download](https://pytorch.org/) pytorch
* Code: 
    * Week 4 and 5 [folder](./Week%204%20and%205)
    * [My MNIST code](./Week%204%20and%205/NeuralNetwork/neuralCode.py) w/out pytorch 
* Meeting powerpoint/recording? 

## Findings/Notes

*[Download](https://pytorch.org/) pytorch

**Samson**
* Excellent tutorial for MNIST neural network of learning by doing  
* Starting/beginning implementing own real neural network of recognizing mnist dataset (handwritten numbers) without pytorch, just understanding it.
* Samson tutorial uses Jupyter notebook- can be confusing, instead can look at [my code](./Week%204%20and%205/NeuralNetwork/neuralCode.py)

**MNIST without pytorch summary**
* 1 hidden layer
* Use CSV data portion to load the data and test showing the image
* Initialize weights and biases randomly 
* Use `weights*x+bias` and `ReLu` activation function in feed forward algorithm 
* Get prediction of current model with feedforward
* Use `softmax` to see which number 1-10 it is most likely to be 
* In `one_hot`, set specific index to 1 to signify it is the right answer (labels)
* In Back Propogation calculation, find the derivative of loss over the the weights and bias
* Then change the weights and biases accordingly
* In `train_and_predict`, train the data with feedforward and back_prop functions, then use predict function to predict the answer
* See how many you got correct in the dataset using for loop

project 
* Using convolutional neural network (haven't learned yet) to identify faults in products 

## WEEK 5- MNIST NN with Pytorch 
### Overview 
| Day | Done |
|---|---|
|1 | Learning PyTorch|
|2 | Watching PyTorch videos and Learning more PyTorch|
|3 | More PyTorch Videos|
|4 | PyTorch|
|5 | PyTorch MNIST|
|6 | PyTorch MNIST and intro to project|
|7 | Watching convolutional nn videos |

### Resources
* [Week 5 plan](./Week%204%20and%205/week5plan.md)
* [PyTorch tutorial Playlist](https://www.youtube.com/playlist?list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4)
* Code: 
    * Week 4 and 5 [folder](./Week%204%20and%205)
    * [Getting familiar with MNIST with numpy](./Week%204%20and%205/MNIST.md)
    * [MNIST with pytorch](./Week%204%20and%205/NeuralNetwork/mnist_with_torch.py) 

### Findings/Notes 
* Follow along pytorch video exercises (1-10)

**MNIST with pytorch summary**
* Using torch's classes, construct model with the first layer of inputs and first hidden layer, first hidden layer and second hidden layer, then second hidden layer and number of outputs(10)
* Add the forward method that returns the prediction 
* Create the training and testing dataset from the MNIST download and transform them into tensors so you can keep track of derivatives
* Train the model by creating a model, optimizer with a set learning rate, and set loss function as CrossEntropyLoss
    * Use a train loader to load the inputs in for training
    * Epochs is how many times you use data to train the model 
    * After transforming image, use the model to get the result
    * Calculate the loss between result and label using loss function 
    * Using the optimizer, first clear the grad with `zero_grad()` then use back propogation based off loss 
    * Adjust weights and biases with `optimizer.step()`
    * Print the loss of every 100th to track the change of the loss- should be getting smaller 
    * Save the model so you can use it to test 
* Test the model first load the model from the training then pick an index you want to test- will show label and prediction so you can compare 
* To get accuracy, test all the images using the model to see how many correct by comparing prediction with label 
* You can check the dataset with `check_dataset` showing image, label, data, and shapes 

## WEEK 6- Convolutional NN and Start Project
| Day | Done |
|---|---|
|1 | Watch more convolutional nn videos|
|2 | Questions on convolutional nn|
|3 | Questions on convolutional nn|
|4 | Break|
|5 | Data loading |
|6 | Data loading |
|7 | Training |

### Resources 
* Convolutional neural network videos 
    * [Filters](https://www.youtube.com/watch?v=qPKsVAI_W6M)
    * [Max pooling](https://www.youtube.com/watch?v=pRWq_mtuppU&t=860s)
* [Real Dataset](https://drive.google.com/file/d/1KVBfMk5IBij2u2JGyE-WKG6WkcX_oqOA/view?usp=sharing)
* Code: 
    * [Week 6 plan](./Week%206%20and%207/week6plan.md)
    * [Learn image loading](./Week%206%20and%207/project/source/learn-img.py)
    * [Project code](./Week%206%20and%207/project/source/defect-detection.py)

### Notes
* Don't worry about implementing filters and max pooling etc, the library will do it for you so very similar to MNIST 
* Don't neglect data loading, actually putting together the data is difficult

**Learn Image Loading**
* Use `pyplot`, `torch`, `numpy`, and `torchvision.transforms`
* Identify the image you want to use and get it's path to label and source- same name, different folder
* Open the image and `src_image_data` turn into numpy array and divide by 255 to make it grayscale less than 1
* `label_image_data` becomes a numpy array 
* Create a plot with subplots so you can see the src_image and label_image side by side
* Transform the source image for processing- resizing, using interpolation, transform to tensor, and normalize the data
* Resize the label image 
* `defect_class` sets things over 0 to 1
* `backgroud_class` sets things that are 0 to 1
* `label_image_classes` becomes a tensor with background_class in first index, defect_class in second



## WEEK 7- Finish Project and test accuracy 
| Day | Done |
|---|---|
|1 | Convolutional network lookover|
|2 | Convolutional network questions answered|
|3 | Testing the data and training|
|4 | Getting accuracy |
|5 | Fixing things and improving|
|6 | Testing different|
|7 | Testing |

### Resources 
* Code: 
    * [Project code](./Week%206%20and%207/project/source/defect-detection.py)

### Notes

**Project Code**
* Get locations of all the folders 
* Set `batch_size` (images processed at once), `learning_rate` (subject to change based on experimentation), `epochs` (times you train your data using the training images)
* You should train it around 60 times honestly. You can leave it running overnight. 
* For class `ImageDataset` refer to Learn Image Loading above
* `SegmentationModel`
    * Use the existing `deeplabv3_resnet50` module for convolutional neural networks (no training, the two classes are background or defect)
    * Define a forward algorithm with the model to get results of model 
* `test_dataset`
    * Get the dataset from the ImageDataset already there 
    * Set src_image and label_image at specific index 
    * Show the plots to make sure the dataset is working correctly 
* `test_train_one` is a practice for `train_and_save_model` just to make sure things are working fine 
* `train_and_save_model`
    * Get label and image data using the ImageDataset class 
    * Use the train_loader to load the batches and stack randomly
    * Criterion is loss function, optimizer for cradients
    * Train with `train_loop`
* `train_loop`
    * Looping through all the epochs and batches in the train_loader, 
    * Get a prediction and with the targets, get the loss 
    * Use the optimizer and loss to do back propogation and step the weights and biases
    * Print out every 10th or 20th epoch to track progress of the loss 
    * Save it so you can use the trained model later 
    * Should try to train the model 12 or more epochs- will take a while, you can let it run overnight
* `load_checkpoint_and_train_and_save_model` just trains the already trained model by getting it from the checkpoint
* `test_model`
    * Get the trained model and testing dataset and loader 
    * You can print out the results for target_index and test different ones to see how your model is predicting
* `test_model_accuracy`
    * Calculating the accuracy of the model 
    * Using intersection(`logical_and`)/union(`logical_or`)
    * Storing all accuracy in an array so you can find the average, smallest, and largest accuracies

## WEEK 8- Documentation 
|Day | Done|
|---|---|
|1 | Documentation and organization|
