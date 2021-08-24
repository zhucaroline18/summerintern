# Summer Intern

## essay thing 


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


## WEEK 4- Build MNIST NN without Pytorch then Learn Pytorch 
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
* MNIST Dataset [download link](https://www.kaggle.com/oddrationale/mnist-in-csv)
* [Samson Perceptron tutorial w/out pytorch](https://www.youtube.com/watch?v=w8yWXqWQYmU) (30 minutes)
* [My MNIST code](./Week%204%20and%205/NeuralNetwork/neuralCode.py) w/out pytorch 
* [PyTorch tutorial Playlist](https://www.youtube.com/playlist?list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4)
* [week 4 plan](./Week%204%20and%205/week4plan.md)
* meeting powerpoint/recording? 

## Findings/Notes
* Samson tutorial uses jupiter notebook- can be confusing, instead can look at [my code](./Week%204%20and%205/NeuralNetwork/neuralCode.py)
* download pytorch

MNIST without pytorch summary
* 1 hidden layer
* use CSV data portion to load the data and test showing the image
* initialize weights and biases randomly 
* use `weights*x+bias` and `ReLu` activation function in feed forward algorithm 
* get prediction of current model with feedforward
* use softmax to see which number 1-10 it is most likely to be 
* in one_hot, set specific index to 1 to signify it is the right answer (labels)
* In Back Propogation calculation, find the derivative of loss over the the weights and bias
* then change the weights and biases accordingly
* in train_and_predict, train the data with feedforward and back_prop functions, then use predict function to predict the answer
* see how many you got correct in the dataset using for loop

project 
* using convolutional neural network to identify faults in products 

## WEEK 5- MNIST NN with Pytorch 
### Overview 
| Day | Done |
|---|---|
|1 | Learning PyTorch|
|2 | Watching PyTorch videos and Learning more PyTorch|
|3 | more PyTorch Videos|
|4 | PyTorch|
|5 | PyTorch MNIST|
|6 | PyTorch MNIST and intro to project|
|7 | watching convolutional nn videos |

### Resources
* [week 5 plan](./Week%204%20and%205/week5plan.md)
* [getting familiar with MNIST with numpy](./Week%204%20and%205/MNIST.md)
* [PyTorch tutorial Playlist](https://www.youtube.com/playlist?list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4)
* MNIST with pytorch [code](./Week%204%20and%205/NeuralNetwork/mnist_with_torch.py) 

### Findings/Notes 
* follow along pytorch video exercises (1-10)

MNIST with pytorch summary 
* using torch's classes, construct model with the first layer of inputs and first hidden layer, first hidden layer and second hidden layer, then second hidden layer and number of outputs(10)
* add the forward method that returns the prediction 
* create the training and testing dataset from the MNIST download and transform them into tensors so you can keep track of derivatives
* train the model by creating a model, optimizer with a set learning rate, and set loss function as CrossEntropyLoss
    * use a train loader to load the inputs in for training
    * epochs is how many times you use data to train the model 
    * after transforming image, use the model to get the result
    * calculate the loss between result and label using loss function 
    * using the optimizer, first clear the grad with `zero_grad()` then use back propogation based off loss 
    * adjust weights and biases with `optimizer.step()`
    * print the loss of every 100th to track the change of the loss- should be getting smaller 
    * save the model so you can use it to test 
* test the model first load the model from the training then pick an index you want to test- will show label and prediction so you can compare 
* to get accuracy, test all the images using the model to see how many correct by comparing prediction with label 
* you can check the dataset with `check_dataset` showing image, label, data, and shapes 

## WEEK 6- Convolutional NN and Start Project
| Day | Done |
|---|---|
|1 | watch more convolutional nn videos|
|2 | questions on convolutional nn|
|3 | questions on convolutional nn|
|4 | break|
|5 | data loading |
|6 | data loading |
|7 | training |

### Resources 
* [week 6 plan](./Week%206%20and%207/week6plan.md)
* convolutional neural network videos 
    * [filters](https://www.youtube.com/watch?v=qPKsVAI_W6M)
    * [max pooling](https://www.youtube.com/watch?v=pRWq_mtuppU&t=860s)
* [Real Dataset](https://drive.google.com/file/d/1KVBfMk5IBij2u2JGyE-WKG6WkcX_oqOA/view?usp=sharing)
* [learn image loading](./Week%206%20and%207/project/source/learn-img.py)
* [project code](./Week%206%20and%207/project/source/defect-detection.py)

### Notes
* don't worry about implementing filters and max pooling etc, the library will do it for you so very similar to MNIST 
* don't neglect data loading, actually putting together the data is difficult

Learn Image Loading
* use `pyplot`, `torch`, `numpy`, and `torchvision.transforms`
* identify the image you want to use and get it's path to label and source- same name, different folder
* open the image and `src_image_data` turn into numpy array and divide by 255 to make it grayscale less than 1
* `label_image_data` becomes a numpy array 
* create a plot with subplots so you can see the src_image and label_image side by side
* transform the source image for processing- resizing, using interpolation, transform to tensor, and normalize the data
* resize the label image 
* defect_class sets things over 0 to 1
* backgroud_class sets things that are 0 to 1
* `label_image_classes` becomes a tensor with background_class in first index, defect_class in second



## WEEK 7- Finish Project and test accuracy 
| Day | Done |
|---|---|
|1 | convolutional network lookover|
|2 | convolutional network questions answered|
|3 | testing the data and training|
|4 | getting accuracy |
|5 | fixing things and improving|
|6 | testing different|
|7 | testing |

### Resources 
* [project code](./Week%206%20and%207/project/source/defect-detection.py)
* accuracy links 

### Notes

Project Code
* get locations of all the folders 
* set `batch_size` (images processed at once), `learning_rate` (subject to change based on experimentation), `epochs` (times you train your data using the training images)
* for class `ImageDataset` refer to Learn Image Loading above
* `SegmentationModel`
    * use the existing deeplabv3_resnet50 for convolutional neural networks (no training, the two classes are background or defect)
    * define a forward algorithm with the model to get results of model 
* `test_dataset`
    * get the dataset from the ImageDataset already there 
    * set src_image and label_image at specific index 
    * show the plots to make sure the dataset is working correctly 
* `test_train_one` is a practice for `train_and_save_model` just to make sure things are working fine 
* `train_and_save_model`
    * get label and image data using the ImageDataset class 
    * use the train_loader to load the batches and stack randomly
    * criterion is loss function, optimizer for cradients
    * train with `train_loop`
* `train_loop`
    * looping through all the epochs and batches in the train_loader, 
    * get a prediction and with the targets, get the loss 
    * use the optimizer and loss to do back propogation and step the weights and biases
    * print out every 10th or 20th epoch to track progress of the loss 
    * save it so you can use the trained model later 
    * should try to train the model 12 or more epochs- will take a while, you can let it run overnight
* `load_checkpoint_and_train_and_save_model` just trains the already trained model by getting it from the checkpoint
* `test_model`
    * get the trained model and testing dataset and loader 
    * you can print out the results for target_index and test different ones to see how your model is predicting
* `test_model_accuracy`
    * calculating the accuracy of the model 

## WEEK 8- Documentation 
|Day | Done|
|---|---|
|1 | documentation and organization|
