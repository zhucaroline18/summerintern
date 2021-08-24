import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from PIL import Image

import numpy as np
import os
import matplotlib.pyplot as plt

current_location = os.path.dirname(__file__)
train_src_image_folder = os.path.join(current_location, "../dataset/train_val_image_label/train_image_label/srcImg")
train_label_image_folder = os.path.join(current_location, "../dataset/train_val_image_label/train_image_label/label")
test_src_image_folder = os.path.join(current_location, "../dataset/train_val_image_label/val_image_label/srcImg")
test_label_image_folder = os.path.join(current_location, "../dataset/train_val_image_label/val_image_label/val_label")
 
batch_size = 2
image_width = 716//4
image_height = 920//4
learning_rate = 0.0001
num_epochs = 1

class ImageDataset(Dataset):

    #finding the folders and getting rid of extras not in batches
    def __init__(self, src_image_folder, label_image_folder):
        super(ImageDataset, self).__init__()
        self.src_image_folder = src_image_folder
        self.label_image_folder = label_image_folder
        self.src_image_names = os.listdir(src_image_folder)
        mod = len(self.src_image_names)%batch_size
        self.src_image_names = self.src_image_names[0:len(self.src_image_names)-mod]
        #print(len(self.src_img_names))

    def __len__ (self):
        return len(self.src_image_names)

    def __getitem__(self, index):
        src_img_loc = os.path.join(self.src_image_folder, self.src_image_names[index])
        src_image = Image.open(src_img_loc).convert("RGB")

        #changing the image so we can process it through deeplab 

        preprocess = transforms.Compose([
                                            transforms.Resize((image_height, image_width), transforms.InterpolationMode.BICUBIC),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        ])

        src_image_data = preprocess(src_image)


        trf_resize = transforms.Resize((image_height, image_width), transforms.InterpolationMode.BICUBIC)
        label_img_loc = os.path.join(self.label_image_folder, self.src_image_names[index])
        label_image = Image.open(label_img_loc).convert("L")
        label_image_resized = trf_resize(label_image)

        label_image_data = np.array(label_image_resized)

        defect_class = np.zeros_like(label_image_data)
        defect_class[label_image_data>0]=1.0

        background_class = np.zeros_like(label_image_data)
        background_class[label_image_data == 0]= 1.0

        label_image_classes = torch.tensor([background_class, defect_class])

        return src_image_data, label_image_classes


class SegmentationModel(nn.Module):
    def __init__(self):
        super(SegmentationModel, self).__init__()
        self.d1 = models.segmentation.deeplabv3_resnet50(pretrained=False, progress = False, num_classes =2)

    def forward(self, x):
        results = self.d1(x)
        ret = results["out"]
        return ret

def test_dataset():
    dataset = ImageDataset(train_src_image_folder, train_label_image_folder)
    src_image, label_image =  dataset[10]
    print (src_image)
    print(src_image.shape)
    print(label_image)
    print (label_image.shape)

    fig, axarr = plt.subplots(2,3)

    axarr[0,0].imshow(src_image[0], cmap = "gray")
    axarr[0,1].imshow(src_image[1], cmap = "gray")
    axarr[0,2].imshow(src_image[2], cmap = "gray")

    axarr[1,0].imshow(label_image[0], cmap = "gray")
    axarr[1,1].imshow(label_image[1], cmap = "gray")
    plt.show()

def test_train_one():
    train_dataset = ImageDataset(train_src_image_folder, train_label_image_folder) #getting the label and image data using method defined earlier 
    train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True) #???
    model = SegmentationModel() # creating a new segmentation model 

    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate) #making it easier to work with 

    train_iter = iter(train_loader) #??? maybe establishing something that will look through the data one by one 
    inputs, labels = train_iter.next() #getting the data of the first dataset 
    outputs = model.forward(inputs) #going forward and getting the prediction 

    #printing out the shapes to make sure they are correct 
    print(outputs.shape)
    print(labels.shape)

    targets = labels.float() # converting labels into floats but kind of unnecessary 
    criterion = nn.BCEWithLogitsLoss() # loss function 
    loss = criterion(outputs, targets) # finding the loss 
    print(loss)

    optimizer.zero_grad() # clear the derivatives
    loss.backward() # backwards propogation 
    optimizer.step() # changing the weights and biases 

def train_loop(train_loader, model, criterion, optimizer):
    n_total_steps = len(train_loader)
    for epoch in range(num_epochs): #looping through the number of rows 
        for batch_index, (images, labels) in enumerate(train_loader): #what is enumerate???
            predicts = model(images) #gets prediciton with the segmentation model passed in

            targets = labels.float() #making the labels into floats
            loss = criterion(predicts, targets) # finding the loss

            optimizer.zero_grad() #resetting the gradients
            loss.backward()
            optimizer.step() #adjusting the weights and biases 

            if batch_index %10==0: #print every 20th epoch, step, and loss
                print(f"epoch {epoch}, step {batch_index} / {n_total_steps}, loss={loss}")

    torch.save(model, os.path.join(current_location, "saved-deeplab.pth")) #saving the training so can train again later
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    } 

    torch.save(state, os.path.join(current_location, "saved-deeplab-checkpoint.pth"))


def train_and_save_model():
    train_dataset = ImageDataset(train_src_image_folder, train_label_image_folder) #getting the label and image data using method defined previously
    train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True) # loading the data into batches and stacking randomly

    model = SegmentationModel()

    criterion = nn.BCEWithLogitsLoss() #loss function 
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate) #creating optimizer with gradients
    train_loop(train_loader, model, criterion, optimizer) #training it in a loop 


def load_checkpoint_and_train_and_save_model():
    train_dataset = ImageDataset(train_src_image_folder, train_label_image_folder)
    train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)

    model = SegmentationModel()

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    checkpoint = torch.load(os.path.join(current_location, "saved-deeplab-checkpoint.pth"))
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    train_loop(train_loader, model, criterion, optimizer)


def test_model():
    test_dataset = ImageDataset(test_src_image_folder, test_label_image_folder) #getting the label and image data using method defined previously 
    test_loader = DataLoader (dataset = test_dataset, batch_size = 1, shuffle = False) # loading data into batches of one and just straight since not adjusting
    model = torch.load(os.path.join(current_location, "saved-deeplab.pth")) ### maybe putting it in a place???
    
    model.eval() # don't use gradients or keep track of derivatives- much faster
    criterion = nn.BCEWithLogitsLoss() #loss function 
    

    with torch.no_grad(): ###???
        target_index = 174
        index = 0

        for image, label in test_loader:
            if index == target_index: # checking the target index 
                output = model(image) #getting result through the model 
                target = label.float() #converting label to float in order to calculate loss 
                loss = criterion (output, target) #using loss function to calculate how much was correct 
                print (f"loss = {loss}")

                output_raw = output.squeeze(0) 

                output_background = output_raw[0] #first element of array is background
                output_defect = output_raw[1] #second element is which ones are defect

                background_img = transforms.ToPILImage()(output_background) # transforming the image into something you can see?"
                defect_img = transforms.ToPILImage()(output_defect)

                output_label = output_raw.argmax(dim = 0) ###??? finidng the largest argument to see if there's a defect or not?
                print (output_label.shape)#checking the shape
                print(output_label)

                f, axarr = plt.subplots(2,3)
                image_data = image.squeeze(0)[0]
                label_data = label.squeeze(0)[1] #???? making the image back to regualr size??

                

                axarr[0,0].imshow(image_data, cmap = "gray") #show image in grayscale 
                axarr[0,1].imshow(torch.sigmoid(output_raw[0]), cmap = "gray") #background
                axarr[0,2].imshow(torch.sigmoid(output_raw[1]), cmap = "gray") #defect

                axarr[1,1].imshow(output_label, cmap = "gray")
                axarr[1,2].imshow(label_data, cmap = "gray")
                plt.show()

            index +=1
'''
def get_accuracy_of_model():
    test_dataset = ImageDataset(test_src_image_folder, test_label_image_folder) #getting the label and image data using method defined previously 
    test_loader = DataLoader (dataset = test_dataset, batch_size = 1, shuffle = False) # loading data into batches of one and just straight since not adjusting
    model = torch.load(os.path.join(current_location, "saved-deeplab.pth")) ### maybe putting it in a place???
    
    model.eval()
    criterion = nn.BCEWithLogitsLoss() #loss function 
    
    correct = 0

    with torch.no_grad(): 
        for image, label in test_loader:
            output = model(image) #getting result through the model 
            target = label.float() #converting label to float in order to calculate loss 
            loss = criterion (output, target) #using loss function to calculate how much was correct 
            
            if (loss<=0.02):
                correct +=1

    print (f"{correct} correct out of {test_dataset.__len__}")
'''

def test_model_accuracy():
    test_dataset = ImageDataset(test_src_image_folder, test_label_image_folder) #getting test data which is formatted and everything
    test_loader = torch.utils.data.DataLoader(dataset= test_dataset, batch_size = 1, shuffle = False) #loading the data one by one in same order

    model = torch.load(os.path.join(current_location, "saved-deeplab.pth")) #loading the locaiton of the deeplab and all the weights and biases
    model.eval() #setting it to evaluation mode so it doesn't keep track of weights and biases and takes a long time
    criterion = nn.BCEWithLogitsLoss() #setting the loss funciton 
    smallest_accuracy = 1.0
    smallest_index = 0
    largest_accuracy = 0.0
    largest_index = 0

    with torch.no_grad():
        index = 0 #starting from the first index and going through them all 
        #target_index = None ###??? don't need this probably can get rid of it 
        accuracy_list = [] # creating a list of the accurate ones so we can keep track of them

        for image, label in test_loader: #loading through images and labels in all the laoded ones
            output = model(image) #getting the output by running image through the model

            #target = label.float() #making it into float values for calculation
            #loss = criterion (output, target) #calculating the loss with the criterion function
            #print(f"loss = {loss}") ###??? for all of them??

            output_raw = output.squeeze(0) #getting rid of first dimension because unnecessary and not right format

            output_label = output_raw.argmax(dim=0) #each element of output_label is 0 or 1 based on whether defect or background is bigger for that pixel
            image_data = image.squeeze(0)[0] #squeeze first dimension 
            label_data = label.squeeze(0)[1] ###??? squeeze second dimension


            #the pixels with difference is the ones with different values so we can subtract it and get absolute value of each 
            #then divide this sum which will be the number of pixels different with the amoung of pixels to get amount wrong 
            #then do 1- that to find the percentage correct.

            accuracy = 1.0 - torch.abs(label_data-output_label).sum().item()/(image_height*image_width)
            accuracy_list.append(accuracy) #add the accuracy of that one image to the accuracy list
            if accuracy>largest_accuracy:
                largest_accuracy = accuracy
                largest_index=index
            if accuracy<smallest_accuracy:
                smallest_accuracy = accuracy
                smallest_index = index

            index = index+1

        accuracy_list_np = np.array(accuracy_list) #turning accuracy list into a numpy array to be able to use numpy features easily 
        print (f"Accuracy: min = {accuracy_list_np.min()} at index {smallest_index}, max = {accuracy_list_np.max()} at index {largest_index}")
        print (f"average = {np.average(accuracy_list_np)}")

if __name__=="__main__":
    #test_model_accuracy()
    #test_model()
    train_and_save_model()