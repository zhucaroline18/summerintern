Week 6 is to work on the real project. As you need to learn new things and also work on the project, it's OK if you take more than 1 week to finish it.

# 6th Week
Tensor
Please note that tensors not only store values, but also keep track of the derivatives.
``` python
import torch

a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
# y = x*x + 3 x
# the derivative should be (2 x + 3)
b = a * a + 3 * a
loss = b.sum()
loss.backward()

# now could print the value
print(a)
# and the derivative
print(a.grad)
```

## Convolutional Neural Network
The real project requires understanding of Convolutional Neural Network.

1. Please watch the following video to understand what is "Filter", "Stride" and "Max Pooling". As the code in the following two vides are written in JavaScript, you do not need to write code, but try to understand the concept.
* https://www.youtube.com/watch?v=qPKsVAI_W6M&t=940s
* https://www.youtube.com/watch?v=pRWq_mtuppU
2. Then read the article that explain the concepts. https://towardsdatascience.com/pytorch-basics-how-to-train-your-neural-net-intro-to-cnn-26a14c2ea29 . As you may not have time, you do not need to try the code in this article, but please try to understand the concept of "channels", "kernel" or "filter", "stride" and "max pooling"

3. Watch video "PyTorch Tutorial 14 - Convolutional Neural Network (CNN)", https://www.youtube.com/watch?v=pDdP0TFzsoQ&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=14. As you may not have time, you do not need to try the code there.

4. We will also use torch.save(), torch.load(), model.dict_state() API to save the model. Those API are explained in the "PyTorch Tutorial 17 - Saving and Loading Models", https://www.youtube.com/watch?v=9L9jEOwRrCg&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=17 . Please watch the video to understand how the API will be used.

5. There is cs231n course from Stanford. The course is listed at YouTube. https://www.youtube.com/watch?v=vT1JzLTH4G4&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk. While it may be too much to watch, it's very useful and it talks about the details of convolutional neural networks.

### Semantic Segmentation
For this project, we will build a model based on the existing FCN ResNet or DeepLab model to detect the defect.

Please read the article https://towardsdatascience.com/semantic-hand-segmentation-using-pytorch-3e7a0a0386fa to understand how to build your own model based on the existing model built in PyTorch. When you read the article, you do not need to write the code, but try to understand the concept.

As we know, it will be difficult for everyone to rebuild some advanced model. That's why PyTorch provided some library for us to use them directly. We could build our own model based on the exsting model included int the PyTorch library. For example
``` python
class SegModel(nn.Module):
    def __init__(self):
        super(SegModel, self).__init__()
        # the model just use the deeplabv3
        self.dl = models.segmentation.deeplabv3_resnet50(pretrained=False, progress=False, num_classes=2)

    def forward(self, x):
        results = self.dl(x)
        # as the deeplabv3 returns a dict, we will get the "out" from the dictionary
        ret = results["out"]
        return ret
```
### Some basic PyTorch and Image APIs
We will use a few more PyTorch and Image APIs. Please get familiar with those APIs

squeeze/unsqueeze
``` python
import torch

# one dimensional array
a = torch.tensor([1,2,3])
print(a)
print(a.shape)

# convert to two dimensional array
b = torch.unsqueeze(a, dim=0)
print(b)
print(b.shape)

# convert to three dimensional array
c = torch.unsqueeze(b, dim=0)
print(c)
print(c.shape)

# convert back to two dimensional array
d = c.squeeze(dim = 0)
print(d)
print(d.shape)

# the squeeze could cause data loss
e = torch.tensor([[10,20,30],[40,50,60]])
f = c.squeeze(dim = 0)
print(f)
print(f.shape)

```

Image API
``` python
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms

# assume the following directory structure
# src/a.py
# dataset/train_val_image_label/train_image_label/srcImg/*.bmp
# dataset/train_val_image_label/train_image_label/label/*.bmp

# please change path if the directory is not same as the above.


current_location = os.path.dirname(__file__)
train_src_image_folder = os.path.join(current_location, "../dataset/train_val_image_label/train_image_label/srcImg")
train_label_image_folder = os.path.join(current_location, "../dataset/train_val_image_label/train_image_label/label")

src_image_name = "54b44ab9-17b0-4807-b1ce-b54830dc901e.bmp"

src_image_full_path = os.path.join(train_src_image_folder, src_image_name)
print(f"src={src_image_full_path}")
label_image_full_path = os.path.join(train_label_image_folder, src_image_name)
print(f"label={label_image_full_path}")

# open image
src_image = Image.open(src_image_full_path)
print(src_image.size)
src_image_rgb = src_image.convert("RGB")
src_image_rgb.show()

label_image = Image.open(label_image_full_path)
print(label_image.size)

# get image data into np.array
src_image_data = np.array(src_image, dtype=np.float32)
print(f"src_image_data.shape={src_image_data.shape}")
print(src_image_data.max())
# convert to 0 to 1
src_image_data = src_image_data / 255.0

label_image_data = np.array(label_image, dtype=np.float32)
print(label_image_data.max())
# replace all element greater than 0 with value 1.0
# we only want to use 0 or 1
label_image_data[label_image_data > 0] = 1.0
print(label_image_data.max())

# show two images
fig, axarr = plt.subplots(2)
# show the first image with src_image_data
axarr[0].imshow(src_image_data, cmap="gray")
# show the second image with label_image_data
axarr[1].imshow(label_image_data, cmap="gray")
plt.show()


# When we pass the value to Full Convolutional Network, the input should have three channel, we will also resize it.
# please note that we use // operator so that we will get integer value
image_width = 716 // 4
image_height = 920 // 4
src_image = Image.open(src_image_full_path).convert("RGB")
preprocess = transforms.Compose([
                            transforms.Resize((image_height, image_width), transforms.InterpolationMode.BICUBIC),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
src_image_data = preprocess(src_image)
print(src_image_data.shape)

# the above code using transforms.Compose() has the same result as
y = transforms.Resize((image_height, image_width), transforms.InterpolationMode.BICUBIC)(src_image)
y = transforms.ToTensor()(y)
y = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(y)

# now, process the label image
label_image = Image.open(label_image_full_path).convert("L")
# create a function that will be used to resize the image
trf_resize = transforms.Resize((image_height, image_width), transforms.InterpolationMode.BICUBIC)
label_image_resized = trf_resize(label_image)
# create numpy data from the image
label_image_data = np.array(label_image_resized)
print(label_image_data)
# we will then create two classes, the first is the background and the second is the label.
defect_class = np.zeros_like(label_image_data)
defect_class[label_image_data > 0] = 1.0
# background class
background_class = np.zeros_like(label_image_data)
background_class[label_image_data == 0] = 1.0
label_image_classes = torch.tensor([background_class, defect_class])
print(label_image_classes.shape)
```

Conv2d
This is to show the effects of Conv2d
``` python
import os
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

current_location = os.path.dirname(__file__)
train_src_image_folder = os.path.join(current_location, "../dataset/train_val_image_label/train_image_label/srcImg")
train_label_image_folder = os.path.join(current_location, "../dataset/train_val_image_label/train_image_label/label")

src_image_name = "54b44ab9-17b0-4807-b1ce-b54830dc901e.bmp"
src_image = Image.open(os.path.join(train_src_image_folder, src_image_name))

src_image_data = np.array(src_image, dtype=np.float32)

src_image_tensor = torch.tensor(src_image_data)
print(src_image_tensor.shape)

# The data is 2 dimensional, convert it to 4 dimensional. It's because nn.Conv2d requires 4 dimensional data. 
# the first dimensional is the the index in the batch
# the second dimensional is channel
# the third and fourth is the 2D data
src_image_tensor = torch.unsqueeze(src_image_tensor, dim=0)
src_image_tensor = torch.unsqueeze(src_image_tensor, dim=0)
print(src_image_tensor.shape)


conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
conv_result = conv.forward(src_image_tensor)
print(conv_result.shape)

conv_result_data = conv_result.squeeze().detach().numpy()

fig, axarr = plt.subplots(2)
# the first is source image
axarr[0].imshow(src_image_data, cmap="gray")
# the second is the image after convolutional process
axarr[1].imshow(conv_result_data, cmap="gray")
plt.show()
```
### Segmentation
We will build your own model based on the existing model included in the PyTorch library, then train it with our own data and test it. The following is the sample code.

Please note that it may take very long to train the model. You could update the num_epochs to 1, or even reduce the image_width and image_height further.

You also could try to replace the deeplabv3_resnet50 with fcn_resnet50 to show the impact.

You also could try to replace the loss function with MSELoss to show the impact.
``` python
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from PIL import Image

import numpy as np
import os
import matplotlib.pyplot as plt


# assume the following directory structure
# src/a.py
# dataset/train_val_image_label/train_image_label/srcImg/*.bmp
# dataset/train_val_image_label/train_image_label/label/*.bmp

# please change path if the directory is not same as the above.

current_location = os.path.dirname(__file__)
train_src_image_folder = os.path.join(current_location, "../dataset/train_val_image_label/train_image_label/srcImg")
train_label_image_folder = os.path.join(current_location, "../dataset/train_val_image_label/train_image_label/label")
test_src_image_folder = os.path.join(current_location, "../dataset/train_val_image_label/val_image_label/srcImg")
test_label_image_folder = os.path.join(current_location, "../dataset/train_val_image_label/val_image_label/val_label")

batch_size = 2
# make the size smaller so that we could run it quickly
image_width = 716 // 4
image_height = 920 // 4
learning_rate = 0.0001
num_epochs = 4

class ImageDataset(Dataset):
    def __init__(self, src_image_folder, label_image_folder):
        super(ImageDataset, self).__init__()
        self.src_image_folder = src_image_folder
        self.label_image_folder = label_image_folder
        self.src_img_names = os.listdir(src_image_folder)
        mod = len(self.src_img_names) % batch_size
        # remove the entries that not fall into the batch
        self.src_img_names = self.src_img_names[0: len(self.src_img_names) - mod]
        print(len(self.src_img_names))


    def __len__(self):
        return len(self.src_img_names)

    def __getitem__(self, index):
        src_img_loc = os.path.join(self.src_image_folder, self.src_img_names[index])
        src_image = Image.open(src_img_loc).convert("RGB")

        preprocess = transforms.Compose([
                                    transforms.Resize((image_height, image_width), transforms.InterpolationMode.BICUBIC),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        

        src_image_data = preprocess(src_image)


        trf_resize = transforms.Resize((image_height, image_width), transforms.InterpolationMode.BICUBIC)
        label_img_loc = os.path.join(self.label_image_folder, self.src_img_names[index])
        label_image = Image.open(label_img_loc).convert("L")
        label_image_resized = trf_resize(label_image)
        # create numpy data from the image
        label_image_data = np.array(label_image_resized)
        # we will then create two classes, the first is the background and the second is the label.
        defect_class = np.zeros_like(label_image_data)
        defect_class[label_image_data > 0] = 1.0
        # background class
        background_class = np.zeros_like(label_image_data)
        background_class[label_image_data == 0] = 1.0
        label_image_classes = torch.tensor([background_class, defect_class])

        return src_image_data, label_image_classes


class SegmentationModel(nn.Module):
    def __init__(self):
        super(SegmentationModel, self).__init__()
        # the model just use the deeplabv3
        self.dl = models.segmentation.deeplabv3_resnet50(pretrained=False, progress=False, num_classes=2)

    def forward(self, x):
        results = self.dl(x)
        # as the deeplabv3 returns a dict, we will get the "out" from the dictionary
        ret = results["out"]
        return ret



def test_imagedataset():
    dataset = ImageDataset(train_src_image_folder, train_label_image_folder)
    src_image, label_image = dataset[10]
    print(src_image)
    print(src_image.shape)
    print(label_image)
    print(label_image.shape)

    fig, axarr= plt.subplots(2,3)

    # the image has three channels
    axarr[0,0].imshow(src_image[0], cmap = "gray")
    axarr[0,1].imshow(src_image[1], cmap = "gray")
    axarr[0,2].imshow(src_image[2], cmap = 'gray')

    # the labels has two, the first element is background, the second element is the label
    axarr[1,0].imshow(label_image[0], cmap="gray")
    axarr[1,1].imshow(label_image[1], cmap="gray")
    plt.show()

def test_train_one():
    train_dataset = ImageDataset(train_src_image_folder, train_label_image_folder)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    model = SegmentationModel()

    train_iter = iter(train_loader)
    inputs, labels = train_iter.next()
    outputs = model.forward(inputs)
    print(outputs.shape)
    print(labels.shape)

    targets = labels.float()
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(outputs, targets)
    print(loss)


def train_and_save_model():
    train_dataset = ImageDataset(train_src_image_folder, train_label_image_folder)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # model
    model = SegmentationModel()

    # loss
    # criterion = nn.MSELoss(reduction='mean')
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    train_loop(train_loader, model, criterion, optimizer)

def load_checkpoint_and_train_and_save_model():
    train_dataset = ImageDataset(train_src_image_folder, train_label_image_folder)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # model
    model = SegmentationModel()

    # loss
    # criterion = nn.MSELoss(reduction='mean')
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    checkpoint = torch.load("saved-deeplab-checkpoint.pth")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    train_loop(train_loader, model, criterion, optimizer)


def train_loop(train_loader, model, criterion, optimizer):
    # training loop
    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        for batch_index, (images, labels) in enumerate(train_loader):        
            # forward
            predicts = model(images)

            # loss
            targets = labels.float()
            loss = criterion(predicts, targets)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_index % 20 == 0:
                print(f"epoch {epoch}, step {batch_index} / {n_total_steps}, loss={loss}")

            # During test, we could break earlier
            # if batch_index > 40:
            #   break

    torch.save(model, os.path.join(current_location, "saved-deeplab.pth"))
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    torch.save(state, os.path.join(current_location, "saved-deeplab-checkpoint.pth"))


def test_model():
    test_dataset = ImageDataset(test_src_image_folder, test_label_image_folder)
    # test_dataset = ImageDataset(train_src_image_folder, train_label_image_folder)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    model = torch.load(os.path.join(current_location, "saved-deeplab.pth"))
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        # only show the one that we are interesting
        # 20, 30 are OK, but
        # 40, 42 are not OK.
        target_index = 45
        index = 0
        for image, label in test_loader:
            if index == target_index:
                output = model(image)
                target = label.float()
                loss = criterion(output, target)
                print(f"loss={loss}")

                output_raw = output.squeeze(0)

                output_background = output_raw[0]
                output_defect = output_raw[1]

                background_img = transforms.ToPILImage()(output_background)
                defect_image = transforms.ToPILImage()(output_defect)
                # we could show the image
                # background_img.show()
                # defect_image.show()

                # the index who has the max value. As we only have two classes, it's either 0 or 1
                output_label = output_raw.argmax(dim=0)
                print(output_label.shape)
                print(output_label)


                # if the defect class is larger than background, it's defect.
                # ytest should be same as output_label
                ytest = output_raw[1] > output_raw[0]

                # draw the images
                f, axarr = plt.subplots(2,3)
                image_data = image.squeeze(0)[0]
                label_data = label.squeeze(0)[1]

                # show the image
                axarr[0,0].imshow(image_data, cmap = "gray")

                # background
                axarr[0,1].imshow(torch.sigmoid(output_raw[0]), cmap="gray")

                # defect
                axarr[0,2].imshow(torch.sigmoid(output_raw[1]), cmap="gray")

                # expected label
                axarr[1,0].imshow(label_data, cmap = "gray")

                # defect label
                axarr[1,1].imshow(output_label, cmap = "gray")
                # show ytest, it should be same as output_label
                # axarr[1,2].imshow(ytest, cmap="gray")
                plt.show()


            index += 1

def test_predict_using_pretrained_standard_model():
    """
    Predict the object using pretrained standard model
    We will see that the standard model will not work for us
    """
    test_dataset = ImageDataset(test_src_image_folder, test_label_image_folder)
    # test_dataset = ImageDataset(train_src_image_folder, train_label_image_folder)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    model = models.segmentation.deeplabv3_resnet50(pretrained=True, progress=False)
    model.eval()
    with torch.no_grad():
        target_index = 20
        index = 0
        for image, label in test_loader:
            if index == target_index:
                output = model(image)["out"]
                target = label.float()

                output_classes = output.squeeze(0)

                # the index who has the max value
                output_label = output_classes.argmax(dim=0)
                print(output_label.shape)
                print(output_label)


                # draw the images
                f, axarr = plt.subplots(2,2)
                image_data = image.squeeze(0)[0]
                label_data = label.squeeze(0)[1]

                # show the image
                axarr[0,0].imshow(image_data, cmap = "gray")

                # background
                # axarr[0,1].imshow(torch.sigmoid(output_classes[0]), cmap="gray")

                # expected label
                axarr[1,0].imshow(label_data, cmap = "gray")

                # defect label
                axarr[1,1].imshow(output_label, cmap = "gray")
                plt.show()


            index += 1


if __name__ == "__main__":
    # test_imagedataset()
    # test_train_one()
    # train_and_save_model()
    # load_checkpoint_and_train_and_save_model()
    test_model()
    # test_predict_using_pretrained_standard_model()
```