## 5th Week
### PyTorch
Right now, we understand the principle of Neural Network, which is to use gradient descent to find the minimal cost/loss function. We use feed_forward to get the cost/loss value, then apply the derivative formula to update the weights and bias.

For a three-layer Neural Network, it's easy to do it from scratch. We also see that the most difficult part in the three-layer Neural Network is the backward_propagation part. When we need to build more than three layers, the backward_propagation part is more difficult. The PyTorch library will help us to calculate the backward_propagation.

We will follow the tutorial published by "Python Engineer" to learn the basics in PyTorch library. Once we learn the basics, it will be easy to apply the knowledge on the MNIST dataset. The goal of this week is to be able to use PyTorch library to redo the MNIST dataset problem, but use multiple layers instead of just three layers.

There are 16 videos in the PyTorch Turotail. https://www.youtube.com/watch?v=EMXfZB8FVUA&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4 . You could skip the first one as the first one just talks about the installation. You could start with the second one, which talks about Tensor.

Notes:

* You could speed up using 1.5 as Playback speed while you watch the Youtube video.
* You could just watch the 2 to 13 and skip the 14,15,16 for now. You could watch No.14, 15, 16 later.
* In the tutorial 05, you may not understand the sample code using the pure NumPy library. That's OK. But you should try to run the code using Torch, as that's the fundamental of the PyTorch library. The following is the code for tutorial 05.
```python 
import torch
import numpy as np

X = torch.tensor([1,2,3,4], dtype=torch.float32)
Y = torch.tensor([2,4,6,8], dtype=torch.float32)
W = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

def forward(x):
    return W * x

# loss = MSE (mean of error square)
def loss(y, y_predicted):
    return ((y_predicted-y)**2).mean()

# Training
learning_rate = 0.01
n_iters = 100

for epoch in range(n_iters):
    y_pred = forward(X)

    l = loss(Y, y_pred)

    # let the PyTorch library to calculate the derivative_of_loss_over_weight
    l.backward()
    
    # update weights. As the PyTorch should not keep track of the gradient
    # on the weight update operation, we will use torch.no_grade()
    with torch.no_grad():
        W -= learning_rate * W.grad

    # After each training, the grad on the weight should be reset to zero
    W.grad.zero_()

    if epoch % 10 == 0:
        print(f"epoch {epoch}: W={W}, loss={l}")
```

You could skip the PyTorch Tutorial 08 - Logistic Regression
The PyTorch Tutorial 13 - Feed-Forward Neural Network is very important. The following is the code that uses MNIST dataset
``` python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

input_size = 784
hidden_size = 100
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001

# MNIST
train_dataset = torchvision.datasets.MNIST(
    root='data',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

test_dataset = torchvision.datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

# Inspect the data
print(len(train_dataset))
index = 4
img, label = train_dataset[index]
print(label)
print(img)
print(img.shape)
print(img.size())

# show images
# as the img shape is [1,28,28], we will get img[0], which is the data
# plt.imshow(img[0], cmap="grey")
# plt.show()


train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Inspect loader
examples = iter(train_loader)
samples, labels = examples.next()
print(samples.shape, labels.shape)

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

model = NeuralNetwork(input_size, hidden_size, num_classes)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
# use Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

# training loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28 * 28)
        
        # forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        #backwards
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"epoch {epoch}, step {i} / {n_total_steps}, loss={loss}")


# test
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28)
        outputs = model(images)

        # value, index
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'accuancy={acc}')
```

### Real DataSet
Now, we will use the dataset used in production. Please download the dataset from https://drive.google.com/file/d/1KVBfMk5IBij2u2JGyE-WKG6WkcX_oqOA/view?usp=sharing

It's a ZIP file and you could unzip it to your own folder.

Let's explore the dataset a little bit to understand what they are. For example, there is train_image_label folder, which contains subfolders label and srcImg. Under those folders, are the images in bmp format.

Python has a lot of libraries that will help us to read the images. For example, we could use os.listdir() API to get the file names. Once we have a particular file name, we could use os.path.join() API to get the full path of the file. Then we could use PIL.Image.open() to open the image in memory. Once it's loaded in memory, we could use image.show() to display it to make sure we got the correct code. Once we have the image, we will convert it to matrix data. PyTorch has torchvision.transforms.functional.to_tensor() API that will transform images to matrix data. Please note that the tensor data is a three dimensional array instead of a two dimensional array, which is similar to the MNIST dataset.

Please get familiar with those Python APIs. The following are some sample code.
``` python
import os
from PIL import Image
import torchvision.transforms.functional as TF

train_src_image_folder = "c:/git/defect-detection/dataset/train_val_image_label/train_image_label/srcImg"
train_label_image_folder = "c:/git/defect-detection/dataset/train_val_image_label/train_image_label/label"

# List all of the files under the folder
src_image_names = os.listdir(train_src_image_folder)
print(src_image_names)
print(len(src_image_names))

# let's check one of the image
src_image_name = "54b44ab9-17b0-4807-b1ce-b54830dc901e.bmp"

src_image_full_path = os.path.join(train_src_image_folder, src_image_name)
print(f"src={src_image_full_path}")
label_image_full_path = os.path.join(train_label_image_folder, src_image_name)
print(f"label={label_image_full_path}")


# open image
src_image = Image.open(src_image_full_path)
label_image = Image.open(label_image_full_path)

# show the image
print("src_image")
src_image.show()
print("label_image")
label_image.show()

# now convert it to Tensor
src_image_tensor = TF.to_tensor(src_image)
label_image_tensor = TF.to_tensor(label_image)

print(src_image_tensor)
# We will see that it's three dimension data
print(src_image_tensor.size())

# we only need to get the first element, which is two-dimensional array.
print(src_image_tensor[0])
```