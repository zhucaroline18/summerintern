import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.L1 = nn.Linear(784, 32)
        self.relu = nn.ReLU()
        self.L2 = nn.Linear(32, 20)
        self.L3 = nn.Linear(20, 10)

    def forward(self, x):
        z1 = self.L1(x)
        a1 = self.relu(z1)
        z2 = self.L2(a1)
        a2 = self.relu(z2)
        z3 = self.L3(a2)
        a3 = self.relu(z3)

        return a3

train_dataset = torchvision.datasets.MNIST(
    root = 'data',
    train = True, 
    download = True,
    transform = torchvision.transforms.ToTensor()
)


test_dataset = torchvision.datasets.MNIST(
    root = 'data',
    train = False, 
    download = True,
    transform = torchvision.transforms.ToTensor()
)



def train_model():
    model = MyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
    loss_fn = nn.CrossEntropyLoss()

    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = 100, shuffle = True)

    for epoch in range (2):
        for i, (images, labels) in enumerate(train_loader):        
            # foward
            images = images.view(images.shape[0], 28*28)
            y = model(images)
            loss = loss_fn(y, labels)

            #backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i%100 == 0:
                print(f"epoch {epoch}: loss = {loss}")

    torch.save(model, "my_mnist.pth")

def test_model():
    model = torch.load("my_mnist.pth")

    index = 25
    image, label = test_dataset[index]
    image = image.view(image.shape[0],28*28)
    print (label)
    output = model(image)
    print(output)
    print(output.shape)
    predict = output.argmax(dim = 1)
    print(predict)

def test_model_all():
    model = torch.load("my_mnist.pth")

    correct = 0
    for i in range(len(test_dataset)):
        image, label = test_dataset[i]
        image = image.view(image.shape[0],28*28)
        output = model(image)
        predict = output.argmax(dim=1)
        if label == predict:
            correct+=1
        else:
            print(f"instance {i}")
            print(f"label was: {label}, predicted was: {predict}")

    print(f"{correct} correct out of {len(test_dataset)}")
      

def check_dataset():
    index = 6244
    data, label = test_dataset[index]
    print(label)
    print(data)
    print(data.shape)
    image_data = data.squeeze(dim = 0)
    print (image_data.shape)

    plt.imshow(image_data, cmap = "gray")
    plt.show()

if __name__ == "__main__":
    #train_model()
    #test_model_all()
    check_dataset()
    
