##MNIST dataset
NIST (National Institute of Technology) collected hand written from a lot of people and put the image into a CSV (Comma Separated Value) file. Then someone modified the data and called it MNIST (Modified NIST) dataset. You could download it from https://www.kaggle.com/oddrationale/mnist-in-csv

The first row is the coloumn header, the rest rows has 28x28 + 1 = 784 + 1 = 785 columns. The first column is the integer lable for the digit. The other 784 columns are the gray scales from 0 to 255.

Please write code to read data from the file and get familar with the data. The following code requires numpy and matplotlib packages. It uses matplotlib to show visual image. If you get error when the package is not there, please
```python 
pip install numpy
pip install matplotlib
```
Sample code could be:
```python
from csv import reader
from matplotlib import pyplot as plt
import numpy as np

def load_csv(filename, max_rows):
    with open(filename) as file:
        csv_reader = reader(file)
        ret = list()
        count = 0
        for row in csv_reader:
            if row is None:
                continue
            ret.append(row)
            count += 1
            if max_rows > 0 and count >= max_rows:
                break

        return ret

def load_dataset(filename, max_rows):
    # as we will skip the first row, we need to load one more row from the CSV file
    if max_rows > 0:
        max_rows += 1
    csv_data = load_csv(filename, max_rows)

    # skip the first row as it just has column headers
    csv_data = csv_data[1:]

    # Create empty list
    dataset = list()
    labels = list()
    for raw_row in csv_data:
        # call int(s) to convert a string to int
        label = int(raw_row[0])
        labels.append(label)

        # for the column index from 1 to 784, convert the string to integer,
        # then divide it by 255 to get float number represent the scale of the gray in range 0 to 1
        row = [int(col) / 255.0 for col in raw_row[1:]]
        dataset.append(row)
    return dataset, labels

# show image for the data at index
def show_image(dataset, labels, index):
    label = labels[index]

    # reshape data to 28 x 28 matrix
    image = np.array(dataset[index]).reshape((28, 28))
    # convert 0 to 1 scale to 0 to 255
    image = image * 255
    print(f'label={label}')

    # now we want to plot a gray image in memory
    plt.gray()
    plt.imshow(image, interpolation='nearest')
    # finally show it on the screen
    plt.show()

if __name__ == "__main__":
    # As there are a lot of rows, we only want to try to read small number of rows, such as 10
    dataset, labels = load_dataset('mnist_train.csv', 10)
    # choose a index. You could choose another value
    index = 8
    print(labels[index])
    print(dataset[index])
    show_image(dataset, labels, index)
```