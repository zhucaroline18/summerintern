######################################
# CSV CODE 
######################################

from csv import reader
from matplotlib import pyplot as plt
import numpy as np

def load_csv(filename, max_rows):

    ##creating array of the data as strings, each row is an element
    ##create a list with max_rows elements. 

    with open(filename) as file:
        csv_reader = reader(file)
        ret = list()
        count = 0
        for row in csv_reader:

            if row is None:
                continue
            ret.append(row)
            count+=1
            if max_rows>0 and count>=max_rows:
                break
        return ret

def load_dataset(filename, max_rows):
    # using the csv list, create a list of labels (first element)
    ## as well as the list of datasets

    #First load the csv
    if max_rows>0:
        max_rows +=1
    csv_data = load_csv(filename, max_rows)

    # get rid of the first row because it's headers 
    csv_data = csv_data[1:]

    dataset = list()
    labels = list()

    # the label is the first element. parse into an int value
    # add to labels list 
    for raw_row in csv_data:
        label = int(raw_row[0])
        labels.append(label)

        # for the rest, create a list where you 
        # divide each element in csv by 255 to turn into grayscale value
        # add onto the dataset 
        # ???? does this append as a list?
        row = [int(col)/255.0 for col in raw_row[1:]]
        dataset.append(row)

    return dataset, labels

def show_image(dataset, labels, index):
    label = labels[index]

    # each index of dataset is a list with 28x28 elements
    # reshaping that list into a 28x28 matrix
    image = np.array(dataset[index]).reshape((28,28))

    # multiply each value by 255 to get grayscale 
    image = image*255

    #print the label so you know what it is 
    print(f'label={label}')

    #plot it in grayscale in the python drawing thing
    #first plot it in memory then show it on the screen 
    plt.gray()
    plt.imshow(image, interpolation='nearest')
    plt.show()


if __name__=="__main__":

    #doing everything for specific index (row) of the file 
    dataset, labels = load_dataset('mnist_train.csv',5200)
    index = 5013
    print(labels[index])
    print(dataset[index])
    show_image(dataset, labels, index)

