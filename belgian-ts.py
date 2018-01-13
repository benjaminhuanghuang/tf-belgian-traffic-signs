import os
import tensorflow as tf
from skimage import transform
from skimage import data
import numpy as np
from skimage.color import rgb2gray



def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f)
                      for f in os.listdir(label_directory)
                      if f.endswith(".ppm")]
        for f in file_names:
            images.append(data.imread(f))
            labels.append(int(d))
    return images, labels


ROOT_PATH = os.getcwd()  # os.path.realpath(__file__)
train_data_directory = os.path.join(ROOT_PATH, 'data', "TrafficSigns/Training")
test_data_directory = os.path.join(ROOT_PATH, 'data', "TrafficSigns/Testing")

images, labels = load_data(train_data_directory)

images_array = np.array(images)
labels_array = np.array(labels)

# Print the `images` dimensions
print(images_array.ndim)

# Print the number of `images`'s elements
print(images_array.size)

# Print the first instance of `images`
images_array[0]

# Print the `labels` dimensions
print(labels_array.ndim)

# Print the number of `labels`'s elements
print(labels_array.size)

# Count the number of labels
print(len(set(labels_array)))