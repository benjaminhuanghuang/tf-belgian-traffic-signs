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
