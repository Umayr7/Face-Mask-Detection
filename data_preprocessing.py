#!/usr/bin/env python

from sklearn.model_selection import train_test_split
import numpy as np
import os
import PIL
import cv2
import pickle
from PIL import Image
from autocrop import Cropper
import matplotlib.pyplot as plt


# for my personal data
def create_dataset(CATEGORIES, MY_DIRECTORY):
    IMG_SIZE = 64
    X = []
    y = []
    for category in CATEGORIES:
        path = os.path.join(MY_DIRECTORY, category)
        class_num_label = CATEGORIES.index(category)
        for img in os.listdir(path):
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                
                X.append(img_array)
                y.append(class_num_label)
    return [X, y]

# for my data
def reshape_to_np_array(X, y):
    IMG_SIZE = 64
    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE)
    X = X.reshape(-1, IMG_SIZE*IMG_SIZE)
    y = np.array(y)
    
    return [X, y]

# Normalize the data
def normalize_data(X):
    X = X / X.max()
    return X

# def saving_data(X, y, x_path, y_path):
#     pickle_out = open(x_path, "wb")
#     pickle.dump(X, pickle_out)
#     pickle_out.close()

#     pickle_out = open(y_path, "wb")
#     pickle.dump(y, pickle_out)
#     pickle_out.close()