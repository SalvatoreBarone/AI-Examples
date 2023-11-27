#!python
import gc, sys, os, imp, itertools, pickle, click, contextlib, json5, tflite, numpy as np, pandas as pd, tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from distutils.dir_util import mkpath
from scipy.stats import wasserstein_distance
from sklearn.cluster import KMeans
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.exceptions import YellowbrickValueError, YellowbrickWarning
from numba import cuda
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def getCIFAR10(seed = None):
    
    """
    Load and preprocess the CIFAR-10 dataset.

    Returns:
        tuple: A tuple containing lists of training images, training labels, validation images,
        validation labels, test images, test labels, and a dictionary mapping class indices to names.
    """
    
    # Define class names for each category in CIFAR-10
    class_names = {0: 'airplane',1: 'automobile',2: 'bird',3: 'cat',4: 'deer',5: 'dog',6: 'frog',7: 'horse',8: 'ship',9: 'truck'}
    
    # Load CIFAR-10 dataset
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    
    # Split the training data into training and validation sets
    train_images, x_val, train_labels, y_val = train_test_split(train_images, train_labels, test_size=0.1, random_state = seed)
    
    # Convert pixel values to float32 and normalize them to range [0, 1]
    train_images = train_images.astype('float32')
    test_images = test_images.astype('float32')
    x_val = x_val.astype('float32')
    train_images /= 255
    test_images /= 255
    x_val /= 255
    
    # Return the processed data and class names
    return list(train_images), list(train_labels), list(x_val), list(y_val), list(test_images), list(test_labels), class_names

def compute_baseline_accuracy(x_test, y_test, qmodel, output_index):
    Cset = np.zeros((len(y_test),))
    for index, (x, y) in tqdm(enumerate(zip(x_test, y_test)), total=len(y_test), desc=f"Computing the baseline accuracy...", bar_format="{desc:30} {percentage:3.0f}% | {bar:40}{r_bar}{bar:-10b}"):
        if isinstance(y, (list, tuple, np.ndarray)):
            y = y[0] 
        image = np.expand_dims(x, axis = 0).astype(np.float32)
        qmodel.set_tensor(qmodel.get_input_details()[0]["index"], image)
        qmodel.invoke()
        p = qmodel.get_tensor(output_index)
        if  np.argmax(p) == y:
            Cset[index] = 1
    return Cset




if __name__ == "__main__":
    _, _, _, _, x_test, y_test, _ = getCIFAR10(0) #! 0 is for reproducible output across multiple function calls
    qmodel = tf.lite.Interpreter(model_path="ResNet8.tflite", experimental_preserve_all_tensors= True)
    qmodel.allocate_tensors()
    output_index = qmodel.get_output_details()[0]["index"]

    Cset = np.zeros((len(y_test),))
    for index, (x, y) in tqdm(enumerate(zip(x_test, y_test)), total=len(y_test), desc=f"Computing the baseline accuracy...", bar_format="{desc:30} {percentage:3.0f}% | {bar:40}{r_bar}{bar:-10b}"):
        if isinstance(y, (list, tuple, np.ndarray)):
            y = y[0] 
        image = np.expand_dims(x, axis = 0).astype(np.float32)
        qmodel.set_tensor(qmodel.get_input_details()[0]["index"], image)
        qmodel.invoke()
        p = qmodel.get_tensor(output_index)
        if  np.argmax(p) == y:
            Cset[index] = 1
    baseline_accuracy = np.sum(Cset) / len(x_test) * 100
    print(f"Baseline accuracy: {baseline_accuracy}")