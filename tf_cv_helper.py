""" This is a helper file for tensorflow computer vision projects.
"""

__version__ = "0.0.1"
__author__ = "Ngo Hong Thai"

############# Setup environment #############
import os
import glob
import sys
import platform
import itertools
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow.keras
from pathlib import Path
from sklearn.metrics import confusion_matrix

def print_env_info(): 
    """ Print environment information
    
    """
    
    print(f"Python Platform: {platform.platform()}")
    print(f"Python {sys.version}")
    print(f"Tensor Flow Version: {tf.__version__}")
    print(f"Tensor Flow Hub Version: {hub.__version__}")
    print(f"Keras Version: {tensorflow.keras.__version__}")

    gpu = len(tf.config.list_physical_devices('GPU'))>0
    print("GPU is", "available" if gpu else "NOT AVAILABLE")

##### Các hàm hỗ trợ download data, visualize data ... #####

def download_data(url, filename):
    """ Download data from url and extract it to filename

    Args:
        url (str): Link to download data
        filename (str): Name of file to extract data to

    Returns:
        str: Path to extracted data directory
    """
    
    # Download data
    data_dir = tf.keras.utils.get_file(filename, url, extract=True, cache_dir=".")
    # Remove .zip extension
    data_dir, _ = os.path.splitext(data_dir)
    return data_dir

def print_tree(data_dir, level = 0):
    """ Print out a tree structure of data_dir

    Args:
        data_dir (str): Path to data directory
        level (int, optional): Level of directory. Defaults to 0.
    """    
    
    indent = ' ' * 6 * level
    num_files = len(glob.glob(f'{data_dir}/*'))
    print(f'{indent}{os.path.basename(data_dir)}/ ({num_files} files)')
    subindent = ' ' * 6 * (level + 1)
    for f in glob.glob(f'{data_dir}/*'):
        if os.path.isdir(f):
            print_tree(f, level + 1)

def view_random_images(target_dir, num_images = 3, seed = 42):
    """ Show num_images random images from target_dir

    Parameters:
        target_dir (str): Path to target directory
        num_images (int, optional): Number of images to show. Defaults to 3.
        seed (int, optional): Seed for random number generator. Defaults to 42.
    """
    
    np.random.seed(seed)
    # Get all the images in data_path
    all_images = [image_path for image_path in Path(target_dir).rglob("*.*")]
    # Get random images
    random_images = np.random.choice(all_images, size=num_images, replace=False)
    # Plot random images
    plt.figure(figsize=(15, 15))
    for i, image_path in enumerate(random_images):
        ax = plt.subplot(1, num_images, i+1)
        image = plt.imread(image_path)
        plt.imshow(image)
        plt.title(image_path.parent.name)
        plt.axis("off")
        

############# Các hàm hỗ trợ build models ############

# model_name = "efficientnetv2-b0" # @param ['efficientnetv2-s', 'efficientnetv2-m', 'efficientnetv2-l', 'efficientnetv2-s-21k', 'efficientnetv2-m-21k', 'efficientnetv2-l-21k', 'efficientnetv2-xl-21k', 'efficientnetv2-b0-21k', 'efficientnetv2-b1-21k', 'efficientnetv2-b2-21k', 'efficientnetv2-b3-21k', 'efficientnetv2-s-21k-ft1k', 'efficientnetv2-m-21k-ft1k', 'efficientnetv2-l-21k-ft1k', 'efficientnetv2-xl-21k-ft1k', 'efficientnetv2-b0-21k-ft1k', 'efficientnetv2-b1-21k-ft1k', 'efficientnetv2-b2-21k-ft1k', 'efficientnetv2-b3-21k-ft1k', 'efficientnetv2-b0', 'efficientnetv2-b1', 'efficientnetv2-b2', 'efficientnetv2-b3', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7', 'bit_s-r50x1', 'inception_v3', 'inception_resnet_v2', 'resnet_v1_50', 'resnet_v1_101', 'resnet_v1_152', 'resnet_v2_50', 'resnet_v2_101', 'resnet_v2_152', 'nasnet_large', 'nasnet_mobile', 'pnasnet_large', 'mobilenet_v2_100_224', 'mobilenet_v2_130_224', 'mobilenet_v2_140_224', 'mobilenet_v3_small_100_224', 'mobilenet_v3_small_075_224', 'mobilenet_v3_large_100_224', 'mobilenet_v3_large_075_224']

# Dictionary of model names to their corresponding TF Hub handles
model_handle_map = {
    "efficientnetv2-s": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_s/feature_vector/2",
    "efficientnetv2-m": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_m/feature_vector/2",
    "efficientnetv2-l": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_l/feature_vector/2",
    "efficientnetv2-s-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_s/feature_vector/2",
    "efficientnetv2-m-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_m/feature_vector/2",
    "efficientnetv2-l-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_l/feature_vector/2",
    "efficientnetv2-xl-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_xl/feature_vector/2",
    "efficientnetv2-b0-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b0/feature_vector/2",
    "efficientnetv2-b1-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b1/feature_vector/2",
    "efficientnetv2-b2-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b2/feature_vector/2",
    "efficientnetv2-b3-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b3/feature_vector/2",
    "efficientnetv2-s-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_s/feature_vector/2",
    "efficientnetv2-m-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_m/feature_vector/2",
    "efficientnetv2-l-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_l/feature_vector/2",
    "efficientnetv2-xl-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_xl/feature_vector/2",
    "efficientnetv2-b0-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b0/feature_vector/2",
    "efficientnetv2-b1-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b1/feature_vector/2",
    "efficientnetv2-b2-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b2/feature_vector/2",
    "efficientnetv2-b3-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b3/feature_vector/2",
    "efficientnetv2-b0": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b0/feature_vector/2",
    "efficientnetv2-b1": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b1/feature_vector/2",
    "efficientnetv2-b2": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b2/feature_vector/2",
    "efficientnetv2-b3": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b3/feature_vector/2",
    "efficientnet_b0": "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1",
    "efficientnet_b1": "https://tfhub.dev/tensorflow/efficientnet/b1/feature-vector/1",
    "efficientnet_b2": "https://tfhub.dev/tensorflow/efficientnet/b2/feature-vector/1",
    "efficientnet_b3": "https://tfhub.dev/tensorflow/efficientnet/b3/feature-vector/1",
    "efficientnet_b4": "https://tfhub.dev/tensorflow/efficientnet/b4/feature-vector/1",
    "efficientnet_b5": "https://tfhub.dev/tensorflow/efficientnet/b5/feature-vector/1",
    "efficientnet_b6": "https://tfhub.dev/tensorflow/efficientnet/b6/feature-vector/1",
    "efficientnet_b7": "https://tfhub.dev/tensorflow/efficientnet/b7/feature-vector/1",
    "bit_s-r50x1": "https://tfhub.dev/google/bit/s-r50x1/1",
    "inception_v3": "https://tfhub.dev/google/imagenet/inception_v3/feature-vector/4",
    "inception_resnet_v2": "https://tfhub.dev/google/imagenet/inception_resnet_v2/feature-vector/4",
    "resnet_v1_50": "https://tfhub.dev/google/imagenet/resnet_v1_50/feature-vector/4",
    "resnet_v1_101": "https://tfhub.dev/google/imagenet/resnet_v1_101/feature-vector/4",
    "resnet_v1_152": "https://tfhub.dev/google/imagenet/resnet_v1_152/feature-vector/4",
    "resnet_v2_50": "https://tfhub.dev/google/imagenet/resnet_v2_50/feature-vector/4",
    "resnet_v2_101": "https://tfhub.dev/google/imagenet/resnet_v2_101/feature-vector/4",
    "resnet_v2_152": "https://tfhub.dev/google/imagenet/resnet_v2_152/feature-vector/4",
    "nasnet_large": "https://tfhub.dev/google/imagenet/nasnet_large/feature_vector/4",
    "nasnet_mobile": "https://tfhub.dev/google/imagenet/nasnet_mobile/feature_vector/4",
    "pnasnet_large": "https://tfhub.dev/google/imagenet/pnasnet_large/feature_vector/4",
    "mobilenet_v2_100_224": "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4",
    "mobilenet_v2_130_224": "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/feature_vector/4",
    "mobilenet_v2_140_224": "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4",
    "mobilenet_v3_small_100_224": "https://tfhub.dev/google/imagenet/mobilenet_v3_small_100_224/feature_vector/5",
    "mobilenet_v3_small_075_224": "https://tfhub.dev/google/imagenet/mobilenet_v3_small_075_224/feature_vector/5",
    "mobilenet_v3_large_100_224": "https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/feature_vector/5",
    "mobilenet_v3_large_075_224": "https://tfhub.dev/google/imagenet/mobilenet_v3_large_075_224/feature_vector/5",
}

# Dictionary of image sizes for each model.
model_image_size_map = {
    "efficientnetv2-s": 384,
    "efficientnetv2-m": 480,
    "efficientnetv2-l": 480,
    "efficientnetv2-b0": 224,
    "efficientnetv2-b1": 240,
    "efficientnetv2-b2": 260,
    "efficientnetv2-b3": 300,
    "efficientnetv2-s-21k": 384,
    "efficientnetv2-m-21k": 480,
    "efficientnetv2-l-21k": 480,
    "efficientnetv2-xl-21k": 512,
    "efficientnetv2-b0-21k": 224,
    "efficientnetv2-b1-21k": 240,
    "efficientnetv2-b2-21k": 260,
    "efficientnetv2-b3-21k": 300,
    "efficientnetv2-s-21k-ft1k": 384,
    "efficientnetv2-m-21k-ft1k": 480,
    "efficientnetv2-l-21k-ft1k": 480,
    "efficientnetv2-xl-21k-ft1k": 512,
    "efficientnetv2-b0-21k-ft1k": 224,
    "efficientnetv2-b1-21k-ft1k": 240,
    "efficientnetv2-b2-21k-ft1k": 260,
    "efficientnetv2-b3-21k-ft1k": 300, 
    "efficientnet_b0": 224,
    "efficientnet_b1": 240,
    "efficientnet_b2": 260,
    "efficientnet_b3": 300,
    "efficientnet_b4": 380,
    "efficientnet_b5": 456,
    "efficientnet_b6": 528,
    "efficientnet_b7": 600,
    "inception_v3": 299,
    "inception_resnet_v2": 299,
    "nasnet_large": 331,
    "pnasnet_large": 331,
}

def get_model_handle(model_name):
    """ Lấy url của model

    Args:
        model_name (str): Tên của model

    Raises:
        ValueError: Nếu model_name không tồn tại

    Returns:
        str: url của model
    """
    
    if model_name not in model_handle_map:
        raise ValueError(
            "Model name '{}' is undefined. Available models: {}".format(
                model_name, list(model_handle_map.keys())))
    return model_handle_map[model_name]

def get_model_input_size(model_name):
    """ Lấy kích thước của ảnh đầu vào của model

    Args:
        model_name (str): Tên của model

    Raises:
        ValueError: Nếu model_name không tồn tại

    Returns:
        (int, int): Kích thước của ảnh đầu vào của model (image size)
    """
    
    if model_name not in model_image_size_map:
        raise ValueError(
            "Model name '{}' is undefined. Available models: {}".format(
                model_name, list(model_image_size_map.keys())))
    pixel_size = model_image_size_map[model_name]
    return (pixel_size, pixel_size)

def get_base_model(model_name, trainable=False):
    """ Lấy base model từ url

    Args:
        model_name (str): Tên của model
        trainable (bool, optional): Có cho phép train base model hay không. Defaults to False.

    Returns:
        Model: Base model
    """    
    
    base_model_handle = get_model_handle(model_name)
    base_model = hub.KerasLayer(base_model_handle, trainable=trainable)
    return base_model

def create_preprocessing_layer(data_augmentation=False, data_normalization=True):
    preprocessing_layers = Sequential([], name="preprocessing_layers")
    if data_augmentation:
        preprocessing_layers.add(preprocessing.RandomFlip('horizontal'))
        preprocessing_layers.add(preprocessing.RandomHeight(0.2))
        preprocessing_layers.add(preprocessing.RandomWidth(0.2))
        preprocessing_layers.add(preprocessing.RandomZoom(0.2))
        preprocessing_layers.add(preprocessing.RandomRotation(0.2))
    if data_normalization:
        preprocessing_layers.add(preprocessing.Rescaling(1./255))
        
    return preprocessing_layers

def create_input_layer(model_name):
    image_size = get_model_input_size(model_name)
    input_layer = layers.Input(shape=image_size + (3,), name="input_layer")

###### Các hàm hỗ trợ đánh giá model, report ... #####