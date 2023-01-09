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
import random
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
    print("################ ENVIRONMENT ################")
    print(f"Python Platform: {platform.platform()}")
    print(f"Python {sys.version}")
    print(f"Tensor Flow Version: {tf.__version__}")
    print(f"Tensor Flow Hub Version: {hub.__version__}")
    print(f"Keras Version: {tensorflow.keras.__version__}")

    gpu = len(tf.config.list_physical_devices('GPU'))>0
    print("GPU is", "available" if gpu else "NOT AVAILABLE")
    print("#############################################")

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
    """ Tạo preprocessing layer bao gồm data augmentation và data normalization

    Args:
        data_augmentation (bool, optional): Có sử dụng data augmentation hay không. Defaults to False.
        data_normalization (bool, optional): Có sử dụng data normalization hay không. Defaults to True.
    Returns:
        Layer: Preprocessing layer
    """
    
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
    """ Tạo input layer

    Args:
        model_name (str): Tên của model
        
    Returns:
        Layer: Input layer
    """
    
    image_size = get_model_input_size(model_name)
    input_layer = layers.Input(shape=image_size + (3,), name="input_layer")
    
    return input_layer

###### Các hàm hỗ trợ đánh giá model, report ... #####
def plot_loss_curves(results):
    """ Vẽ đồ thị loss và accuracy của model

    Args:
        results (dict): dictionary chứa loss và accuracy của model. Ví dụ:
            {"loss": [...],
            "accuracy": [...],
            "val_loss": [...],
            "val_accuracy": [...]}
    """
    loss = results["loss"]
    test_loss = results["accuracy"]

    accuracy = results["val_loss"]
    test_accuracy = results["val_accuracy"]

    epochs = range(len(results["loss"]))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    
def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15, norm=False, savefig=False): 
    """ Tạo confusion matrix để đánh giá model dựa trên y_true và y_pred 
    
    Nếu classes được truyền vào, confusion matrix sẽ được gán nhãn, 
    nếu không, giá trị class sẽ được gán nhãn bằng số nguyên

    Args:
        y_true: Mảng chứa nhãn thực tế (phải có cùng kích thước với y_pred).
        y_pred: Mảng chứa nhãn dự đoán (phải có cùng kích thước với y_true).
        classes: Mảng chứa nhãn của các class (ví dụ: dạng string). Nếu `None`, nhãn sẽ là số nguyên.
        figsize: Kích thước của figure (default=(10, 10)).
        text_size: Kích thước của chữ (default=15).
        norm: normalize values hay không (default=False).
        savefig: Lưu figure hay không (default=False).
    
    Returns:
        Một labelled confusion matrix được vẽ để so sánh y_true và y_pred.

    Example usage:
        make_confusion_matrix(y_true=test_labels, # ground truth test labels
                            y_pred=y_preds, # predicted labels
                            classes=class_names, # array of class label names
                            figsize=(15, 15),
                            text_size=10)
    """  
    # Create the confustion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalize it
    n_classes = cm.shape[0] # find the number of classes we're dealing with

    # Plot the figure and make it pretty
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=plt.cm.Blues) # colors will represent how 'correct' a class is, darker == better
    fig.colorbar(cax)

    # Are there a list of classes?
    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])
    
    # Label the axes
    ax.set(title="Confusion Matrix",
            xlabel="Predicted label",
            ylabel="True label",
            xticks=np.arange(n_classes), # create enough axis slots for each class
            yticks=np.arange(n_classes), 
            xticklabels=labels, # axes will labeled with class names (if they exist) or ints
            yticklabels=labels)
    
    # Make x-axis labels appear on bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    # Set the threshold for different colors
    threshold = (cm.max() + cm.min()) / 2.

    # Plot the text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if norm:
            plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
                horizontalalignment="center",
                color="white" if cm[i, j] > threshold else "black",
                size=text_size)
        else:
            plt.text(j, i, f"{cm[i, j]}",
                horizontalalignment="center",
                color="white" if cm[i, j] > threshold else "black",
                size=text_size)

    # Save the figure to the current working directory
    if savefig:
        fig.savefig("confusion_matrix.png")
        
def create_tensorboard_callback(dir_name, experiment_name):
    """ Tạo một callback TensorBoard để lưu lại các file log cho TensorBoard

    Args:
        dir_name (str): Tên thư mục chứa các file log của TensorBoard
        experiment_name (str): Tên của experiment

    Returns:
        TensorBoard callback
    """
    
    
    log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir
    )
    print(f"Saving TensorBoard log files to: {log_dir}")
    return tensorboard_callback

def create_learning_rate_callback():
    """ Tạo một callback để thay đổi learning rate theo thời gian

    Returns:
        LearningRateScheduler callback
    """
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10**(epoch/20))
    return lr_schedule

def plot_learning_rate_and_loss(history, epochs):
    # Vẽ biểu đồ learning rate với loss
    lrs = 1e-4 * (10 ** (np.arange(epochs)/20))
    plt.figure(figsize=(10, 7))
    plt.semilogx(lrs, history["loss"]) # muốn trục x (learning rate) theo thang log
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")
    plt.title("Learning rate vs. loss")


def compare_historys(original_history, new_history, initial_epochs=5):
    """ So sánh 2 history của model
        Thường là so sánh history của model gốc với history của model sau khi fine-tuning

    Args:
        original_history (dict): History của model gốc
        new_history (dict): History của model mới
        initial_epochs (int, optional): Số epochs ban đầu của model gốc. Defaults to 5.
    """
    
    # Get original history measurements
    acc = original_history.history["accuracy"]
    loss = original_history.history["loss"]

    val_acc = original_history.history["val_accuracy"]
    val_loss = original_history.history["val_loss"]

    # Combine original history with new history
    total_acc = acc + new_history.history["accuracy"]
    total_loss = loss + new_history.history["loss"]

    total_val_acc = val_acc + new_history.history["val_accuracy"]
    total_val_loss = val_loss + new_history.history["val_loss"]

    # Make plots
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label='Training Accuracy')
    plt.plot(total_val_acc, label='Validation Accuracy')
    plt.plot([initial_epochs-1, initial_epochs-1], plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label='Training Loss')
    plt.plot(total_val_loss, label='Validation Loss')
    plt.plot([initial_epochs-1, initial_epochs-1], plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()
    
def calculate_results(y_true, y_pred):
    """
    Tính toán độ chính xác, precision, recall và f1-score của một mô hình phân loại nhị phân.

    Args:
        y_true: true labels trong dạng 1D array
        y_pred: labels đuợc dự đoán bởi mô hình trong dạng 1D array
        
    Returns:
        Một dictionary của accuracy, precision, recall, f1-score.
    """
    # Calculate model accuracy
    model_accuracy = accuracy_score(y_true, y_pred) * 100
    # Calculate model precision, recall and f1 score using "weighted average
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    model_results = {"accuracy": model_accuracy,
                    "precision": model_precision,
                    "recall": model_recall,
                    "f1": model_f1}
    return model_results

# Tạo hàm vẽ ảnh ngẫu nhiên cùng với các dự đoán của nó
def plot_random_image(model, images, true_labels, classes):
    """Chọn một ảnh ngẫu nhiên, vẽ và dán nhãn nó với nhãn đúng và nhãn đã dự đoán.

    Args:
        model: mô hình đã huấn luyện (huấn luyện trên dữ liệu tương tự với dữ liệu trong ảnh.)
        images: tập hợp các ảnh ngẫu nhiên (ở dạng tensor).
        true_labels: mảng các nhãn gốc cho ảnh.
        classes: mảng các tên lớp cho anhe.
    
    Returns:
        Biểu đồ ảnh ngẫu nhiên từ `images` với nhãn lớp đã dự đoán từ `model`
        cũng như nhãn lớp đúng từ `true_labels`.
    """ 
    # Thiết lập random integer
    i = random.randint(0, len(images))
    
    # Tạo các mục tiêu và dự đoán
    target_image = images[i]
    pred_probs = model.predict(target_image.reshape(1, 28, 28)) # cần reshape để mô hình có đúng kích thước
    pred_label = classes[pred_probs.argmax()]
    true_label = classes[true_labels[i]]

    # Vẽ ảnh mục tiêu
    plt.imshow(target_image, cmap=plt.cm.binary)

    # Thay đổi màu của tiêu đề tùy xem dự đoán đúng hay sai
    if pred_label == true_label:
        color = "green"
    else:
        color = "red"

    # Thêm thông tin xlabel (prediction/true label)
    plt.xlabel("Pred: {} {:2.0f}% (True: {})".format(pred_label,
                                                    100*tf.reduce_max(pred_probs),
                                                    true_label),
                color=color) # đặt color là green hoặc red