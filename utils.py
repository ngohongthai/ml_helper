"""
Contains various utility functions for PyTorch.
"""
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
import zipfile
import requests
from typing import List
from PIL import Image


def download_data(source: str, 
                  destination: str,
                  remove_source: bool = True) -> Path:
    """Downloads a zipped dataset from source and unzips to destination.

    Args:
        source (str): A link to a zipped file containing data.
        destination (str): A target directory to unzip data to.
        remove_source (bool): Whether to remove the source after downloading and extracting.
    
    Returns:
        pathlib.Path to downloaded data.
    
    Example usage:
        download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
                      destination="pizza_steak_sushi")
    """
    # Setup path to data folder
    data_path = Path("data/")
    image_path = data_path / destination

    # If the image folder doesn't exist, download it and prepare it... 
    if image_path.is_dir():
        print(f"[INFO] {image_path} directory exists, skipping download.")
    else:
        print(f"[INFO] Did not find {image_path} directory, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)
        # Download pizza, steak, sushi data
        target_file = Path(source).name
        with open(data_path / target_file, "wb") as f:
            request = requests.get(source)
            print(f"[INFO] Downloading {target_file} from {source}...")
            f.write(request.content)

        # Unzip pizza, steak, sushi data
        with zipfile.ZipFile(data_path / target_file, "r") as zip_ref:
            print(f"[INFO] Unzipping {target_file} data...") 
            zip_ref.extractall(data_path)

        # Remove .zip file
        if remove_source:
            os.remove(data_path / target_file)
    
    return image_path

def walk_through_dir(dir_path):
    """
    Walks through dir_path returning its contents.
    Args:
    dir_path (str): target directory

    Returns:
    A print out of:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        if (len(filenames) > 0):
            print(f"There are {len(filenames)} images in '{dirpath}'.")
            
def detail_dataset(train_dir: str, test_dir: str):
    all_classes = os.listdir(train_dir)
    print("There are {} classes in the dataset".format(len(all_classes)))
    print("="*50)
    title = ['class', 'train', 'test']
    print("{:<20} {:<20} {:<20}".format(*title))
    print("-"*50)
    total_train = 0
    total_test = 0
    for i in all_classes:
        print("{:<20} {:<20} {:<20}".format(i, len(os.listdir(train_dir+"/"+i)), len(os.listdir(test_dir+"/"+i))))
        total_test += len(os.listdir(test_dir+"/"+i))
        total_train += len(os.listdir(train_dir+"/"+i))
    print("-"*50)
    print("{:<20} {:<20} {:<20}".format("Total", total_train, total_test))
            
def get_random_images(num_images: int, 
                      data_path: str, 
                      seed: int = 42) -> List[str]:
    """Gets a list of random images from a directory.

    Args:
        num_images (int): number of random images to return.
        data_path (str): target directory to get random images from.
        seed (int, optional): random seed. Defaults to 42.
    
    Returns:
        List of random images from a directory.
    
    Example usage:
        get_random_images(num_images=5,
                          data_path="pizza_steak_sushi")
    """
    # Set random seed
    np.random.seed(seed)

    # Get all the images in data_path
    all_images = [image_path for image_path in Path(data_path).rglob("*.*")]

    # Get random images
    random_images = np.random.choice(all_images, size=num_images, replace=False)

    return random_images

def plot_random_images(num_images: int, 
                       data_path: str, 
                       seed: int = 42) -> None:
    """Plots a number of random images from a directory.

    Args:
        num_images (int): number of random images to plot.
        data_path (str): target directory to get random images from.
        seed (int, optional): random seed. Defaults to 42.
    
    Example usage:
        plot_random_images(num_images=5,
                           data_path="pizza_steak_sushi")
    """
    # Get random images
    random_images = get_random_images(num_images=num_images,
                                      data_path=data_path,
                                      seed=seed)

    # Plot random images
    plt.figure(figsize=(15, 15))
    for i, image_path in enumerate(random_images):
        ax = plt.subplot(1, num_images, i+1)
        image = plt.imread(image_path)
        plt.imshow(image)
        plt.title(image_path.parent.name)
        plt.axis("off")

def plot_random_augmented_image(num_images: int,
                                datasets: tf.keras.preprocessing.image.DirectoryIterator):
    plt.figure(figsize=(15, 15))
    for i in range(num_images):
        images, labels = datasets.next()
        random_number = np.random.randint(0, images.shape[0])
        ax = plt.subplot(1, num_images, i+1)
        plt.imshow(images[random_number])
        #plt.title(list(datasets.class_indices.keys())[int(labels[random_number])])
        plt.axis("off")
    
# Plot loss curves of a model
def plot_loss_curves(results):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
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
    

# def set_seeds(seed: int=42):
#     """Sets random sets for torch operations.

#     Args:
#         seed (int, optional): Random seed to set. Defaults to 42.
#     """
#     # Set the seed for general torch operations
#     torch.manual_seed(seed)
#     # Set the seed for CUDA torch operations (ones that happen on the GPU)
#     torch.cuda.manual_seed(seed)

# def save_model(model: torch.nn.Module,
#                target_dir: str,
#                model_name: str):
#     """Saves a PyTorch model to a target directory.

#     Args:
#     model: A target PyTorch model to save.
#     target_dir: A directory for saving the model to.
#     model_name: A filename for the saved model. Should include
#       either ".pth" or ".pt" as the file extension.

#     Example usage:
#     save_model(model=model_0,
#                target_dir="models",
#                model_name="05_going_modular_tingvgg_model.pth")
#     """
#     # Create target directory
#     target_dir_path = Path(target_dir)
#     target_dir_path.mkdir(parents=True,
#                         exist_ok=True)

#     # Create model save path
#     assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
#     model_save_path = target_dir_path / model_name

#     # Save the model state_dict()
#     print(f"[INFO] Saving model to: {model_save_path}")
#     torch.save(obj=model.state_dict(),
#              f=model_save_path)
