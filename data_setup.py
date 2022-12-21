"""
Contains functionality for creating Tensorflow Datasets
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator

BATCH_SIZE = 32
CLASS_MODE = "binary" # "categorical" or "binary", "sparse", "input", None
TARGET_SIZE = (224, 224)

default_train_datagen = ImageDataGenerator(rescale=1/255.,
                                             rotation_range=20, # rotate the image slightly between 0 and 20 degrees (note: this is an int not a float)
                                             shear_range=0.2, # shear the image
                                             zoom_range=0.2, # zoom into the image
                                             width_shift_range=0.2, # shift the image width ways
                                             height_shift_range=0.2, # shift the image height ways
                                             horizontal_flip=True) # flip the image on the horizontal axis

# Create ImageDataGenerator training instance without data augmentation
default_test_datagen = ImageDataGenerator(rescale=1/255.) 

def create_datasets(
    train_dir: str, 
    test_dir: str,
    train_datagen: ImageDataGenerator = default_train_datagen,
    test_datagen: ImageDataGenerator = default_test_datagen,
    target_size:(int, int) = TARGET_SIZE,
    batch_size: int = BATCH_SIZE, 
    class_mode: str = CLASS_MODE, # categorical or binary
    shuffle: bool = True,
):
  """Creates training and testing datasets.

  Takes in a training directory and testing directory path and turns
  them into tensorflow datasets

  Args:
    train_dir: Path to training directory.
    test_dir: Path to testing directory.
    train_datagen: tensorflow transform to perform on training data.
    test_datagen: tensorflow transform to perform on testing data.
    batch_size: Number of samples per batch in each of the dataset.
    target_size: Image size after transformation.
    class_mode: One of "categorical", "binary",
    shuffle: Whether to shuffle the data.

  Returns:
    A tuple of (train_datasets, test_datasets).
  """

  train_data = train_datagen.flow_from_directory(train_dir,
                                               target_size=target_size,
                                               batch_size=batch_size,
                                               class_mode=class_mode,
                                               shuffle=shuffle) 
  
  test_data = test_datagen.flow_from_directory(test_dir,
                                             target_size=target_size,
                                             batch_size=batch_size,
                                             class_mode=class_mode,
                                             shuffle=False)
  
  return train_data, test_data
