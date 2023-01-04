"""
Contains functionality for creating Tensorflow Datasets
"""

import tensorflow as tf

BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
TRAIN_SUBSET = "train"
VALIDATION_SUBSET = "test"
LABEL_MODE = "categorical" # "binary" or "categorical"
seed = 123

def _build_dataset(data_dir, subset, image_size):
  return tf.keras.preprocessing.image_dataset_from_directory(
      data_dir,
      validation_split=.20,
      subset=subset,
      label_mode=LABEL_MODE,
      seed=seed,
      image_size=image_size,
      batch_size=1)

def build_train_dataset(data_dir, 
                        image_size=IMAGE_SIZE, 
                        batch_size=BATCH_SIZE, 
                        do_data_augmentation=False):
  train_ds = _build_dataset(data_dir, TRAIN_SUBSET, image_size)
  class_names = tuple(train_ds.class_names)
  train_size = train_ds.cardinality().numpy()
  train_ds = train_ds.unbatch().batch(batch_size)
  train_ds = train_ds.repeat()

  normalization_layer = tf.keras.layers.Rescaling(1. / 255)
  preprocessing_model = tf.keras.Sequential([normalization_layer])
  if do_data_augmentation:
    preprocessing_model.add(
        tf.keras.layers.RandomRotation(40))
    preprocessing_model.add(
        tf.keras.layers.RandomTranslation(0, 0.2))
    preprocessing_model.add(
        tf.keras.layers.RandomTranslation(0.2, 0))
    # Like the old tf.keras.preprocessing.image.ImageDataGenerator(),
    # image sizes are fixed when reading, and then a random zoom is applied.
    # If all training inputs are larger than image_size, one could also use
    # RandomCrop with a batch size of 1 and rebatch later.
    preprocessing_model.add(
        tf.keras.layers.RandomZoom(0.2, 0.2))
    preprocessing_model.add(
        tf.keras.layers.RandomFlip(mode="horizontal"))
  train_ds = train_ds.map(lambda images, labels:
                          (preprocessing_model(images), labels))
  return train_ds, train_size, class_names

def buil_validation_dataset(data_dir,
                            image_size=IMAGE_SIZE,
                            batch_size=BATCH_SIZE):
  val_ds = _build_dataset(data_dir, VALIDATION_SUBSET, image_size)
  normalization_layer = tf.keras.layers.Rescaling(1. / 255)
  valid_size = val_ds.cardinality().numpy()
  val_ds = val_ds.unbatch().batch(image_size)
  val_ds = val_ds.map(lambda images, labels:
                      (normalization_layer(images), labels))
  return val_ds, valid_size