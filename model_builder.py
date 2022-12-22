"""
Contains PyTorch model code to instantiate a TinyVGG model.
"""
import tensorflow as tf

width = int
height = int
channels = int

def create_tiny_vgg(input_shape: (width, height, channels), hidden_units: int, out_features: int) -> tf.keras.Model:
    """Creates the TinyVGG architecture.

    Replicates the TinyVGG architecture from the CNN explainer website in PyTorch.
    See the original architecture here: https://poloclub.github.io/cnn-explainer/

    Args:
    """
    model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(filters=hidden_units,
                              kernel_size=3,
                              activation="relu",
                              input_shape=input_shape),
      tf.keras.layers.Conv2D(filters=hidden_units,kernel_size=3,activation="relu"),
      tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
      tf.keras.layers.Conv2D(filters=hidden_units,kernel_size=3,activation="relu"),
      tf.keras.layers.Conv2D(filters=hidden_units,kernel_size=3,activation="relu"),
      tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(1 if out_features == 2 else out_features, 
                            activation="sigmoid" if out_features == 2 else "softmax"), 
      # sigmoid for binary classification, softmax for multiclass                       
    ])
    
    return model
