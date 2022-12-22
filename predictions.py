"""
Utility functions to make predictions.

Main reference for code creation: https://www.learnpytorch.io/06_pytorch_transfer_learning/#6-make-predictions-on-images-from-the-test-set 
"""
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import List, Tuple


# Predict on a target image with a target model
def pred_and_plot_image(model,
                        image_path: str,
                        image_shape: int = 224,
                        class_names: List[str] = None):
    # import the target image and preprocess it
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, size=[image_shape, image_shape])
    img = img/255
    
    # make prediction
    pred = model.predict(tf.expand_dims(img, axis=0))
    
    # Get the predicted class
    pred_class = class_names[int(tf.round(pred)[0][0])]
    
    plt.imshow(img)
    plt.title(f"Predicted: {pred_class} \n Prob: {pred[0][0]}")
    plt.axis(False)