import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import utils, data_setup, model_builder

# Setup hyperparameters
NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001
IMAGE_SHAPE = 224

# Setup directory
train_dir = "data/pizza_steak/train"
test_dir = "data/pizza_steak/test"

# Create datasets
train_datasets, test_datasets = data_setup.create_datasets(
    train_dir = train_dir, 
    test_dir = test_dir,
    target_size = (IMAGE_SHAPE, IMAGE_SHAPE),
    batch_size = BATCH_SIZE)

# Create model
model = model_builder.create_tiny_vgg(input_shape=(IMAGE_SHAPE, IMAGE_SHAPE, 3),
                                       hidden_units=HIDDEN_UNITS,
                                       out_features=2)

model.compile(
    loss="binary_crossentropy",
    optimizer=Adam(learning_rate=LEARNING_RATE),
    metrics=['accuracy']
)

model.fit(train_datasets,
                     epochs=NUM_EPOCHS,
                     steps_per_epoch=len(train_datasets),
                     validation_data=test_datasets,
                     validation_steps=len(test_datasets))

model.save("models/tiny_vgg_vs2")