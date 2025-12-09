import os
from scripts.preprocess import resize_images
from scripts.train import create_model
from scripts.evaluate import plot_training_history
from scripts.predict import predict_image
from scripts.visualization import display_image
import tensorflow as tf
import numpy as np

input_dir = './data/raw'
output_dir = './data/processed'
resize_images(input_dir, output_dir)

IMG_SIZE = (145, 145)
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    './data/processed',
    labels='inferred',
    label_mode='int',
    image_size=IMG_SIZE,
    batch_size=batch_size,
    shuffle=True,
    validation_split=0.2,
    subset="training",
    seed=42
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    './data/processed',
    labels='inferred',
    label_mode='int',
    image_size=IMG_SIZE,
    batch_size=batch_size,
    shuffle=True,
    validation_split=0.2,
    subset="validation",
    seed=42
)

input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3)
model = create_model(input_shape)
history = model.fit(train_ds, validation_data=val_ds, epochs=10)
plot_training_history(history)

model_path = './models/my_model.h5'
model.save(model_path)

image_path = './data/processed/test_image.jpg'
result = predict_image(image_path, model_path)
print(f"추론 결과: {result}")

display_image(image_path)

