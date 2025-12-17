import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_model(input_shape):
    model = Sequential([
        Conv2D(8, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(),
        Dropout(0.1),
        Conv2D(16, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),
        Dropout(0.1),
        Flatten(),
        Dense(32, activation='relu'),
        Dropout(0.25),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
