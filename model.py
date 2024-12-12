import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

def create_model(input_shape=(145, 145, 3)):
    """
    CNN 모델을 생성합니다.
    
    Parameters:
        input_shape (tuple): 입력 이미지의 크기 (높이, 너비, 채널)
    
    Returns:
        model (tf.keras.Model): 생성된 CNN 모델
    """
    model = Sequential([
        # 첫 번째 Convolutional Layer
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),

        # 두 번째 Convolutional Layer
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),

        # 세 번째 Convolutional Layer
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.4),

        # Fully Connected Layer
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # 이진 분류 (졸음 여부 감지)
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    print(model.summary())  # 모델 요약 출력
    return model
