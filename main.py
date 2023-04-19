import numpy as np
import cv2
from emnist import extract_training_samples, extract_test_samples
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout


def load_emnist_data(fraction=0.1):
    # Загрузка датасета EMNIST
    X_train, y_train = extract_training_samples('byclass')
    X_test, y_test = extract_test_samples('byclass')

    # Выбор подмножества данных с использованием аргумента fraction
    num_train_samples = int(X_train.shape[0] * fraction)
    num_test_samples = int(X_test.shape[0] * fraction)

    train_indices = np.random.choice(X_train.shape[0], num_train_samples, replace=False)
    test_indices = np.random.choice(X_test.shape[0], num_test_samples, replace=False)

    X_train, y_train = X_train[train_indices], y_train[train_indices]
    X_test, y_test = X_test[test_indices], y_test[test_indices]

    # Предобработка данных
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = load_emnist_data(fraction=0.1)

def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(62, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = create_model()
model.summary()

# Обучение модели
history = model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_test, y_test))

# Сохранение обученной модели
model.save(r'C:\emnist_model.h5')

# Загрузка предобученной модели
model = load_model(r'C:\emnist_model.h5')

def preprocess_image(img_path):
    # Загрузка изображения и предобработка
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    img = img / 255
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    return img

def predict_text(img_path, model):
    img = preprocess_image(img_path)
    prediction = model.predict(img)
    return np.argmax(prediction, axis=1)

img_path = r'C:\one.png'

predicted_text = predict_text(img_path, model)
print("Распознанный текст:", predicted_text)

def interpret_prediction(prediction):
    if 0 <= prediction <= 9:
        return str(prediction)  # Цифры
    elif 10 <= prediction <= 35:
        return chr(prediction - 10 + ord('a'))  # Строчные буквы
    elif 36 <= prediction <= 61:
        return chr(prediction - 36 + ord('A'))  # Прописные буквы
    else:
        return "Invalid prediction"

interpreted_result = interpret_prediction(predicted_text)
print("Интерпретированный результат:", interpreted_result)
