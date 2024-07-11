# Librerías
import cv2 as cv
import pandas as pd
import numpy as np
from imutils import paths
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from keras.callbacks import  ModelCheckpoint
from data_processing import limpiar_img

#Ubicación imágenes de train
img_paths_train1= list(paths.list_images('../data/processed/train/true'))
img_paths_train0= list(paths.list_images('../data/processed/train/false'))
img_paths_test1= list(paths.list_images('../data/processed/test/true'))
img_paths_test0= list(paths.list_images('../data/processed/test/false'))

#Juntando y creación de labels
train_imgs = img_paths_train1 + img_paths_train0
train_labels = [1] * len(img_paths_train1) + [0] * len(img_paths_train0)
test_imgs = img_paths_test1 + img_paths_test0
test_labels = [1] * len(img_paths_test1) + [0] * len(img_paths_test0)

#División X/y, Train/Test
X_train = np.array([limpiar_img(img) for img in train_imgs])
y_train = np.array(train_labels)
X_test = np.array([limpiar_img(img) for img in test_imgs])
y_test = np.array(test_labels)

#Entrenamiento Modelo
modelo = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 1)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    Dropout(0.25),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Flatten(),
    Dense(50, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

checkpoint= keras.callbacks.ModelCheckpoint("callback_model.keras")
early_stopping = keras.callbacks.EarlyStopping(patience=10)

datagen = ImageDataGenerator(
    rotation_range=20,       
    width_shift_range=0.1,   
    height_shift_range=0.1,  
    shear_range=0.2,         
    zoom_range=0.2,          
    horizontal_flip=True,   
    fill_mode='nearest'      
)

datagen.fit(X_train)

history = modelo.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=70, validation_data=(X_test, y_test))

#Guardar modelo entrenado
import pickle

with open('final_model.pkl', 'wb') as file:
    pickle.dump(modelo, file)



