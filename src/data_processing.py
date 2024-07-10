# Librerías
import cv2 as cv
import pandas as pd
import numpy as np
from imutils import paths
import os
import cv2 as cv
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.image import imread

#Data
img_paths_train1= list(paths.list_images('../data/train/true'))
img_paths_train0= list(paths.list_images('../data/train/false'))
img_paths_test1= list(paths.list_images('../data/test/true'))
img_paths_test0= list(paths.list_images('../data/test/false'))

#Crear etiquetas
train_imgs = img_paths_train1 + img_paths_train0
train_labels = [1] * len(img_paths_train1) + [0] * len(img_paths_train0)

test_imgs = img_paths_test1 + img_paths_test0
test_labels = [1] * len(img_paths_test1) + [0] * len(img_paths_test0)

#Crear DataFrames
train_df = pd.DataFrame({
    'image_name': train_imgs,
    'label': train_labels
})

test_df = pd.DataFrame({
    'image_name': test_imgs,
    'label': test_labels
})

# Función para cargar y preprocesar imágenes
def limpiar_img(img_path):
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE) 
    img = cv.resize(img, (50, 50))
    img = img.astype(np.uint8)  
    img = cv.equalizeHist(img)  
    img = np.expand_dims(img, axis=-1)  
    img = img / 255.0 
    return img

#Guardar en carpeta /data/processed

pro_dir = '../data/processed'

    #Estructura de directorios
for split in ['train', 'test']:
    for label in ['false', 'true']:
        os.makedirs(os.path.join(pro_dir, split, label), exist_ok=True)

    #Función para guardar las imgs en carpeta 
def procesar_y_guardar_imagenes(df, split):
    for index, row in df.iterrows():
        img_path = row['image_name']
        label = 'true' if row['label'] == 1 else 'false'
        img = limpiar_img(img_path)
        #Path para guardar
        filename = os.path.basename(img_path)
        pro_img_path = os.path.join(pro_dir, split, label, filename)
        cv.imwrite(pro_img_path, (img * 255).astype(np.uint8))  ######CHEQUEA SI IGUAL SE PUEDE USAR


procesar_y_guardar_imagenes(train_df, 'train')
procesar_y_guardar_imagenes(test_df, 'test')