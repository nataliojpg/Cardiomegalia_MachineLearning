#Librerías
import pickle
import numpy as np
from data_processing import limpiar_img

#Cargar modelo
with open('../models/final_model.pkl', 'rb') as file:
    model = pickle.load(file)

#Prueba con radiografía de paciente SIN la condición ✖️
img_neg = '../data/processed/test/false/1.png'
img_neg = limpiar_img(img_neg) 
img_neg = np.expand_dims(img_neg, axis=0) 
predicc_prueban = model.predict(img_neg)
resultadon = (predicc_prueban[0][0] >= 0.5).astype(int)

if resultadon == 1:
    print('El paciente tiene cardiomegalia')
else:
    print('El paciente NO tiene cardiomegalia')

#Prueba con radiografía de paciente CON la condición ✔
img_pos = '../data/processed/test/true/1.png'
img_pos = limpiar_img(img_pos)
img_pos = np.expand_dims(img_pos, axis=0)
predicc_pruebap = model.predict(img_pos)
resultadop = (predicc_pruebap[0][0] >= 0.5).astype(int)

if resultadop == 1:
    print('El paciente tiene cardiomegalia')
else:
    print('El paciente NO tiene cardiomegalia')