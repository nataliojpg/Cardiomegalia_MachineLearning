import streamlit as st
from PIL import Image
import numpy as np
import pickle
import cv2 as cv
import io


# Función para preprocesar la imagen
def limpiar_img(uploaded_file):
    try:
        image = np.array(uploaded_file)
        if len(image.shape) > 2 and image.shape[2] == 3:
            image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        elif len(image.shape) > 2 and image.shape[2] == 4:
            image = cv.cvtColor(image, cv.COLOR_RGBA2GRAY)
        
        image = cv.resize(image, (50, 50))
        image = cv.equalizeHist(image)
        image = np.expand_dims(image, axis=-1)
        image = image / 255.0
        
        return image
    
    except Exception as e:
        st.error(f"Error al procesar la imagen: {e}")
        return None




# Cargar modelo entrenado
def load_model():
    with open('../models/final_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Interfaz de usuario de Streamlit
st.title('Detección de Cardiomegalia')

uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Abrir la imagen con PIL
        image = Image.open(uploaded_file)
        st.image(image, caption='Imagen subida.', use_column_width=True)
        st.write("")

        # Preprocesar la imagen utilizando la función importada
        image = limpiar_img(image)
        image = np.expand_dims(image, axis=0) 

        # Realizar la predicción
        predic = model.predict(image)
        resultado = (predic[0][0] >= 0.5).astype(int)

        # Mostrar las predicciones
        if resultado == 1:
            resul_txt = ('El paciente tiene cardiomegalia')
        else:
            resul_txt = ('El paciente NO tiene cardiomegalia')
        st.write("Predicción:")
        st.write(resul_txt)

    except Exception as e:
        st.error(f"Error: {e}")
