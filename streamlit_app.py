import streamlit as st
import joblib
import pandas as pd
import shap

# Función para cargar el modelo
def cargar_modelo():
    try:
        with open("modelo.pkl", "rb") as f:
            modelo = joblib.load(f)
        return modelo
    except FileNotFoundError:
        st.error("No se pudo cargar el modelo. Verifica que 'modelo.pkl' esté en el directorio.")
        return None

# Función para cargar los datos (opcional si quieres testear)
def cargar_datos():
    try:
        df = pd.read_csv("data.csv")
        # Aquí podrías procesar las columnas como en tu notebook
        return df
    except FileNotFoundError:
        st.error("No se pudo cargar 'data.csv'. Verifica que esté en el directorio.")
        return None

# ✅ ESTA es la función que te hacía falta
def explicar_prediccion(modelo, X):
    explainer = shap.Explainer(modelo, X)
    shap_values = explainer(X)
    return explainer, shap_values

# Main de la app
modelo = cargar_modelo()

if modelo:
    df = cargar_datos()
    if df is not None:
        # Aquí selecciona las features igual que en tu notebook
        features_a_usar = [
            'Facultad', 'Carrera', 'Sexo', 'Pais', 'Estado', 'Tipo',
            'Desc. Becas', 'Donativos', 'Deuda Actual', 'TPT',
            'N° CP', 'N° BA'
        ]
        X = df[features_a_usar]

        # Ahora sí, usamos la función explicativa
        explainer, shap_values = explicar_prediccion(modelo, X)

        st.write("Explainer creado con éxito.")
        st.write(explainer)

        # Mostrar un resumen de SHAP
        st.write("Valores SHAP calculados.")

    else:
        st.write("No se pudo cargar los datos.")
else:
    st.write("No se pudo cargar el modelo.")
