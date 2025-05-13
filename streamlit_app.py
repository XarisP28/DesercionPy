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
        # Ver qué columnas tiene el dataset
        st.write("Columnas disponibles en el dataset:", df.columns.tolist())

        # Aquí selecciona las features que existan en tu CSV
       features_a_usar = [
    'Facultad', 'Carrera', 'Sexo', 'Pais', 'Estado', 'Tipo',
    'Desc. Becas', 'Donativos', 'Deuda Actual', 'TPT',
    'N° CP', 'N° BA'
]

        # Validar qué columnas faltan para evitar el KeyError
        columnas_faltantes = [col for col in features_a_usar if col not in df.columns]
        if columnas_faltantes:
            st.error(f"Estas columnas no existen en el dataset: {columnas_faltantes}")
        else:
            X = df[features_a_usar]
            explainer, shap_values = explicar_prediccion(modelo, X)

            st.write("Explainer creado con éxito.")
            st.write(explainer)

            st.write("Valores SHAP calculados.")

    else:
        st.write("No se pudo cargar los datos.")
else:
    st.write("No se pudo cargar el modelo.")
