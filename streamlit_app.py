import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import pickle
from modeloshap import cargar_modelo, explicar_prediccion

# Configuración de la página
st.set_page_config(page_title="Proyecto Salud - SHAP", layout="wide")

st.title("📊 Análisis de Datos de Salud con SHAP")

# Cargar datos
st.subheader("📑 Datos")
df = pd.read_csv("data.csv")
st.write(f"El dataset contiene {df.shape[0]} registros y {df.shape[1]} columnas.")

if st.checkbox("Mostrar datos"):
    st.dataframe(df)

# Mostrar estadísticas
st.subheader("📈 Estadísticas Descriptivas")
st.write(df.describe())

# Gráfica interactiva
st.subheader("📊 Distribución de variables")

columna = st.selectbox("Selecciona una variable numérica:", df.select_dtypes(include=['float64', 'int64']).columns)

fig, ax = plt.subplots()
sns.histplot(df[columna], kde=True, ax=ax)
st.pyplot(fig)

# Cargar modelo
st.subheader("🤖 Predicción y Explicación con SHAP")

if st.button("Cargar modelo"):
    modelo = cargar_modelo()  # tu función para cargar modelo desde modeloshap.py
    st.success("Modelo cargado correctamente.")

    # Elegir un índice de muestra
    idx = st.slider("Selecciona el índice de muestra para explicar:", 0, len(df)-1, 0)

    # Explicar predicción
    explainer, shap_values = explicar_prediccion(modelo, df, idx)

    st.write("Predicción y valores SHAP para la muestra seleccionada:")
    st.write(df.iloc[idx])

    # Mostrar gráfico SHAP de barras
    st.subheader("📊 Valores SHAP para esta muestra:")
    fig2 = shap.plots.bar(shap_values[idx], show=False)
    st.pyplot(fig2)

    # Mostrar gráfico summary SHAP
    st.subheader("📊 Importancia global de las variables (Summary Plot):")
    fig3 = shap.summary_plot(shap_values, df, show=False)
    st.pyplot(fig3)

# Footer
st.markdown("---")
st.caption("Desarrollado por Xaris Pérez — Proyecto Salud 2025")
