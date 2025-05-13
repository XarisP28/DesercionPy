import streamlit as st
import shap
import matplotlib.pyplot as plt
from modeloshap import cargar_modelo, cargar_datos

modelo = cargar_modelo()

def explicar_prediccion(modelo, X):
    explainer = shap.Explainer(modelo, X)
    shap_values = explainer(X)
    return explainer, shap_values
    
if modelo is not None:
    X, y = cargar_datos()
    explainer, shap_values = explicar_prediccion(modelo, X)
    st.write("Modelo y explicaciones cargadas correctamente.")
else:
    st.error("No se pudo cargar el modelo. Verifica que 'modelo.pkl' esté en el directorio.")
explainer, shap_values = explicar_prediccion(modelo, X)

# 📊 Mostrar summary plot
st.subheader("📊 Importancia global de las variables (Summary Plot)")
fig, ax = plt.subplots()
shap.summary_plot(shap_values, X, show=False)
st.pyplot(fig)
