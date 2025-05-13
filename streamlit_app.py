import streamlit as st
import shap
import matplotlib.pyplot as plt
from modeloshap import cargar_modelo, cargar_datos

modelo = cargar_modelo()

if modelo is None:
    st.error("No se pudo cargar el modelo. Verifica que 'modelo.pkl' esté en el directorio.")
else:
    X, y = cargar_datos()
    # Aquí sigue el resto de tu código
    st.write("Modelo y datos cargados correctamente.")

explainer, shap_values = explicar_prediccion(modelo, X)

# 📊 Mostrar summary plot
st.subheader("📊 Importancia global de las variables (Summary Plot)")
fig, ax = plt.subplots()
shap.summary_plot(shap_values, X, show=False)
st.pyplot(fig)
