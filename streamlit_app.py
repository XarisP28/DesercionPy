import streamlit as st
import shap
import matplotlib.pyplot as plt
from modeloshap import cargar_modelo, cargar_datos, explicar_prediccion

modelo = cargar_modelo()
X, y = cargar_datos()

explainer, shap_values = explicar_prediccion(modelo, X)

# 📊 Mostrar summary plot
st.subheader("📊 Importancia global de las variables (Summary Plot)")
fig, ax = plt.subplots()
shap.summary_plot(shap_values, X, show=False)
st.pyplot(fig)
