import pandas as pd
import numpy as np
import pickle
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Cargar y preparar datos
def cargar_datos():
    df = pd.read_csv('data.csv')

    df['Edad_Rango'] = pd.cut(df['F.Nac.'], bins=[17, 20, 25, 30, 35, 60], labels=[1, 2, 3, 4, 5])

    df['Facultad'] = df['Facultad'].astype('category').cat.codes
    df['Carrera'] = df['Carrera'].astype('category').cat.codes
    df['Sexo'] = df['Sexo'].map({'M': 0, 'F': 1})
    df['Pais'] = df['Pais'].astype('category').cat.codes
    df['Estado'] = df['Estado'].fillna('Extranjero').astype('category').cat.codes
    df['Tipo'] = df['Tipo'].astype('category').cat.codes
    df['Desc. Becas'] = df['Desc. Becas'].fillna('none').map({'none': 0, 'low': 1, 'medium': 2})
    df['Donativos'] = df['Donativos'].fillna('none').map({'none': 0, 'low': 1, 'medium': 2, 'high': 3})
    df['Deuda Actual'] = df['Deuda Actual'].fillna('none').map({'none': 0, 'low': 1, 'medium': 2, 'high': 3})
    df['TPT'] = df['TPT'].map({'1.Favorable': 1, '2.Limitado': 2, '3.Moderado': 3, '4.Riesgo': 4, '5.Invalidado': 5})

    df['N_Mat_Rango'] = pd.cut(df['N° Mat'], bins=[0, 4, 8, 12, 20], labels=[1, 2, 3, 4])
    df['N_AC_Rango'] = pd.cut(df['N° AC'], bins=[-1, 2, 5, 8, 11], labels=[1, 2, 3, 4])
    df['N_NA_Rango'] = pd.cut(df['N° NA'], bins=[-1, 0, 2, 4, 7], labels=[1, 2, 3, 4]).cat.add_categories(0).fillna(0)
    df['N_Cr_Rango'] = pd.cut(df['N° Cr'], bins=[0, 20, 40, 60, 100], labels=[1, 2, 3, 4])
    df['N_Cr_NA_Rango'] = pd.cut(df['N° Cr.NA'], bins=[-1, 0, 10, 20, 60], labels=[1, 2, 3, 4])

    features = [
        'Facultad', 'Carrera', 'Sexo', 'Pais', 'Estado', 'Tipo',
        'Desc. Becas', 'Donativos', 'Deuda Actual', 'TPT',
        'Edad_Rango', 'N_Mat_Rango', 'N_AC_Rango', 'N_NA_Rango',
        'N_Cr_Rango', 'N_Cr_NA_Rango','N° CP','N° BA'
    ]

    X = df[features]
    y = df['target']

    return X, y

# Entrenar y guardar modelo
def entrenar_modelo():
    X, y = cargar_datos()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo = RandomForestClassifier()
    modelo.fit(X_train, y_train)
    with open("modelo.pkl", "wb") as f:
        pickle.dump(modelo, f)
    return modelo, X_test

# Cargar modelo desde archivo
import os
import joblib

def cargar_modelo():
    if os.path.exists("modelo.pkl"):
        with open("modelo.pkl", "rb") as f:
            modelo = joblib.load(f)
        return modelo
    else:
        print("Archivo modelo.pkl no encontrado")
        return None

# Explicar predicción con SHAP
def explicar_prediccion(modelo, X_test):
    explainer = shap.Explainer(modelo, X_test)
    shap_values = explainer.shap_values(X_test)

    # Detectar estructura de shap_values
    if isinstance(shap_values, list):
        # Clasificación binaria: retornar solo para clase 1
        valores_shap = shap_values[1]
    else:
        # Regresión u otro caso
        valores_shap = shap_values

    return explainer, valores_shap

