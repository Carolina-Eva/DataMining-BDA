import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import plotly.express as px
from sklearn.tree import plot_tree


# -----------------------------
# CONFIGURACIÓN DEL DASHBOARD
# -----------------------------
st.set_page_config(
    page_title="Stellar Dashboard",
    page_icon="✨",
    layout="wide"
)

st.title("🌌 Dashboard Interactivo de Análisis Estelar")
st.markdown("Este dashboard integra ETL, Machine Learning, Clustering y Visualización Astronómica.")

# -----------------------------
# CARGA DEL DATASET Y MODELO
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("estrellas_limpias.csv")
    return df

@st.cache_resource
def load_model():
    return joblib.load("modelo_estelar_RF.pkl")

@st.cache_resource
def load_tree():
    return joblib.load("modelo_arbol_estelar.pkl")

# Cargar árboles y modelos
df = load_data()
model = load_model()
modelo_arbol = load_tree()

# Acceder a los elementos guardados en el árbol
tree = modelo_arbol["tree"]
scaler = modelo_arbol["scaler"]
le_color = modelo_arbol["encoder_color"]
le_spec = modelo_arbol["encoder_spec"]
features = modelo_arbol["features"]
class_names = modelo_arbol["class_names"]

# Paleta de colores
star_palette = {
    0: "#8c564b",
    1: "#d62728",
    2: "#1f77b4",
    3: "#2ca02c",
    4: "#9467bd",
    5: "#ff7f0e"
}

star_names = {
    0: "Brown Dwarf",
    1: "Red Dwarf",
    2: "White Dwarf",
    3: "Main Sequence",
    4: "Supergiant",
    5: "Hypergiant"
}

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.image("star.png", width=200)
st.sidebar.title("Opciones")
section = st.sidebar.radio("Navegar a:", [
    "Ver dataset",
    "H-R Diagram",
    "Clustering (PCA + KMeans)",
    "Importancia del Modelo",
    "Predicción de Tipo Estelar",
    "Predicción con Árbol",
    "Árbol de Decisión"
])

# -----------------------------
# 1. VER DATASET
# -----------------------------
if section == "Ver dataset":
    st.subheader("📄 Dataset procesado")
    st.dataframe(df)
    st.markdown(f"Total de registros: **{len(df)}**")

# -----------------------------
# 2. H-R DIAGRAM
# -----------------------------
if section == "H-R Diagram":
    st.subheader("🔥 Hertzsprung–Russell Diagram")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x="Temperature (K)",
        y="Luminosity(L/Lo)",
        hue="Star type",
        palette=star_palette,
        s=70,
        ax=ax
    )
    ax.set_yscale("log")
    ax.invert_xaxis()
    st.pyplot(fig)

# -----------------------------
# 3. CLUSTERING + PCA
# -----------------------------
if section == "Clustering (PCA + KMeans)":
    st.subheader("🔭 PCA 2D + Clustering KMeans")

    fig = px.scatter(
        df,
        x="PC1",
        y="PC2",
        color=df["cluster"].astype(str),
        title="PCA + KMeans",
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Tabla Clusters vs Tipos Reales")
    st.write(pd.crosstab(df["cluster"], df["Star type"]))

# -----------------------------
# 4. IMPORTANCIA DEL MODELO
# -----------------------------
if section == "Importancia del Modelo":
    st.subheader("📈 Importancia de Variables (Random Forest)")
    features = [
        "Temperature (K)",
        "Luminosity(L/Lo)",
        "Radius(R/Ro)",
        "Absolute magnitude(Mv)",
        "color_encoded",
        "spectral_encoded"
    ]

    importances = model.feature_importances_

    fig2 = px.bar(
        x=importances,
        y=features,
        orientation="h",
        title="Importancia de las características",
        labels={"x": "Importancia", "y": "Variable"}
    )
    st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# 5. PREDICCIÓN
# -----------------------------
if section == "Predicción de Tipo Estelar":
    st.subheader("🔮 Predicción con el Modelo Entrenado")

    temp = st.number_input("Temperatura (K)", 1000.0, 50000.0, 5800.0)
    lum = st.number_input("Luminosidad (L/Lo)", 0.001, 100000.0, 1.0)
    rad = st.number_input("Radio (R/Ro)", 0.001, 1000.0, 1.0)
    mag = st.number_input("Magnitud Absoluta (Mv)", -10.0, 20.0, 4.8)

    col = st.selectbox("Star color", le_color.classes_)
    spec = st.selectbox("Spectral Class", le_spec.classes_)

    # Codificar variables categóricas
    col_enc = le_color.transform([col])[0]
    spec_enc = le_spec.transform([spec])[0]

    # Matriz de entrada
    X_new = np.array([[temp, lum, rad, mag, col_enc, spec_enc]])

    # ESCALAR
    X_new_scaled = scaler.transform(X_new)

    # Predicción
    pred = model.predict(X_new_scaled)[0]

    st.success(f"⭐ El modelo predice que la estrella es: **{star_names[pred]}**")


if section == "Predicción con Árbol":
    st.subheader("🔮 Predicción usando Árbol de Decisión")

    temp = st.number_input("Temperatura (K)", 1000.0, 50000.0, 5800.0)
    lum = st.number_input("Luminosidad (L/Lo)", 0.001, 100000.0, 1.0)
    rad = st.number_input("Radio (R/Ro)", 0.001, 1000.0, 1.0)
    mag = st.number_input("Magnitud Absoluta (Mv)", -10.0, 20.0, 4.8)

    # Opciones originales
    col = st.selectbox("Star color", le_color.classes_)
    spec = st.selectbox("Spectral Class", le_spec.classes_)

    # Encoding
    col_enc = le_color.transform([col])[0]
    spec_enc = le_spec.transform([spec])[0]

    X_new = np.array([[temp, lum, rad, mag, col_enc, spec_enc]])

    # Escalar
    X_scaled_new = scaler.transform(X_new)

    # Predicción
    pred_tree = tree.predict(X_scaled_new)[0]

    st.success(f"🌟 El Árbol predice: **{class_names[pred_tree]}**")

# -----------------------------
# 6. ÁRBOL DE DECISIÓN (IMAGEN)
# -----------------------------
if section == "Árbol de Decisión":
    st.subheader("🌳 Árbol de Decisión")

    fig, ax = plt.subplots(figsize=(22, 12))
    plot_tree(
        tree,
        feature_names=features,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=10
    )
    st.pyplot(fig)

