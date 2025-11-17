import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import plotly.express as px

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
    return joblib.load("modelo_estelar.pkl")

df = load_data()
model = load_model()

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
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/8d/Hertzsprung-Russell_diagram.svg/640px-Hertzsprung-Russell_diagram.svg.png", width=220)
st.sidebar.title("Opciones")
section = st.sidebar.radio("Navegar a:", [
    "Ver dataset",
    "H-R Diagram",
    "Clustering (PCA + KMeans)",
    "Importancia del Modelo",
    "Predicción de Tipo Estelar",
    "Árbol de Decisión (Imagen)"
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
    col = st.number_input("Color codificado", 0, 10, 3)
    spec = st.number_input("Espectral codificado", 0, 10, 3)

    X_new = np.array([[temp, lum, rad, mag, col, spec]])
    pred = model.predict(X_new)[0]

    st.success(f"El modelo predice que la estrella es: **{star_names[pred]}**")

# -----------------------------
# 6. ÁRBOL DE DECISIÓN (IMAGEN)
# -----------------------------
if section == "Árbol de Decisión (Imagen)":
    st.subheader("🌳 Árbol de Decisión (Exportado)")
    st.image("arbol_estelar_coloreado.png", caption="Árbol de decisión exportado")
