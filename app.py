import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="TP Estrellas", layout="wide")

st.title("🌟 Análisis de Estrellas - TP Data Mining")

@st.cache_data
def load_data():
    df = pd.read_csv("estrellas_limpias.csv")
    return df

df = load_data()

st.subheader("Vista previa del dataset")
st.dataframe(df.head())

st.subheader("Distribución por tipo estelar")
fig, ax = plt.subplots()
df["Star type"].value_counts().sort_index().plot(kind="bar", ax=ax)
ax.set_xlabel("Star type")
ax.set_ylabel("Cantidad")
st.pyplot(fig)
