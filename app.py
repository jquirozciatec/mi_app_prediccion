import streamlit as st
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="Predicción PM10 e IRAS", layout="wide")

st.title("📊 Predicción de PM10 e IRAS")
st.markdown("Sube tu archivo Excel de entrada para predecir PM10 e IRAS (usa modelos entrenados).")

@st.cache_resource
def cargar_modelos():
    modelo_pm10 = torch.load("modelo_final_PM10.pth", map_location=torch.device("cpu"))
    modelo_iras = torch.load("modelo_final_IRAS_1_4.pth", map_location=torch.device("cpu"))
    return modelo_pm10.eval(), modelo_iras.eval()

modelo_pm10, modelo_iras = cargar_modelos()

archivo = st.file_uploader("📎 Sube tu archivo Excel de entrada", type=["xlsx"])

if archivo:
    df = pd.read_excel(archivo)

    st.write("### Vista previa del archivo:")
    st.dataframe(df.head())

    try:
        X = df.drop(columns=["fecha"]) if "fecha" in df.columns else df
        entradas = torch.tensor(X.values).float()

        pred_pm10 = modelo_pm10(entradas).detach().numpy().flatten()
        pred_iras = modelo_iras(entradas).detach().numpy().flatten()

        df_resultado = df.copy()
        df_resultado["Pred_PM10"] = pred_pm10
        df_resultado["Pred_IRAS_1_4"] = pred_iras

        st.write("## 📈 Predicciones:")

        if "fecha" in df_resultado.columns:
            df_resultado["fecha"] = pd.to_datetime(df_resultado["fecha"])
            col1, col2 = st.columns(2)
            with col1:
                fecha_inicio = st.date_input("Desde:", df_resultado["fecha"].min())
            with col2:
                fecha_fin = st.date_input("Hasta:", df_resultado["fecha"].max())

            mask = (df_resultado["fecha"] >= pd.to_datetime(fecha_inicio)) & \
                   (df_resultado["fecha"] <= pd.to_datetime(fecha_fin))
            df_resultado = df_resultado[mask]

        st.dataframe(df_resultado)

        output_csv = df_resultado.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Descargar predicciones (.csv)", output_csv, file_name="predicciones_resultado.csv")

        if "fecha" in df_resultado.columns:
            st.write("### 📊 Gráfica PM10 vs IRAS")

            fig, ax1 = plt.subplots(figsize=(10, 4))

            ax1.plot(df_resultado["fecha"], df_resultado["Pred_PM10"], color='tab:blue', label='PM10')
            ax1.set_ylabel("PM10", color='tab:blue')
            ax1.tick_params(axis='y', labelcolor='tab:blue')

            ax2 = ax1.twinx()
            ax2.plot(df_resultado["fecha"], df_resultado["Pred_IRAS_1_4"], color='tab:red', label='IRAS')
            ax2.set_ylabel("IRAS", color='tab:red')
            ax2.tick_params(axis='y', labelcolor='tab:red')

            fig.tight_layout()
            st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Error en el procesamiento del archivo: {e}")

# --- Agradecimientos e instrucciones ---
st.markdown("---")
st.markdown("### 🤝 Agradecimientos")
st.markdown("""
Esta app fue desarrollada como parte del proyecto de modelado predictivo ambiental y salud.  
Agradecemos el apoyo de CIATEC A.C. y de todos los colaboradores que hicieron posible este trabajo.

**Desarrollador:** Dr. Juan José Quiroz  
**Institución:** CIATEC A.C.  
**Contacto:** [juan.quiroz@ciatec.mx](mailto:juan.quiroz@ciatec.mx)
""")

st.markdown("### ℹ️ Instrucciones de uso")
st.markdown("""
1. Sube un archivo Excel con tus datos de entrada (incluyendo columna `fecha` si deseas visualización temporal).
2. La app aplicará los modelos entrenados para predecir valores de PM10 e IRAS.
3. Visualiza los resultados en tabla y gráficos interactivos.
4. Descarga el archivo con resultados si lo necesitas.
""")

st.caption("© 2025 CIATEC A.C. | App desarrollada con Streamlit")
