import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
import plotly.express as px
from fpdf import FPDF
import base64

st.set_page_config(page_title="Predicción PM10 e IRAS", layout="wide")

# --- Encabezado con logos grandes ---
col1, col2, col3 = st.columns([2, 5, 2])
with col1:
    st.image("LOGO_CIATEC.png", use_column_width=True)
with col2:
    st.title("📊 Predicción de PM10 e IRAS")
with col3:
    st.image("LOGO_INNOVACION.webp", use_column_width=True)

# --- Definición de modelos LSTM ---
class PM10Model(nn.Module):
    def __init__(self, input_dim):
        super(PM10Model, self).__init__()
        self.lstm = nn.LSTM(input_dim, 158, num_layers=2, batch_first=True, dropout=0.299)
        self.fc1 = nn.Linear(158, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        x = out[:, -1, :]
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class IRASModel(nn.Module):
    def __init__(self, input_dim):
        super(IRASModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, 138, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(138, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        x = out[:, -1, :]
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# --- Función para generar secuencias ---
def create_sequences(data, window_size):
    return np.array([data[i:i + window_size] for i in range(len(data) - window_size + 1)])

# --- Sidebar: carga o selección de datos ---
st.sidebar.header("📂 Datos de entrada")
archivo = st.sidebar.file_uploader("Sube tu archivo Excel", type=["xlsx"])
usar_ejemplo = st.sidebar.checkbox("Usar datos de ejemplo")

# --- Sidebar: selección de periodo ---
st.sidebar.header("📅 Opciones de predicción")
opcion_prediccion = st.sidebar.radio(
    "Selecciona el tipo de análisis",
    ("Todo el periodo", "Por rango de fechas", "Por estación del año")
)

df = None
if archivo:
    df = pd.read_excel(archivo)
elif usar_ejemplo:
    try:
        df = pd.read_excel("datos_ejemplo.xlsx")
        st.success("Datos de ejemplo cargados correctamente.")
    except FileNotFoundError:
        st.error("⚠️ El archivo 'datos_ejemplo.xlsx' no se encontró. Sube el archivo manualmente o colócalo en el mismo directorio.")

if df is not None:
    df = df.dropna()
    if 'fecha' in df.columns:
        df['fecha'] = pd.to_datetime(df['fecha'])

        if opcion_prediccion == "Por rango de fechas":
            fecha_inicio = st.sidebar.date_input("Fecha inicio", df["fecha"].min().date())
            fecha_fin = st.sidebar.date_input("Fecha fin", df["fecha"].max().date())
            if fecha_inicio <= fecha_fin:
                df = df[(df["fecha"] >= pd.to_datetime(fecha_inicio)) & (df["fecha"] <= pd.to_datetime(fecha_fin))]
            else:
                st.sidebar.error("⚠️ La fecha de inicio no puede ser mayor que la fecha final.")

        elif opcion_prediccion == "Por estación del año":
            estacion = st.sidebar.selectbox("Selecciona una estación", ("Primavera", "Verano", "Otoño", "Invierno"))
            def get_estacion(mes):
                return (
                    "Primavera" if mes in [3, 4, 5] else
                    "Verano" if mes in [6, 7, 8] else
                    "Otoño" if mes in [9, 10, 11] else
                    "Invierno"
                )
            df['estacion'] = df['fecha'].dt.month.apply(get_estacion)
            df = df[df['estacion'] == estacion]

    fechas = df['fecha'].iloc[6:].values if 'fecha' in df.columns else pd.date_range(start="2025-01-01", periods=len(df), freq='D')[6:]

    # --- Predicción al presionar botón ---
    if st.button("🔍 Ejecutar predicción"):
        with st.spinner("Procesando y generando predicciones..."):
            scaler = MinMaxScaler()
            datos = df.drop(columns=["fecha", "estacion"], errors="ignore")
            scaled_data = scaler.fit_transform(datos.values)

            window_size = 7
            X_seq = create_sequences(scaled_data, window_size)
            X_tensor = torch.tensor(X_seq, dtype=torch.float32)
            input_dim = X_tensor.shape[2]
            device = torch.device("cpu")

            try:
                model_pm10 = PM10Model(input_dim=input_dim).to(device)
                model_pm10.load_state_dict(torch.load("modelo_final_PM10.pth", map_location=device))
                model_pm10.eval()
                pred_pm10 = model_pm10(X_tensor).detach().cpu().numpy().flatten()

                model_iras = IRASModel(input_dim=input_dim).to(device)
                model_iras.load_state_dict(torch.load("modelo_final_IRAS_1_4.pth", map_location=device))
                model_iras.eval()
                pred_iras = model_iras(X_tensor).detach().cpu().numpy().flatten()

                result_df = df.iloc[window_size - 1:].copy()
                result_df["Pred_PM10"] = pred_pm10
                result_df["Pred_IRAS_1_4"] = pred_iras
                result_df["fecha"] = fechas

                st.subheader("📊 Predicciones")
                st.dataframe(result_df[["fecha", "Pred_PM10", "Pred_IRAS_1_4"]])

                csv_data = result_df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Descargar predicciones", csv_data, file_name="predicciones_resultado.csv")

                # --- Gráficas interactivas con Plotly ---
                st.subheader("📈 Visualizaciones Interactivas")
                fig1 = px.line(result_df, x='fecha', y='Pred_PM10', title='PM10 Predicho')
                st.plotly_chart(fig1)
                fig2 = px.line(result_df, x='fecha', y='Pred_IRAS_1_4', title='IRAS 1-4 Predicho')
                st.plotly_chart(fig2)

                # --- Exportar informe en PDF ---
                if st.button("📄 Generar informe PDF"):
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)
                    pdf.cell(200, 10, txt="Informe de Predicción PM10 e IRAS", ln=1, align="C")
                    pdf.cell(200, 10, txt="Fecha de generación: " + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'), ln=2, align="L")
                    pdf.ln(10)
                    pdf.multi_cell(0, 10, txt="Resumen de predicciones generadas automáticamente por el sistema usando modelos LSTM.")
                    pdf.output("informe_predicciones.pdf")
                    with open("informe_predicciones.pdf", "rb") as file:
                        btn = st.download_button(
                            label="📥 Descargar informe PDF",
                            data=file,
                            file_name="informe_predicciones.pdf",
                            mime="application/pdf"
                        )

            except Exception as e:
                st.error(f"❌ Error al cargar los modelos o realizar la predicción: {e}")

# --- Agradecimientos e info ---
st.markdown("---")
st.markdown("### 🧠 Proyecto")
st.markdown("""
**Predicción y Estrategias Basadas en Ciencia de Datos para la Gestión de Riesgos Respiratorios Asociados a la Contaminación Atmosférica en el corredor industrial del bajío**

Esta herramienta permite procesar automáticamente datos ambientales de entrada para generar predicciones de contaminantes atmosféricos (PM10) y sus posibles efectos en salud (IRAS en menores de 5 años) usando modelos LSTM previamente entrenados.

Además, la app permite:
- Visualizar resultados por fechas
- Exportar los resultados predichos
- Comparar predicciones con valores reales
- Apoyar la toma de decisiones y estrategias de gestión de riesgos
""")

st.markdown("### 🤝 Agradecimientos")
st.markdown("""
Esta app fue desarrollada como parte del proyecto mencionado, con el apoyo del **Instituto de Innovación, Ciencia y Emprendimiento para la Competitividad** y el equipo del CIATEC A.C.

**Desarrollador:** Dr. Juan José Quiroz  
**Institución:** CIATEC A.C.  
**Contacto:** [jquiroz@ciatec.mx](mailto:jquiroz@ciatec.mx)
""")

st.markdown("### ℹ️ Instrucciones de uso")
st.markdown("""
1. Sube un archivo Excel con tus datos de entrada (incluyendo columna `fecha` si deseas visualización temporal), o usa el dataset de ejemplo.
2. Selecciona el tipo de predicción por periodo o estación.
3. Da clic en "Ejecutar predicción".
4. Visualiza y descarga tus resultados o genera un informe PDF.
""")

st.caption("© 2025 CIATEC A.C. | App desarrollada con Streamlit")
