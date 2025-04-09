import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Predicción PM10 e IRAS", layout="wide")
st.title("📊 Predicción de PM10 e IRAS")

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

# --- Upload del archivo ---
archivo = st.file_uploader("📎 Sube tu archivo Excel de entrada", type=["xlsx"])

if archivo:
    df = pd.read_excel(archivo)
    df = df.dropna()

    if 'fecha' in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"])
        fechas = df["fecha"].iloc[6:].values
    else:
        fechas = pd.date_range(start="2025-01-01", periods=len(df), freq='D')[6:]

    # --- Escalamiento ---
    scaler = MinMaxScaler()
    datos = df.drop(columns=["fecha"], errors="ignore")
    scaled_data = scaler.fit_transform(datos.values)

    # --- Secuencias LSTM ---
    window_size = 7
    X_seq = create_sequences(scaled_data, window_size)
    X_tensor = torch.tensor(X_seq, dtype=torch.float32)

    input_dim = X_tensor.shape[2]
    device = torch.device("cpu")

    # --- Cargar modelos entrenados (state_dict) ---
    try:
        model_pm10 = PM10Model(input_dim=input_dim).to(device)
        model_pm10.load_state_dict(torch.load("modelo_final_PM10.pth", map_location=device))
        model_pm10.eval()

        with torch.no_grad():
            pred_pm10 = model_pm10(X_tensor).cpu().numpy().flatten()

        model_iras = IRASModel(input_dim=input_dim).to(device)
        model_iras.load_state_dict(torch.load("modelo_final_IRAS_1_4.pth", map_location=device))
        model_iras.eval()

        with torch.no_grad():
            pred_iras = model_iras(X_tensor).cpu().numpy().flatten()

        # --- Resultados ---
        result_df = df.iloc[window_size - 1:].copy()
        result_df["Pred_PM10"] = pred_pm10
        result_df["Pred_IRAS_1_4"] = pred_iras
        result_df["fecha"] = fechas

        st.subheader("📊 Predicciones")
        st.dataframe(result_df[["fecha", "Pred_PM10", "Pred_IRAS_1_4"]])

        csv_data = result_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Descargar predicciones", csv_data, file_name="predicciones_resultado.csv")

        # --- Gráficas ---
        fig, ax = plt.subplots(2, 1, figsize=(10, 6))
        if "PM10" in result_df.columns:
            ax[0].plot(result_df["fecha"], result_df["PM10"], label="PM10 Real", linestyle="--")
        ax[0].plot(result_df["fecha"], result_df["Pred_PM10"], label="PM10 Predicho")
        ax[0].set_title("PM10")
        ax[0].legend()
        ax[0].grid(True)

        if "IRAS_1_4" in result_df.columns:
            ax[1].plot(result_df["fecha"], result_df["IRAS_1_4"], label="IRAS Real", linestyle="--")
        ax[1].plot(result_df["fecha"], result_df["Pred_IRAS_1_4"], label="IRAS Predicho")
        ax[1].set_title("IRAS 1-4")
        ax[1].legend()
        ax[1].grid(True)

        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Error al cargar los modelos o realizar la predicción: {e}")

✅ Fragmento agregado al final del app.py:
python
Copiar
Editar
# --- Agradecimientos, descripción e instrucciones ---
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
1. Sube un archivo Excel con tus datos de entrada (incluyendo columna `fecha` si deseas visualización temporal).
2. La app aplicará modelos LSTM para predecir valores de PM10 e IRAS.
3. Visualiza los resultados en tabla y gráficos interactivos.
4. Descarga el archivo con resultados si lo necesitas.
""")

st.caption("© 2025 CIATEC A.C. | App desarrollada con Streamlit")