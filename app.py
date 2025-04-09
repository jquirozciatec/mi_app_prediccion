#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import torch.nn as nn

# ---------------- MODELOS ----------------
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

# ---------------- APP ----------------
st.title("Predicción de PM10 e IRAS")

uploaded_file = st.file_uploader("Sube tu archivo Excel de entrada", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df = df.dropna()

    if 'fecha' in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"])
        fechas = df["fecha"].iloc[6:].values
    else:
        fechas = pd.date_range(start="2025-01-01", periods=len(df), freq='D')[6:]

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df.drop(columns=['fecha'], errors='ignore').values)

    def create_sequences(data, window_size):
        return np.array([data[i:i + window_size] for i in range(len(data) - window_size + 1)])

    window_size = 7
    X_seq = create_sequences(scaled_data, window_size)
    X_tensor = torch.tensor(X_seq, dtype=torch.float32)

    device = torch.device("cpu")
    model_pm10 = PM10Model(input_dim=X_tensor.shape[2]).to(device)
    model_pm10.load_state_dict(torch.load("modelo_final_PM10.pth", map_location=device))
    model_pm10.eval()

    with torch.no_grad():
        pred_pm10 = model_pm10(X_tensor).cpu().numpy().flatten()

    model_iras = IRASModel(input_dim=X_tensor.shape[2]).to(device)
    model_iras.load_state_dict(torch.load("modelo_final_IRAS_1_4.pth", map_location=device))
    model_iras.eval()

    with torch.no_grad():
        pred_iras = model_iras(X_tensor).cpu().numpy().flatten()

    result_df = df.iloc[window_size - 1:].copy()
    result_df["Pred_PM10"] = pred_pm10
    result_df["Pred_IRAS_1_4"] = pred_iras
    result_df["fecha"] = fechas

    st.write("### Predicciones:")
    st.dataframe(result_df[["fecha", "Pred_PM10", "Pred_IRAS_1_4"]])

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

