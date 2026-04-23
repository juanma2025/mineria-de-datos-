"""
Interfaz Web - Taller Minería de Datos
Predicción con modelos de Regresión Lineal Múltiple
"""

import streamlit as st
import joblib
import numpy as np
import os

# ── Configuración de página ──────────────────────────────────
st.set_page_config(
    page_title="Predicciones ML - Minería de Datos",
    page_icon="🧠",
    layout="centered"
)

# ── CSS personalizado ────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2rem; font-weight: 700;
        color: #1a237e; text-align: center; margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center; color: #555; margin-bottom: 2rem;
    }
    .result-box {
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        border-left: 5px solid #1565C0;
        border-radius: 8px; padding: 1.2rem;
        font-size: 1.3rem; font-weight: 600; color: #0d47a1;
        margin-top: 1rem;
    }
    .info-box {
        background: #f3f4f6; border-radius: 8px;
        padding: 1rem; font-size: 0.9rem; color: #374151;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Título ───────────────────────────────────────────────────
st.markdown('<div class="main-title">🧠 Predicciones con Regresión Lineal</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Taller Minería de Datos · CRISP-DM · Ingeniería de Software</div>', unsafe_allow_html=True)
st.divider()

# ── Selección de ejercicio ───────────────────────────────────
ejercicio = st.selectbox(
    "📌 Selecciona el modelo de predicción",
    options=[
        "💵 Ejercicio 1 — Precio del Dólar",
        "🩺 Ejercicio 2 — Nivel de Glucosa",
        "⚡ Ejercicio 3 — Consumo de Energía"
    ]
)

st.markdown("---")

# ── Rutas de modelos (ajustar si es necesario) ────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

# ═══════════════════════════════════════════════════════════════
# EJERCICIO 1 — PRECIO DEL DÓLAR
# ═══════════════════════════════════════════════════════════════
if ejercicio.startswith("💵"):
    st.subheader("💵 Predicción del Precio del Dólar (COP)")
    st.markdown(
        '<div class="info-box">'
        '<b>Modelo:</b> Regresión Lineal Múltiple<br>'
        '<b>Variables:</b> Día, Inflación, Tasa de interés<br>'
        '<b>R² del modelo:</b> 0.9963 &nbsp;|&nbsp; <b>RMSE:</b> 48.75 COP'
        '</div>', unsafe_allow_html=True
    )
    st.markdown("### Ingresa los valores")
    col1, col2, col3 = st.columns(3)
    with col1:
        dia = st.number_input("📅 Día (número)", min_value=1, max_value=1000, value=250, step=1)
    with col2:
        inflacion = st.number_input("📊 Inflación diaria", min_value=0.0, max_value=0.2,
                                     value=0.020, step=0.001, format="%.4f")
    with col3:
        tasa = st.number_input("💰 Tasa de interés (%)", min_value=0.0, max_value=20.0,
                                value=5.0, step=0.1)

    if st.button("🔮 Predecir Precio del Dólar", use_container_width=True, type="primary"):
        try:
            modelo = joblib.load(os.path.join(BASE, "modelos", "modelo_dolar.joblib"))
            pred = modelo.predict([[dia, inflacion, tasa]])[0]
            st.markdown(
                f'<div class="result-box">💵 Precio estimado del Dólar: <b>${pred:,.2f} COP</b></div>',
                unsafe_allow_html=True
            )
            st.info(f"Con Día={dia}, Inflación={inflacion:.4f}, Tasa={tasa:.2f}% → el modelo predice **${pred:,.2f} COP**")
        except FileNotFoundError:
            st.error("⚠️ Modelo no encontrado. Ejecuta primero `modelos_regresion.py`.")

# ═══════════════════════════════════════════════════════════════
# EJERCICIO 2 — GLUCOSA
# ═══════════════════════════════════════════════════════════════
elif ejercicio.startswith("🩺"):
    st.subheader("🩺 Predicción del Nivel de Glucosa (mg/dL)")
    st.markdown(
        '<div class="info-box">'
        '<b>Modelo:</b> Regresión Lineal Múltiple<br>'
        '<b>Variables:</b> Edad, IMC, Actividad Física<br>'
        '<b>R² del modelo:</b> 0.6814 &nbsp;|&nbsp; <b>RMSE:</b> 15.29 mg/dL<br>'
        '<b>Variable de mayor impacto:</b> Edad'
        '</div>', unsafe_allow_html=True
    )
    st.markdown("### Ingresa los valores del paciente")
    col1, col2, col3 = st.columns(3)
    with col1:
        edad = st.number_input("👤 Edad (años)", min_value=10, max_value=100, value=45, step=1)
    with col2:
        imc = st.number_input("⚖️ IMC (kg/m²)", min_value=10.0, max_value=50.0,
                               value=25.0, step=0.1)
    with col3:
        actividad = st.number_input("🏃 Actividad física (h/semana)",
                                     min_value=0.0, max_value=40.0, value=4.0, step=0.5)

    if st.button("🔮 Predecir Nivel de Glucosa", use_container_width=True, type="primary"):
        try:
            modelo = joblib.load(os.path.join(BASE, "modelos", "modelo_glucosa.joblib"))
            pred = modelo.predict([[edad, imc, actividad]])[0]
            color = "result-box"
            if pred > 200:
                alerta = "⚠️ Nivel MUY ALTO — consultar médico"
            elif pred > 140:
                alerta = "🟡 Nivel ELEVADO"
            elif pred < 70:
                alerta = "⬇️ Nivel BAJO"
            else:
                alerta = "✅ Nivel NORMAL"
            st.markdown(
                f'<div class="result-box">🩸 Nivel de Glucosa estimado: <b>{pred:.1f} mg/dL</b></div>',
                unsafe_allow_html=True
            )
            st.info(f"{alerta} &nbsp;|&nbsp; Edad={edad}, IMC={imc}, Actividad={actividad}h/sem → **{pred:.1f} mg/dL**")
        except FileNotFoundError:
            st.error("⚠️ Modelo no encontrado. Ejecuta primero `modelos_regresion.py`.")

# ═══════════════════════════════════════════════════════════════
# EJERCICIO 3 — ENERGÍA
# ═══════════════════════════════════════════════════════════════
elif ejercicio.startswith("⚡"):
    st.subheader("⚡ Predicción del Consumo de Energía (kWh)")
    st.markdown(
        '<div class="info-box">'
        '<b>Modelo:</b> Regresión Lineal Múltiple<br>'
        '<b>Variables:</b> Temperatura, Hora, Día de la semana<br>'
        '<b>R² del modelo:</b> 0.8968 &nbsp;|&nbsp; <b>RMSE:</b> 20.72 kWh<br>'
        '<b>Variable de mayor impacto:</b> Temperatura'
        '</div>', unsafe_allow_html=True
    )
    dias_semana = {
        "Lunes (1)": 1, "Martes (2)": 2, "Miércoles (3)": 3,
        "Jueves (4)": 4, "Viernes (5)": 5, "Sábado (6)": 6, "Domingo (7)": 7
    }
    st.markdown("### Ingresa las condiciones")
    col1, col2, col3 = st.columns(3)
    with col1:
        temp = st.number_input("🌡️ Temperatura (°C)", min_value=-10.0, max_value=50.0,
                                value=25.0, step=0.5)
    with col2:
        hora = st.slider("🕐 Hora del día", min_value=1, max_value=24, value=12)
    with col3:
        dia_label = st.selectbox("📅 Día de la semana", list(dias_semana.keys()))
        dia_num = dias_semana[dia_label]

    if st.button("🔮 Predecir Consumo de Energía", use_container_width=True, type="primary"):
        try:
            modelo = joblib.load(os.path.join(BASE, "modelos", "modelo_energia.joblib"))
            pred = modelo.predict([[temp, hora, dia_num]])[0]
            st.markdown(
                f'<div class="result-box">⚡ Consumo estimado: <b>{pred:.2f} kWh</b></div>',
                unsafe_allow_html=True
            )
            st.info(f"Temperatura={temp}°C, Hora={hora}h, {dia_label} → **{pred:.2f} kWh**")
        except FileNotFoundError:
            st.error("⚠️ Modelo no encontrado. Ejecuta primero `modelos_regresion.py`.")

# ── Footer ───────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small>Laboratorio Minería de Datos · CRISP-DM · "
    "Modelos entrenados con scikit-learn · Exportados con joblib</small></center>",
    unsafe_allow_html=True
)
