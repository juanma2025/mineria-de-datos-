"""
Taller Minería de Datos - CRISP-DM
Regresión Lineal Múltiple: Dólar, Glucosa, Energía
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

os.makedirs("modelos", exist_ok=True)
os.makedirs("graficas", exist_ok=True)

# ─────────────────────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────────────────────
def separador(titulo):
    print("\n" + "="*60)
    print(f"  {titulo}")
    print("="*60)

def graficar_variables(df, features, target, titulo, filename):
    fig, axes = plt.subplots(1, len(features), figsize=(5*len(features), 4))
    if len(features) == 1:
        axes = [axes]
    colores = ['#2196F3', '#4CAF50', '#FF5722', '#9C27B0']
    for i, (ax, feat) in enumerate(zip(axes, features)):
        ax.scatter(df[feat], df[target], alpha=0.4, s=10,
                   color=colores[i % len(colores)], label='Datos')
        m, b = np.polyfit(df[feat], df[target], 1)
        x_line = np.linspace(df[feat].min(), df[feat].max(), 100)
        ax.plot(x_line, m*x_line + b, color='red', linewidth=1.5, label='Tendencia')
        ax.set_xlabel(feat, fontsize=11)
        ax.set_ylabel(target, fontsize=11)
        ax.set_title(f'{feat} vs {target}', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.suptitle(titulo, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"graficas/{filename}", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Gráfica guardada: graficas/{filename}")


# ─────────────────────────────────────────────────────────────
# EJERCICIO 1: PRECIO DEL DÓLAR
# ─────────────────────────────────────────────────────────────
separador("EJERCICIO 1 — Predicción del Precio del Dólar")

df_dolar = pd.read_csv("dolar_data.csv")
print("\n📋 Estadísticas descriptivas:")
print(df_dolar.describe().round(4))

# Fase: Preparación de datos
X1 = df_dolar[['Dia', 'Inflacion', 'Tasa_interes']].values
y1 = df_dolar['Precio_Dolar'].values
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

# Fase: Modelado
model_dolar = LinearRegression()
model_dolar.fit(X1_train, y1_train)

# Evaluación
y1_pred = model_dolar.predict(X1_test)
mse1  = mean_squared_error(y1_test, y1_pred)
rmse1 = np.sqrt(mse1)
r2_1  = r2_score(y1_test, y1_pred)

print("\n📊 Coeficientes del modelo:")
for feat, coef in zip(['Dia', 'Inflacion', 'Tasa_interes'], model_dolar.coef_):
    print(f"  {feat:15s}: {coef:+.4f}")
print(f"  Intercepto    : {model_dolar.intercept_:.4f}")

print(f"\n📈 Desempeño del modelo (Test set):")
print(f"  MSE  : {mse1:.4f}")
print(f"  RMSE : {rmse1:.4f}")
print(f"  R²   : {r2_1:.4f}  ({r2_1*100:.2f}% de varianza explicada)")

print("\n🔍 Interpretación de coeficientes:")
coefs = dict(zip(['Dia', 'Inflacion', 'Tasa_interes'], model_dolar.coef_))
print(f"  • Dia           : Por cada día adicional, el precio {'aumenta' if coefs['Dia']>0 else 'disminuye'} en {abs(coefs['Dia']):.4f} COP.")
print(f"  • Inflacion     : Un punto más de inflación {'aumenta' if coefs['Inflacion']>0 else 'disminuye'} el precio en {abs(coefs['Inflacion']):.4f} COP.")
print(f"  • Tasa_interes  : Una unidad más de tasa {'aumenta' if coefs['Tasa_interes']>0 else 'disminuye'} el precio en {abs(coefs['Tasa_interes']):.4f} COP.")

# Exportar modelo
joblib.dump(model_dolar, "modelos/modelo_dolar.joblib")
print("\n✅ Modelo exportado → modelos/modelo_dolar.joblib")

# Gráficas
graficar_variables(df_dolar, ['Dia','Inflacion','Tasa_interes'], 'Precio_Dolar',
                   'Ejercicio 1: Variables vs Precio del Dólar', 'dolar_scatter.png')


# ─────────────────────────────────────────────────────────────
# EJERCICIO 2: NIVEL DE GLUCOSA
# ─────────────────────────────────────────────────────────────
separador("EJERCICIO 2 — Predicción del Nivel de Glucosa")

df_glucosa = pd.read_csv("glucosa_data.csv")
print("\n📋 Estadísticas descriptivas:")
print(df_glucosa.describe().round(4))

X2 = df_glucosa[['Edad', 'IMC', 'Actividad_Fisica']].values
y2 = df_glucosa['Nivel_Glucosa'].values
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

model_glucosa = LinearRegression()
model_glucosa.fit(X2_train, y2_train)

y2_pred = model_glucosa.predict(X2_test)
mse2  = mean_squared_error(y2_test, y2_pred)
rmse2 = np.sqrt(mse2)
r2_2  = r2_score(y2_test, y2_pred)

print("\n📊 Coeficientes del modelo:")
for feat, coef in zip(['Edad', 'IMC', 'Actividad_Fisica'], model_glucosa.coef_):
    print(f"  {feat:20s}: {coef:+.4f}")
print(f"  Intercepto          : {model_glucosa.intercept_:.4f}")

print(f"\n📈 Desempeño del modelo (Test set):")
print(f"  MSE  : {mse2:.4f}")
print(f"  RMSE : {rmse2:.4f}")
print(f"  R²   : {r2_2:.4f}  ({r2_2*100:.2f}% de varianza explicada)")

# Importancia relativa (coeficientes estandarizados)
scaler = StandardScaler()
X2_scaled = scaler.fit_transform(df_glucosa[['Edad', 'IMC', 'Actividad_Fisica']])
model_std = LinearRegression().fit(X2_scaled, y2)
importancias = dict(zip(['Edad', 'IMC', 'Actividad_Fisica'], np.abs(model_std.coef_)))
var_max = max(importancias, key=importancias.get)

print("\n🔬 Importancia de variables (coeficientes estandarizados):")
for k, v in sorted(importancias.items(), key=lambda x: -x[1]):
    print(f"  {k:20s}: {v:.4f}")
print(f"\n  ★ Variable de mayor impacto: {var_max}")

coefs2 = dict(zip(['Edad', 'IMC', 'Actividad_Fisica'], model_glucosa.coef_))
print("\n🔍 Interpretación:")
print(f"  • Edad           : Un año más de edad {'incrementa' if coefs2['Edad']>0 else 'reduce'} la glucosa en {abs(coefs2['Edad']):.4f} mg/dL.")
print(f"  • IMC            : Una unidad más de IMC {'incrementa' if coefs2['IMC']>0 else 'reduce'} la glucosa en {abs(coefs2['IMC']):.4f} mg/dL.")
print(f"  • Actividad_Fis. : Una hora más de ejercicio {'incrementa' if coefs2['Actividad_Fisica']>0 else 'reduce'} la glucosa en {abs(coefs2['Actividad_Fisica']):.4f} mg/dL.")

joblib.dump(model_glucosa, "modelos/modelo_glucosa.joblib")
print("\n✅ Modelo exportado → modelos/modelo_glucosa.joblib")

graficar_variables(df_glucosa, ['Edad','IMC','Actividad_Fisica'], 'Nivel_Glucosa',
                   'Ejercicio 2: Variables vs Nivel de Glucosa', 'glucosa_scatter.png')


# ─────────────────────────────────────────────────────────────
# EJERCICIO 3: CONSUMO DE ENERGÍA
# ─────────────────────────────────────────────────────────────
separador("EJERCICIO 3 — Predicción del Consumo de Energía")

df_energia = pd.read_csv("energia_data.csv")
print("\n📋 Estadísticas descriptivas:")
print(df_energia.describe().round(4))

X3 = df_energia[['Temperatura', 'Hora', 'Dia_Semana']].values
y3 = df_energia['Consumo_Energia'].values
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.2, random_state=42)

model_energia = LinearRegression()
model_energia.fit(X3_train, y3_train)

y3_pred = model_energia.predict(X3_test)
mse3  = mean_squared_error(y3_test, y3_pred)
rmse3 = np.sqrt(mse3)
r2_3  = r2_score(y3_test, y3_pred)

print("\n📊 Coeficientes del modelo:")
for feat, coef in zip(['Temperatura', 'Hora', 'Dia_Semana'], model_energia.coef_):
    print(f"  {feat:15s}: {coef:+.4f}")
print(f"  Intercepto    : {model_energia.intercept_:.4f}")

print(f"\n📈 Desempeño del modelo (Test set):")
print(f"  MSE  : {mse3:.4f}")
print(f"  RMSE : {rmse3:.4f}")
print(f"  R²   : {r2_3:.4f}  ({r2_3*100:.2f}% de varianza explicada)")

scaler3 = StandardScaler()
X3_scaled = scaler3.fit_transform(df_energia[['Temperatura', 'Hora', 'Dia_Semana']])
model_std3 = LinearRegression().fit(X3_scaled, y3)
importancias3 = dict(zip(['Temperatura', 'Hora', 'Dia_Semana'], np.abs(model_std3.coef_)))
var_max3 = max(importancias3, key=importancias3.get)

print("\n🔬 Importancia de variables (coeficientes estandarizados):")
for k, v in sorted(importancias3.items(), key=lambda x: -x[1]):
    print(f"  {k:15s}: {v:.4f}")
print(f"\n  ★ Variable de mayor impacto: {var_max3}")

joblib.dump(model_energia, "modelos/modelo_energia.joblib")
print("\n✅ Modelo exportado → modelos/modelo_energia.joblib")

graficar_variables(df_energia, ['Temperatura','Hora','Dia_Semana'], 'Consumo_Energia',
                   'Ejercicio 3: Variables vs Consumo de Energía', 'energia_scatter.png')


# ─────────────────────────────────────────────────────────────
# RESUMEN FINAL
# ─────────────────────────────────────────────────────────────
separador("RESUMEN COMPARATIVO DE MODELOS")
print(f"\n{'Modelo':<20} {'MSE':>12} {'RMSE':>12} {'R²':>8}")
print("-"*56)
print(f"{'Precio Dólar':<20} {mse1:>12.4f} {rmse1:>12.4f} {r2_1:>8.4f}")
print(f"{'Nivel Glucosa':<20} {mse2:>12.4f} {rmse2:>12.4f} {r2_2:>8.4f}")
print(f"{'Consumo Energía':<20} {mse3:>12.4f} {rmse3:>12.4f} {r2_3:>8.4f}")
print("\n✅ Todos los modelos exportados en /modelos/")
print("✅ Todas las gráficas guardadas en /graficas/")
