import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.pyplot
import Funciones_MCF as MCF
import scipy.stats as stats
from scipy.stats import kurtosis, skew, shapiro ,norm
from datetime import datetime

st.title("Análisis estadístico rendimientos acciones NVIDIA")

@st.cache_data

#Obtenemos los datos desde 2010 al día actual que se esté consultando el precio de la acción
def obtener_datos(stocks):
    end_date = datetime.today().strftime("%Y-%m-%d")  # Obtiene la fecha actual
    df = yf.download(stocks, start="2010-01-01", end=end_date)['Close']
    return df

@st.cache_data
def calcular_rendimientos(df):
    return df.pct_change().dropna()


# Lista de acciones de ejemplo
stocks_lista = ['NVDA']

with st.spinner("Descargando datos..."):
    df_precios = obtener_datos(stocks_lista)
    df_rendimientos = calcular_rendimientos(df_precios)

# Selector de acción
stock_seleccionado = st.selectbox("Selecciona una acción", stocks_lista)

if stock_seleccionado:
    st.subheader(f"Métricas de Rendimiento: {stock_seleccionado}")
    
    rendimiento_medio = df_rendimientos[stock_seleccionado].mean()
    Kurtosis = kurtosis(df_rendimientos[stock_seleccionado])
    skew = skew(df_rendimientos[stock_seleccionado])
    
    col1, col2, col3= st.columns(3)
    col1.metric("Rendimiento Medio Diario", f"{rendimiento_medio:.4%}")
    col2.metric("Kurtosis", f"{Kurtosis:.4}")
    col3.metric("Skew", f"{skew:.2}")

    # Gráfico de rendimientos diarios
    st.subheader(f"Gráfico de Rendimientos: {stock_seleccionado}")
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(df_rendimientos.index, df_rendimientos[stock_seleccionado], label=stock_seleccionado)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    ax.legend()
    ax.set_title(f"Rendimientos de {stock_seleccionado}")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Rendimiento Diario")
    st.pyplot(fig)
    
    # Histograma de rendimientos
    st.subheader("Distribución de Rendimientos")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df_rendimientos[stock_seleccionado], bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(rendimiento_medio, color='red', linestyle='dashed', linewidth=2, label=f"Promedio: {rendimiento_medio:.4%}")
    ax.legend()
    ax.set_title("Histograma de Rendimientos")
    ax.set_xlabel("Rendimiento Diario")
    ax.set_ylabel("Frecuencia")
    st.pyplot(fig)

    st.subheader("Test de Normalidad (Shapiro-Wilk)")
    stat, p = shapiro(df_rendimientos[stock_seleccionado])

    st.write(f"**Shapiro-Wilk Test Statistic:** {stat:.4f}")
    st.write(f"**P-value:** {p:.4f}")

    # Interpretación del test
    alpha = 0.05
    if p > alpha:
        st.success("La distribución parece ser normal (No se rechaza H0)")
    else:
        st.error("La distribución NO es normal (Se rechaza H0)")

    st.subheader("Q-Q Plot de Rendimientos")

    fig, ax = plt.subplots(figsize=(6, 6))
    stats.probplot(df_rendimientos[stock_seleccionado], dist="norm", plot=ax)
    ax.set_title("Q-Q Plot de los Rendimientos")
    st.pyplot(fig)



    ############################################ METRICAS DE RIESGO #############################################################
    
    ############################################ CONFIANZA DEL 95% ###############################################################

    # VaR Parametrico
    mean = np.mean(df_rendimientos[stock_seleccionado])
    stdev = np.std(df_rendimientos[stock_seleccionado])
    VaR_95 = (norm.ppf(1-0.95,mean,stdev))

    # Historical VaR
    hVaR_95 = (df_rendimientos[stock_seleccionado].quantile(0.05))

    # Monte Carlo

    # Number of simulations
    n_sims = 100000

    # Simulate returns and sort
    sim_returns = np.random.normal(mean, stdev, n_sims)

    MCVaR_95 = np.percentile(sim_returns, 5)

    CVaR_95 = (df_rendimientos[stock_seleccionado][df_rendimientos[stock_seleccionado] <= hVaR_95].mean())
    st.subheader("Métricas de riesgo con confianza del 95%")
    
    col4, col5, col6, col7= st.columns(4)
    col4.metric("VaR paramétrico", f"{VaR_95:.4%}")
    col5.metric("Var histórico", f"{hVaR_95:.4%}")
    col6.metric("Var Monte Carlo", f"{MCVaR_95:.4%}")
    col7.metric("CVaR", f"{CVaR_95:.4%}")

    

    # Crear la figura y el eje
    fig, ax = plt.subplots(figsize=(13, 5))

    # Generar histograma
    n, bins, patches = ax.hist(df_rendimientos[stock_seleccionado], bins=50, color='blue', alpha=0.7, label='Returns')

    # Identificar y colorear de rojo las barras a la izquierda de hVaR_95
    for bin_left, bin_right, patch in zip(bins, bins[1:], patches):
        if bin_left < hVaR_95:
            patch.set_facecolor('red')

    # Marcar las líneas de VaR y CVaR
    ax.axvline(x=VaR_95, color='skyblue', linestyle='--', label='VaR 95% (Paramétrico)')
    ax.axvline(x=MCVaR_95, color='grey', linestyle='--', label='VaR 95% (Monte Carlo)')
    ax.axvline(x=hVaR_95, color='green', linestyle='--', label='VaR 95% (Histórico)')
    ax.axvline(x=CVaR_95, color='purple', linestyle='-.', label='CVaR 95%')

    # Configurar etiquetas y leyenda
    ax.set_title("Histograma de Rendimientos con VaR y CVaR")
    ax.set_xlabel("Rendimiento Diario")
    ax.set_ylabel("Frecuencia")
    ax.legend()

    # Mostrar la figura en Streamlit
    st.pyplot(fig)


 ############################################ CONFIANZA DEL 97.5% ###############################################################


    # VaR Parametrico

    VaR_97_5 = (norm.ppf(1-0.975,mean,stdev))

    # Historical VaR
    hVaR_97_5 = (df_rendimientos[stock_seleccionado].quantile(0.025))

    # Monte Carlo

    MCVaR_97_5 = np.percentile(sim_returns,2.5 )
 

    # CVar

    CVaR_97_5 = (df_rendimientos[stock_seleccionado][df_rendimientos[stock_seleccionado] <= hVaR_97_5].mean())
    st.subheader("Métricas de riesgo con confianza del 97.5%")
    
    col4, col5, col6, col7= st.columns(4)
    col4.metric("VaR paramétrico", f"{VaR_97_5:.4%}")
    col5.metric("Var histórico", f"{hVaR_97_5:.4%}")
    col6.metric("Var Monte Carlo", f"{MCVaR_97_5:.4%}")
    col7.metric("CVaR", f"{CVaR_97_5:.4%}")

    

    # Crear la figura y el eje
    fig, ax = plt.subplots(figsize=(13, 5))

    # Generar histograma
    n, bins, patches = ax.hist(df_rendimientos[stock_seleccionado], bins=50, color='blue', alpha=0.7, label='Returns')

    # Identificar y colorear de rojo las barras a la izquierda de hVaR_95
    for bin_left, bin_right, patch in zip(bins, bins[1:], patches):
        if bin_left < hVaR_97_5:
            patch.set_facecolor('red')

    # Marcar las líneas de VaR y CVaR
    ax.axvline(x=VaR_97_5, color='skyblue', linestyle='--', label='VaR 97.5% (Paramétrico)')
    ax.axvline(x=MCVaR_97_5, color='grey', linestyle='--', label='VaR 97.5% (Monte Carlo)')
    ax.axvline(x=hVaR_97_5, color='green', linestyle='--', label='VaR 97.5% (Histórico)')
    ax.axvline(x=CVaR_97_5, color='purple', linestyle='-.', label='CVaR 97.5%')

    # Configurar etiquetas y leyenda
    ax.set_title("Histograma de Rendimientos con VaR y CVaR")
    ax.set_xlabel("Rendimiento Diario")
    ax.set_ylabel("Frecuencia")
    ax.legend()

    # Mostrar la figura en Streamlit
    st.pyplot(fig)



    ############################################ CONFIANZA DEL 99% ###############################################################


    # VaR Parametrico

    VaR_99 = (norm.ppf(1-0.99,mean,stdev))

    # Historical VaR
    hVaR_99 = (df_rendimientos[stock_seleccionado].quantile(0.01))

    # Monte Carlo

    MCVaR_99 = np.percentile(sim_returns,1 )
 

    # CVar

    CVaR_99 = (df_rendimientos[stock_seleccionado][df_rendimientos[stock_seleccionado] <= hVaR_99].mean())
    st.subheader("Métricas de riesgo con confianza del 99%")
    
    col4, col5, col6, col7= st.columns(4)
    col4.metric("VaR paramétrico", f"{VaR_99:.4%}")
    col5.metric("Var histórico", f"{hVaR_99:.4%}")
    col6.metric("Var Monte Carlo", f"{MCVaR_99:.4%}")
    col7.metric("CVaR", f"{CVaR_99:.4%}")

    

    # Crear la figura y el eje
    fig, ax = plt.subplots(figsize=(13, 5))

    # Generar histograma
    n, bins, patches = ax.hist(df_rendimientos[stock_seleccionado], bins=50, color='blue', alpha=0.7, label='Returns')

    # Identificar y colorear de rojo las barras a la izquierda de hVaR_95
    for bin_left, bin_right, patch in zip(bins, bins[1:], patches):
        if bin_left < hVaR_99:
            patch.set_facecolor('red')

    # Marcar las líneas de VaR y CVaR
    ax.axvline(x=VaR_99, color='skyblue', linestyle='--', label='VaR 99% (Paramétrico)')
    ax.axvline(x=MCVaR_99, color='grey', linestyle='--', label='VaR 99% (Monte Carlo)')
    ax.axvline(x=hVaR_99, color='green', linestyle='--', label='VaR 99% (Histórico)')
    ax.axvline(x=CVaR_99, color='purple', linestyle='-.', label='CVaR 99%')

    # Configurar etiquetas y leyenda
    ax.set_title("Histograma de Rendimientos con VaR y CVaR")
    ax.set_xlabel("Rendimiento Diario")
    ax.set_ylabel("Frecuencia")
    ax.legend()

    # Mostrar la figura en Streamlit
    st.pyplot(fig)
