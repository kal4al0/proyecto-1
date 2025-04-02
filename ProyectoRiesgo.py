import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import kurtosis, skew, shapiro ,norm
from datetime import datetime

st.title("Evaluación del Riesgo Financiero de NVIDIA")

# Descarga los datos de los ultimos 10 años
@st.cache_data
def obtener_datos(stock):
    df = yf.download(stock, start="2010-01-01")["Close"]
    return df

# Calcula los rendimientos diarios
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


    ##################################################################################################################################
    ################################################# VaR y CVar Rolling Windows #####################################################


    # Función para calcular VaR paramétrico con ventanas móviles
    def rolling_var_param(returns, alpha=0.95, window=252):

        z_alpha = norm.ppf(1-alpha)  # Cuantil de la normal estándar (negativo)

        # Media y desviación estándar móviles
        rolling_mean = returns.rolling(window=window).mean()
        rolling_std = returns.rolling(window=window).std()

        # VaR paramétrico
        var = rolling_mean + z_alpha * rolling_std  # Esto es negativo (pérdida esperada)

        return var

    # Función para calcular CVaR paramétrico con ventanas móviles
    def rolling_cvar_param(returns, alpha=0.95, window=252):

        # Obtener z_alpha (cuantil de la normal)
        z_alpha = norm.ppf(1-alpha)

        # Media y desviación estándar móviles
        rolling_mean = returns.rolling(window=window).mean()
        rolling_std = returns.rolling(window=window).std()

        # VaR paramétrico
        var = rolling_var_param(returns, alpha, window)

        # CVaR (Expected Shortfall)
        cvar = rolling_mean - (norm.pdf(z_alpha) / (1-alpha)) * rolling_std

        return cvar

    # Función para calcular el VaR histórico con ventanas móviles
    def rolling_var_hist(returns, alpha=0.95, window=252):
        hVaR_list = [np.nan] * (window-1)

        for i in range(window, len(returns)+1):
            df_rendimientos = returns[stock_seleccionado].iloc[i-window:i]  # Ventana deslizante

            hVaR = df_rendimientos.quantile(1 - alpha)

            hVaR_list.append(hVaR)

        return pd.DataFrame(hVaR_list, index=returns.index)


    # Función para calcular el CVaR histórico con ventanas móviles
    def rolling_cvar_hist(returns, alpha=0.95, window=252):
        hCVaR_list = [np.nan] * (window-1)

        for i in range(window, len(returns)+1):
            df_rendimientos = returns[stock_seleccionado].iloc[i-window:i]  # Ventana deslizante

            var = df_rendimientos.quantile(1 - alpha)  # VaR histórico

            valores_extremos = df_rendimientos[df_rendimientos <= var]

            hCVaR = valores_extremos.mean() if not valores_extremos.empty else np.nan

            hCVaR_list.append(hCVaR)

        return pd.DataFrame(hCVaR_list, index=returns.index)
    
    # Var y CVar al 95
    NVDA_var_param_95 = rolling_var_param(df_rendimientos)
    NVDA_cvar_param_95 = rolling_cvar_param(df_rendimientos)

    # Var y CVar al 99
    NVDA_var_param_99 = rolling_var_param(df_rendimientos, alpha=0.99)
    NVDA_cvar_param_99 = rolling_cvar_param(df_rendimientos, alpha=0.99)

    # Var y CVar historico al 95
    NVDA_var_hist_95 = rolling_var_hist(df_rendimientos)
    NVDA_cvar_hist_95 = rolling_cvar_hist(df_rendimientos)

    # Var y CVar historico al 99
    NVDA_var_hist_99 = rolling_var_hist(df_rendimientos, alpha=0.99)
    NVDA_cvar_hist_99 = rolling_cvar_hist(df_rendimientos, alpha=0.99)

    # Asegúrate de haber hecho el desplazamiento de un día
    NVDA_var_param_95_1 = NVDA_var_param_95.shift(1)
    NVDA_var_param_99_1 = NVDA_var_param_99.shift(1)
    NVDA_var_hist_95_1 = NVDA_var_hist_95.shift(1)
    NVDA_var_hist_99_1 = NVDA_var_hist_99.shift(1)
    NVDA_cvar_param_95_1 = NVDA_cvar_param_95.shift(1)
    NVDA_cvar_param_99_1 = NVDA_cvar_param_99.shift(1)
    NVDA_cvar_hist_95_1 = NVDA_cvar_hist_95.shift(1)
    NVDA_cvar_hist_99_1 = NVDA_cvar_hist_99.shift(1)

    st.subheader("Análisis de VaR y CVaR mediante ventanas moviles a 95% y 99% de Confianza")


    # Crear la figura
    fig, ax = plt.subplots(figsize=(12, 6))

    # Graficar los rendimientos
    ax.plot(df_rendimientos[stock_seleccionado], label="Rendimientos", color="blue", alpha=0.6)
    ax.plot(NVDA_var_param_95_1, label="VaR paramétrico 95%", color="red", linestyle="dashed")
    ax.plot(NVDA_var_param_99_1, label="VaR paramétrico 99%", color="yellow", linestyle="dashed")
    ax.plot(NVDA_var_hist_95_1, label="VaR histórico 95%", color="green", linestyle="dashed")
    ax.plot(NVDA_var_hist_99_1, label="VaR histórico 99%", color="orange", linestyle="dashed")
    ax.plot(NVDA_cvar_param_95_1, label="CVaR paramétrico 95%", color="cyan", linestyle="dotted")
    ax.plot(NVDA_cvar_param_99_1, label="CVaR paramétrico 99%", color="purple", linestyle="dotted")
    ax.plot(NVDA_cvar_hist_95_1, label="CVaR histórico 95%", color="pink", linestyle="dotted")
    ax.plot(NVDA_cvar_hist_99_1, label="CVaR histórico 99%", color="lightblue", linestyle="dotted")

    # Mejorar la visualización
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")  # Línea en cero
    ax.legend()
    ax.set_title("Rendimientos vs. VaR y CVaR Paramétrico y Histórico (95% y 99%)")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Rendimiento")
    ax.grid(True, linestyle="--", alpha=0.5)

    # Mostrar gráfico en Streamlit
    st.pyplot(fig)


##################################################################################################################################
###################################################### Eficinecia  ###############################################################
    
    # Funcion para calcular la eficiencia
    def eficiencia_estimacion(df_rendimientos, risk_est, window=252):
        violaciones_risk = 0
        total_datos = len(df_rendimientos) - window

        for i in range(window, len(df_rendimientos) - 1):  # Ajuste para evitar el índice fuera de rango
            if df_rendimientos[stock_seleccionado].iloc[i + 1] < risk_est.iloc[i,0]:  # Comparar rendimiento del día siguiente
                violaciones_risk += 1

        porcentaje_violaciones_risk = (violaciones_risk / total_datos) * 100

        return violaciones_risk, porcentaje_violaciones_risk
    # Calcular violaciones y porcentajes para los VaR y CVaR
    violaciones_var_param_95, porcentaje_var_param_95 = eficiencia_estimacion(df_rendimientos, NVDA_var_param_95)
    violaciones_cvar_param_95, porcentaje_cvar_param_95 = eficiencia_estimacion(df_rendimientos, NVDA_cvar_param_95)

    violaciones_var_param_99, porcentaje_var_param_99 = eficiencia_estimacion(df_rendimientos, NVDA_var_param_99)
    violaciones_cvar_param_99, porcentaje_cvar_param_99 = eficiencia_estimacion(df_rendimientos, NVDA_cvar_param_99)

    violaciones_var_hist_95, porcentaje_var_hist_95 = eficiencia_estimacion(df_rendimientos, NVDA_var_hist_95)
    violaciones_cvar_hist_95, porcentaje_cvar_hist_95 = eficiencia_estimacion(df_rendimientos, NVDA_cvar_hist_95)

    violaciones_var_hist_99, porcentaje_var_hist_99 = eficiencia_estimacion(df_rendimientos, NVDA_var_hist_99)
    violaciones_cvar_hist_99, porcentaje_cvar_hist_99 = eficiencia_estimacion(df_rendimientos, NVDA_cvar_hist_99)

   # Crear un DataFrame con los resultados sin los índices
    resultados = pd.DataFrame({
        "Método": ["VaR Paramétrico 95%", "CVaR Paramétrico 95%", "VaR Paramétrico 99%", "CVaR Paramétrico 99%",
                "VaR Histórico 95%", "CVaR Histórico 95%", "VaR Histórico 99%", "CVaR Histórico 99%"],
        "Violaciones": [violaciones_var_param_95, violaciones_cvar_param_95, violaciones_var_param_99, 
                        violaciones_cvar_param_99, violaciones_var_hist_95, violaciones_cvar_hist_95,
                        violaciones_var_hist_99, violaciones_cvar_hist_99],
        "Porcentaje de Violaciones (%)": [porcentaje_var_param_95, porcentaje_cvar_param_95, porcentaje_var_param_99, 
                                        porcentaje_cvar_param_99, porcentaje_var_hist_95, porcentaje_cvar_hist_95,
                                        porcentaje_var_hist_99, porcentaje_cvar_hist_99]
    })

    # Resetear el índice y eliminarlo
    resultados = resultados.reset_index(drop=True)

    # Mostrar la tabla sin los índices
    st.subheader("Resultados de Violaciones y Porcentaje de Violaciones para VaR y CVaR")
    st.dataframe(resultados)

st.write("De acuerdo con los resultados obtenidos en la gráfica y la tabla, en general, nuestras estimaciones son precisas. Sin embargo, se observa que las métricas de VaR histórico y paramétrico al 99% de confianza presentan un mayor margen de error. Esto podría explicarse por el hecho de que se está utilizando un modelo basado en una distribución normal, cuando en realidad los rendimientos podrían seguir una distribución con colas más pesadas, lo cual es típico en los mercados financieros.")

st.write("Por otro lado, es evidente que las métricas con menos errores son las de CVaR. Esto no es sorprendente, ya que estas métricas están diseñadas para capturar los valores más extremos en la cola de la distribución, lo que las hace más precisas para detectar los eventos de riesgo más severos.")
