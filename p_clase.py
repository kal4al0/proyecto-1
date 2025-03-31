import pandas as pd
import numpy as np 
import Funciones_MCF as MCF
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew, shapiro


'''
Streamlit de la clase 

'''

st.title('Clase 1')
st.header('Visualizacion de Activos')


M7 = ['AAPL','MSFT','NVDA','TSLA','GOOGL','NFLX','AMZN']
with st.spinner('Descargando data...'):
    df_precios = MCF.obtener_datos(M7)
    df_rendimientos = MCF.calcular_rendimientos(df_precios)

stock_seleccionado = st.selectbox('Selecciona una Accion',M7)

if stock_seleccionado:

    st.subheader(f'Metricas : {stock_seleccionado}')

    promedio_rendi_diario = df_rendimientos[stock_seleccionado].mean()
    kurtosis = kurtosis(df_rendimientos[stock_seleccionado])
    skew = skew(df_rendimientos[stock_seleccionado])
    col1,col2,col3 = st.columns(3)

    col1.metric("Rendimiento Medio Diario", f"{promedio_rendi_diario:.4%}")
    col2.metric("Kurtosis",f"{kurtosis:.4}")
    col3.metric("Skew",f"{skew:.3}")

    st.subheader(f'Gráfico de Rendimientos : {stock_seleccionado}')

    fig,ax = plt.subplots(figsize = (10,5))
    ax.plot(df_rendimientos.index,df_rendimientos[stock_seleccionado])
    ax.axhline(y = 0,linestyle = '-',alpha = 0.7)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Rendimiento Diario")
    st.pyplot(fig)

    # Histograma 
    st.subheader(f'Histograma de Rendimientos : {stock_seleccionado}')
    fig,ax = plt.subplots(figsize =(10,5))
    ax.hist(df_rendimientos[stock_seleccionado],bins=50,alpha=0.5,color = 'green')
    ax.set_title('Histograma')
    ax.set_xlabel('Rendimiento Diario')
    ax.set_ylabel('Frecuencia')
    st.pyplot(fig)

    st.subheader('Test de Normalidad (Shapiro-Wilk)')

    stat , p = shapiro(df_rendimientos[stock_seleccionado])
    st.write(f'Shapirop-Wilk Test : {stat:.4}' )
    st.write(f'P_value : {p:.6}')
















