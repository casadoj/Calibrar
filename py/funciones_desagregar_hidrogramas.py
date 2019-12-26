
# coding: utf-8

# _Autor:_    __Jesús Casado__ <br> _Revisión:_ __16/07/2018__ <br>
# 
# __Introducción__<br>
# En este código se incluyen las funciones generadas para desagregar un hidrograma en sus dos componentes: flujo rápido (escorrentía superficial) y flujo lento (flujo base).
# 
# __Funciones__<br>
# Evento:
# -  `extract_event`: recorta el 'data frame' de caudales a las fechas indicadas, calcula la pendiente y curvatura y muestra su evolución temporal si así se desea.
# 
# Curvas de recesión:
# -  `k_recession`: calcula la *k*, constante de recesión, de un evento de recesión de caudal.
# -  `Qt`: calcula el caudal base por medio de la ley de recesión, dados el caudal en un momento y la *k*
# 
# Desagregación de hidrogramas:
# -  `key_points2`: encuentra los puntos de inicio, pico, inflexión y fin de cada evento de escorrentía superficial en una hidrograma.
# -  `mml`: desagrega el hidrograma mediante el método de los mínimos locales
# -  `mlr`: desagrega el hidrograma mediante el método de la línea recta
# -  `mbf`: desagrega el hidrograma mediante el método de la base fija
# -  `mpv`: desagrega el hidrograma mediante el método de la pendiente variable
# 
# __Cosas a corregir__ <br>
# 
# __Índice__<br>
# 
# __[1. Evento](#1.-Evento)__<br>
# 
# __[2. Curvas de recesión](#2.-Curvas-de-recesión)__<br>
# 
# __[3. Desagregación de hidrogramas](#3.-Desagregación-de-hidrogramas)__<br>
# [3.1. Puntos clave](#3.1.-Puntos-clave)<br>
# [3.2. Métodos de desagregación](#3.2.-Métodos-de-desagregación)<br>

# In[1]:


import numpy as np

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('seaborn-whitegrid')

import pandas as pd
import datetime
from calendar import monthrange
from math import ceil, floor, log, exp

import os

import itertools,operator

import sys
sys._enablelegacywindowsfsencoding() # leer archivos con caracteres especiales

#from scipy.interpolate import griddata


# ## 1. Evento

# In[4]:


def extract_event(data, start=None, end=None, fillna=False, smoothing=None, window=12, alpha=0.2, max_freq=50, 
                  plot=True):
    """Sobre una serie de datos de caudal de cualquier resolución, se extraen la serie correspondiente a las fechas indicadas, se suaviza la serie mediante una media móvil, y se calculan tres nuevos campos sobre la serie suavizada: 'dQ' es la derivada del caudal con respecto al tiempo (la pendiente del hidrograma), 'm' es el signo de la pendiente (-1:decreciente, 0:nulo, 1:creciente), y 'd2Q' es la segunda derivada del caudal respecto al tiempo (la curvatura del hidrograma).
    
    Parámetros:
    -----------
    data:      data frame. Las filas representan pasos temporales y tendrá al menos un campo llamado 'Q' con la serie de caudal
    start:     datetime. Inicio de la serie a extraer. Si es 'None' no se recorta.
    end:       datetime. Fin de la serie a extraer. Si es 'None' no se recorta.
    fillna:    boolean. Si se quieren rellenar los huecos mediante interpolación lineal
    smoothing: boolean. Si se quiere suavizar la serie temporal. Por defecto 'None', no se suaviza la serie. Los métodos disponibles son la media móvil ('ma'), exponencial ('ex') o transformada de Fourier ('ft'). Dependiendo de qué método se escoja, deberá definirse otro parámetro: 'window', 'alpha' o 'max_freq'
               respectivamente
    window:    integer. Ancho de la ventan móvil. Por defecto se ponen 12 para hacer la media horaria a partir de
               datos cincuminutales
    alpha:     float. Define la fuerza que se le da al dato original frente al suavizado. Debe estar entre 0 y 1; si es
               1 no hay suavizado; si es 0 se obtiene una serie constante con el valor inicial de la serie original           
    max_freq:  integer. Máxima frecuencia de la descomposición de Fourier que se utilizará en la inversión de la
               transformada para suavizar la serie. Cuanto menor sea este parámetro mayor será el suavizado
    plot:      boolean. Si se quieren plotear las series de caudal, pendiente, signo de pendiente y curvatura
    
    Salidas:
    --------
    flood:     data frame. Corte de 'data' con los tres nuevos campos: 'dQ', 'm' ,'d2Q'
    Si 'plot=True' se muestra una figura con cuatro gráficos de las series de caudal, pendiente, signo de la pendiente y
    curvatura.
    """
    
    if start and end:
        # Extraer la serie para el evento
        flood = data.loc[start:end,:].copy()
    else:
        flood = data.copy()
        start, end = flood.index[0], flood.index[-1]
        
    # Corregir huecos en la serie mediante interpolación lineal
    if fillna == True:
        # Pasos temporales
        At = (flood.index[1] - flood.index[0]).total_seconds()
        delta = datetime.timedelta(seconds=At)
        # 'Data frame' auxiliar con los huecos en la serie
        aux = flood.loc[np.isnan(flood.Q),:]
        # Definir el inicio ('d1') y fin ('d2') de cada hueco
        for i, d1 in enumerate(aux.index):
            for j, d2 in enumerate(aux.index[i+1:]):
                if int((d2 - d1).total_seconds() / 3600) != j + 1:
                    d2 = aux.index[j]
                    break
            # Paso previo ('st') y posterior ('en') al hueco
            st, en = d1 - datetime.timedelta(seconds=At), d2 + datetime.timedelta(seconds=At)
            tt = (en - st).total_seconds() / At
            # Caudal observado en 'st' y 'en'
            Qst, Qen = flood.Q[st], flood.Q[en]
            # Interpolación lineal
            for t, d in enumerate(data[d1:d2].index):
                flood.Q[d] = Qst + (Qen - Qst) * (t + 1) / tt
    
    # Suavizado de la serie
    # ---------------------
    if smoothing == 'ma':
        flood = moving_average_smoothing(flood, window=window, plot=False)
        serie = 'Qma'
    elif smoothing == 'ex':
        flood = exponential_smoothing(flood, alpha=alpha, plot=False)
        serie = 'Qex'
    elif smoothing == 'ft':
        flood = fourier_smoothing(flood, max_freq=max_freq, plot=False)
        serie = 'Qft'
    else:
        serie = 'Q'
    
    # Calcular la derivada del caudal
    flood['dQ'] = np.nan
    for i in flood.index[1:]:
        flood.loc[i, 'dQ'] = (flood[serie][i] - flood[serie][i-1])

    # Calcular la derivada del caudal
    flood['d2Q'] = np.nan
    for i in flood.index[2:]:
        flood.loc[i, 'd2Q'] = (flood.dQ[i] - flood.dQ[i-1])

    # Signo de la pendiente
    flood['m'] = np.nan
    flood.loc[flood.dQ < 0, 'm'] = -1
    flood.loc[flood.dQ == 0, 'm'] = 0
    flood.loc[flood.dQ > 0, 'm'] = 1
    
    if plot == True:
        # Visualizar
        fig, ax = plt.subplots(nrows=4, figsize=(18,10))
        # Caudal
        if smoothing != None:
            ax[0].plot(flood.Q, linewidth=1, c='steelblue', label='Qo')
            ymax = ceil(flood.Q[start:end].max() / 10) * 10
            ax[0].plot(flood[serie], '--k', linewidth=1, label=serie)
        else:
            ax[0].plot(flood.Q, linewidth=1, c='steelblue', label='Q')
            ymax = ceil(flood.Q[start:end].max() / 10) * 10
        ax[0].set(xlim=(start, end), ylim=(0,ymax))
        ax[0].set_ylabel('Q (m³/s)', fontsize=13)
        ax[0].legend(fontsize=12)
        # Derivada del caudal
        ax[1].plot(flood.dQ, linewidth=1, c='orange', label='dQ')
        ymax = ceil(flood.loc[start:end, 'dQ'].max() / 1) * 1
        ymin = floor(flood.loc[start:end, 'dQ'].min() / 1) * 1
        ax[1].set(xlim=(start, end), ylim=(ymin,ymax))
        ax[1].set_ylabel('dQ/dt (m³/s²)', fontsize=13)
        # Signo de la pendiente
        ax[2].plot(flood.m, linewidth=1, c='red', label='dQ')
        ax[2].set(xlim=(start, end))#, ylim=(ymin,ymax))
        ax[2].set_ylabel('signo (-)', fontsize=13);
        # Segunda derivada del caudal
        ax[3].plot(flood.d2Q, linewidth=1, c='green', label='dQ')
        ymax = ceil(flood.loc[start:end, 'd2Q'].max() / 0.25) * 0.25
        ymin = floor(flood.loc[start:end, 'd2Q'].min() / 0.25) * 0.25
        ax[3].set(xlim=(start, end), ylim=(ymin,ymax))
        ax[3].set_ylabel('d²Q/dt² (m³/s³)', fontsize=13);
    
    return flood


# ## 2. Curvas de recesión

# In[2]:


def k_recession(data, tf, to=None, timesteps=96):
    """Se calcula la constante de decaimiento de la recesión de caudales.
    
    Parámetros:
    -----------
    data:      series. La serie de caudal a analizar
    tf:        datetime. Final (o punto posterior al inicio) de la curva de recesión.
    to:        datetime. Inicio de la curva de recesión. Opcional, si no se incluye, el 'to' se calcula como el momento 'timesteps' previos a 'tf'
    timesteps: integer. Número de pasos temporales entre 'to' y 'tf'. Sólo se usa si no se especifica 'to'
    
    Salidas:
    --------
    k:         float. Constante de decaimiento (s)
    """
    
    # Incremento temporal en segundos
    idxf = data.index.get_loc(tf)
    tf = data.index[idxf]
    if to == None: # si no se introduce directamente 'to'
        to = tf - timesteps
    else:
        idxo = data.index.get_loc(to)
        to = data.index[idxo]
    At = tf - to
    At = At.total_seconds()
    # Caudal de inicio y fin
    Qo, Q = data[to], data[tf]
    if Qo <= Q:
        print('ERROR: caudal final superior al inicial')
        return
    
    # Constante de decaimiento exponencial
    k = At / log(Qo / Q)
    if k <= 0:
        k = np.nan
    
    return k


# In[4]:


def Qt(Qo, to, t, k):
    """Esta función genera el caudal en un tiempo 't' mediante la curva maestra de recesión del caudal base.
    
    Parámetros:
    -----------
    Qo:        float. Valor de caudal base al inicio de la curva de recesión (m³/s).
    to:        datetime. Fecha y hora correspondiente a 'Qo'.
    t:         datetime. Fecha y hora a la que se quiere calcular el caudal.
    k:         constante de decaimiento exponencial (s).
    
    Salidas:
    --------
    Q:         float. Valor de caudal base en el momento 't' (m³/s)."""
    
    # Calcular el incremento temporal en segundos
    At = t - to
    At = At.total_seconds()
    # Calcular el caudal en el tiempo t
    Q = Qo * exp(-At / k)
    
    return Q


# ## 3. Desagregación de hidrogramas
# ### 3.1. Puntos clave

# In[ ]:


def encontrar_ml(data, A, tipo=1, m=1.5):
    """Econtrar los mínimos locales de una serie (en principio de caudal).
    
    Entradas:
    ---------
    data:     series. Serie temporal (en principio de caudal)
    A:        float. Área de la cuenca hidrográfica de la estación de aforo (km²)
    tipo:     integer. Método empleado para la búsqueda de los mínimos locales: 1, método de los mínimos locales; 2, del tipo 1 se le eliminan los mínimos locales con un caudal superior a 'm' veces el caudal mínimo; 3, al tipo 2 se le añaden mínimos intermedios; 4, al tipo 1 se le eliminan mínimos locales que superan 'm' veces el caudal medio entre los mínimos locales adyacentes 
    m:        float. Factor multiplicador del caudal mínimo utilizado en los tipos 2 y 3 para reducir el número de mínimos locales
    
    Salidas:
    --------
    valles:   list. Fechas en las que acontece un mínimo local en la serie"""
    
    # Calcular el número de días a observar antes y después
    N = int(round(0.8 * A**0.2))
    
    # Iniciar la serie de mínimos locales con el primer intervalo (sea o no mínimo local)
    valles1 = [data.index[0]]
    # Encontrar mínimos locales en la serie de caudal
    for t in data.index[N:-N]:
        if data[t] == data[t-N:t+N].min():
            valles1.append(t)
    # Añadir el último punto de la serie
    valles1.append(data.index[-1])
    
    # TIPO 1
    # ------
    if tipo == 1:
        return valles1
    
    # TIPO 2|3
    # --------
    elif (tipo == 2) or (tipo == 3):
        # reducir el número de mínimos locales
        valles2 = [valles1[0]]
        for i, valle in enumerate(valles1[1:-1]):
            if data[valle] <= m * data.min():
                valles2.append(valle)
        valles2.append(valles1[-1])
        
        # Calcular el caudal base mediante 'mml'
        Qslow, Qquick = mml(data, valles2, plot=False)
        
        # TIPO 2
        # ------
        if tipo == 2:
            return valles2
        
        # TIPO 3
        # ------
        else:
            # agregar mínimos sobre 'valles2'
            valles_add = []
            for i in range(len(valles2) - 1): # intervalos de 'valles2'
                # buscar mínimos locales (de 'valles') dentro del intervalo
                l1, l2 = valles2[i], valles2[i+1]
                laux = []
                for valle in valles1:
                    if (valle > l1) and (valle < l2):
                        laux.append(valle)

                # seleccionar mínimos a añadir
                if len(laux) == 0:
                    continue
                else:
                    for j in range(1, len(laux) - 1):
                        a0, a1, a2 = laux[j-1], laux[j], laux[j+1]
                        # añadir el punto 'a1' si su caudal real es menor que la recta de interpolación desagregada
                        if (data[a1] <= Qslow[a1]):
                            valles_add.append(a1)
                            continue
                        # añadir mínimos dentro del intervalo
                        if (data[a1] < data[a0]) and (data[a1] < data[a2]):
                            valles_add.append(a1)

            # unir los nuevos mínimos a los mínimos reducidos
            valles3 = valles2 + valles_add
            valles3.sort()
    
            return valles3
    
    if tipo == 4:
        valles4 = valles1.copy()
        i = 1
        while i < len(valles4) - 1:
            v0, v1, v2 = valles4[i-1], valles4[i], valles4[i+1]
            if data[v1] > m * np.mean((data[v0], data[v2])):
                valles4.remove(v1)
            else:
                i +=1

        return valles4


# In[5]:


def key_points(data, serie='Q', k=1e6, dQ_threshold=0.25, d2Q_threshold=0.1, window=8, plot=True):
    """Identifica los valles (inicio de un evento de escorrentía superficial) y los picos (caudal máximo de dicho evento) en una serie de caudal. Se considera valle al primer intervalo temporal con una pendiente del hidrograma superior a 'dQ_threshold' o una curvatura superior a 'd2Q_threshold'. Se considera un intervalo como pico si la pendiente en los 'window' intervalos anteriores es positiva o nula y en los 'window' intervalos posteriores es negativa.
    
    Parámetros:
    -----------
    data:          data frame. Las filas representan pasos temporales y habrá al menos cuatro campos: 'serie' contiene los datos de caudal (m³/s), 'dQ' contiene la pendiente del hidrograma(m³/s²), 'm' contiene el signo de la pendiente del hidrograma (-), y 'd2Q' con la curvatura del hidrograma (m³/s³).
    serie:         string. Nombre del campo que contiene la serie de caudal
    dQ_threshold:  float. Valor mínimo de la pendiente del hidrograma a partir del cual se considera que empieza el evento de escorrentía directa.
    d2Q_threshold: float. Valor mínimo de la curvatura del hidrograma a partirl del cual se ocnsidera que empieza el evento de escorrentía directa.
    window:        integer. Número de pasos temporales que se tendrán en cuenta para encontrar los valles y los picos
    plot:          
    
    Salidas:
    --------
    valleys:       lista. Fechas en las que aparecen los valles.
    peaks:         lista. Fechas en las que ocurren los picos"""
    
    # Listas donde se guardan las fechas de los puntos clave
    valleys, peaks, recess, inflex = [], [], [], []
    
    # intervalo temporal en segundos
    At = (data.index[1] - data.index[0]).total_seconds() 
    
    # Identificar valles y picos, es decir, eventos de escorrentía directa
    # --------------------------------------------------------------------
    # Condición de los valles
    mask = (data.dQ > dQ_threshold) | (data.d2Q > d2Q_threshold)
    i = 0
    while i <= data.shape[0] - window:
        # Inicio del hidrograma de escorrentía directa
        tv = data.iloc[i:].loc[mask].first_valid_index()
        i = data.index.get_loc(tv)
        if (data.m[i:i+window] >= 0).all(): # al menos 'window' intervalos tienen pendiente creciente
            valleys.append(tv)
        
            # Pico del hidrograma de escorrentía directa
            for i in range(i, data.shape[0]):
                if (data.m[i+1-window:i+1] >= 0).all() & (data.m[i+1:i+1+window] == -1).all():
                    tp = data.index[i]
                    Qp = data.Q[tp]
                    if Qp < data.loc[tv:tp+window, 'Q'].max():
                        tp = data.loc[tv:tp+window, 'Q'].idxmax()
                    peaks.append(tp)
                    #print('tp =', tp)
                    break
        i += 1
    
    # Identificar puntos de recesión e inflexión
    # ------------------------------------------
    for i, (tv, tp) in enumerate(zip(valleys, peaks)):
        # 'tr' es el punto final del flujo rápido. Es el primer intervalo a partir del cual la pendiente de la curva 
        # ln(Q()/t es menor que -1/k en al menos 'w' intervalos posteriores
        w = 8
        # Punto final sobre el que buscar el receso
        if i < len(valleys) - 1:
            tf = valleys[i+1]
        else:
            tf = data.index[-w]
        # Encontrar el 'tr' preliminar
        for tr in data[tp:tf].index:
            if (log(data.Q[tr+w]) - log(data.Q[tr])) / (w * At) <= -k**-1:
                break
        # Calcular la curva de recesión retroactivamente desde 'tr'
        Qr = data.loc[tr, 'Q']
        data['Qrec'] = np.nan
        for t in data[tp:tr].index:
            data.loc[t, 'Qrec'] = Qt(Qr, tr, t, k)
        # establecer un nuevo 'tr' si se cruzan el caudal y la curva de recesión
        tr_aux = data.loc[data.Qrec > data.Q, :].first_valid_index()
        if tr_aux != None:
            tr = tr_aux
        del tr_aux
        # Guardar 'tr' en la lista
        recess.append(tr)

        # 'ti' es el punto de inflexión de la curva descendente. 
        # se hace una media móvil sobre la segunda derivada del caudal para suavizar la curva
        data['d2Qma'] = np.nan
        for t in data[tp:tr].index:
            data.loc[t, 'd2Qma'] = data.d2Q[t-3:t+3].mean()
        # el punto de inflexión es el del primer valor positivo o nulo
        ti = data[data['d2Qma'] >= 0].first_valid_index() # ARREGLAR ES EL PASO ANTERIOR!!!!
        if ti == None:
            ti = tp + (tr - tp) / 2
        # Guardar 'ti' en la lista
        inflex.append(ti)
    
    # Visualizar
    if plot == True:
        print('Nº de valles:', len(valleys), '\tNº de picos:', len(peaks), '\t\tNº de inflexiones:', len(inflex),
              '\t\tNº de recesiones:', len(recess))
        plt.figure(figsize=(18,3))
        plt.fill_between(data.index, data.Q, alpha=0.3, label='hidrograma')
        plt.scatter(valleys, data.loc[valleys, 'Q'], color='green', s=10, label='valles')
        plt.scatter(peaks, data.loc[peaks, 'Q'], color='maroon', s=10, label='picos')
        plt.scatter(recess, data.loc[recess, 'Q'], marker='x', color='black', s=10, label='inflexión')
        plt.scatter(inflex, data.loc[inflex, 'Q'], marker='X', color='grey', s=10, label='recesión')
        plt.ylabel('Q (m³/s)', fontsize=13)
        ymax = ceil(data.Q.max() / 10) * 10
        plt.xlim((data.index[0], data.index[-1]))
        plt.ylim((0,ymax))
        plt.legend(fontsize=12);
    
    # Crear el diccionario de salida
    key_points = {'valley': valleys, 'peak': peaks, 'recession': recess, 'inflexion': inflex}
    
    return key_points


# In[7]:


def key_points2(data, serie='Q', k=1e6, dQ_threshold=0.1, d2Q_threshold=1, multiple=None, fraction=4, window=8, 
                plot=True):
    """Identifica los valles (inicio de un evento de escorrentía superficial) y los picos (caudal máximo de dicho evento) en una serie de caudal. Se considera valle al primer intervalo temporal con una pendiente del hidrograma superior a 'dQ_threshold' o una curvatura superior a 'd2Q_threshold'. Se considera un intervalo como pico si la pendiente en los 'window' intervalos anteriores es positiva o nula y en los 'window' intervalos posteriores es negativa.
    
    Parámetros:
    -----------
    data:          data frame. Las filas representan pasos temporales y habrá al menos cuatro campos: 'serie' contiene
                   los datos de caudal (m³/s), 'dQ' contiene la pendiente del hidrograma(m³/s²), 'm' contiene el signo
                   de la pendiente del hidrograma (-), y 'd2Q' con la curvatura del hidrograma (m³/s³).
    serie:         string. Nombre del campo que contiene la serie de caudal
    k:             float. Constante de recesión de caudal (s)
    dQ_threshold:  float. Valor mínimo de la pendiente del hidrograma a partir del cual se considera que empieza el evento de
                   escorrentía directa.
    d2Q_threshold: float. Valor mínimo de la curvatura del hidrograma a partirl del cual se ocnsidera que empieza el evento 
                   de escorrentía directa.
    multiple:      float. Tanto por uno aplicado sobre el caudal en un valle para que otro punto pueda ser considerado valle
    fraction:      integer. Fracción utilizada para aceptar un punto como inicio de la recesión. Un punto no se aceptará como
                   tal si su caudal es superior a 'fraction' entre el pico y el valle.
    window:        integer. Número de pasos temporales que se tendrán en cuenta para encontrar los valles y los picos
    plot:          boolean. Si se quiere mostrar el hidrograma con los puntos
    
    Salidas:
    --------
    key_points:    dictionary. Contiene cuatro listas con los puntos clave del hidrograma de escorrentía directa: 
                   'valley' contiene la lista de los puntos de inicio de cada evento
                   'peak' contiene la lista con los picos de cada evento
                   'recession' contiene la lista con los puntos finales de cada evento
                   'inflexion0 contiene la lista con los puntos de inflexión de la rama descendente de cada evento'"""
    
    # Listas donde se guardan las fechas de los puntos clave
    valleys, peaks, recess, inflex = [], [], [], []
    # Campos que hace falta crear
    data['Qrec'], data['d2Qma'] = np.nan, np.nan
    # se hace una media móvil sobre la segunda derivada del caudal para suavizar la curva
    data.d2Qma = data.d2Q.rolling(window=window).mean()
    
    # intervalo temporal en segundos
    At = (data.index[1] - data.index[0]).total_seconds()
    delta = datetime.timedelta(seconds=At)
    if At == 300:
        freq = '5min'
    elif At == 3600:
        freq = 'H'
    elif At == 86400:
        freq = 'D'
    
    # Identificar los inicios del hidrograma de escorrentía directa
    # -------------------------------------------------------------
    # Puntos que cumplen las condiciones de primera y segunda derivada
    mask = ((data.dQ > dQ_threshold) | (data.d2Q > d2Q_threshold))
    date = data.index[0]
    aux1 = []
    while date <= data.index[-1] - window:
        # Inicio del hidrograma de escorrentía directa
        if ((data.dQ[date] > dQ_threshold) | (data.d2Q[date] > d2Q_threshold)):
            aux1.append(date)
            date += window
        else:
            date += 1
    valleys = aux1
    
    # De los puntos anteriores, cuáles tienen pendiente positiva en 'window' intervalos posteriores
    aux2 = []
    for valley in valleys:
        dates = pd.date_range(valley, periods=window, freq=freq)
        if (data.m[dates] >= 0).all():
            aux2.append(valley)
    aux2.append(data.index[-1])
    valleys = aux2
    
    # De los anteriores, eliminar aquellos que no tengan una pendiente menor que un umbral durante 
    aux3 = []
    for valley in valleys:
        dates = pd.date_range(valley - window, periods=window, freq=freq)
        if (abs(data.dQ[dates]) < dQ_threshold).all():
            aux3.append(valley)
    valleys = aux3
    
    # De los puntos anteriores, eliminar aquellos que no tengan un pico entre él y el anterior
    for v1, v2 in zip(valleys[:-2], valleys[1:]):
        if data[serie][v1:v2].max() <= max(data[serie][v1], data[serie][v2]):
            valleys.remove(v2)
    #valleys = aux2
    
    # De los anteriores, eliminar aquellos cuyo caudal sea superior en un 15% al del punto anterio
    if multiple != None:
        aux4 = [valleys[0]]
        for i, v1 in enumerate(valleys[:-1]):
            if v1 < aux4[-1]: # saltar si v1 ya fue eliminado
                continue
            for v2 in valleys[i+1:]:
                if data[serie][v2] < data[serie][v1] * multiple:
                    aux4.append(v2)
                    break
        valleys = aux4
    
    # Identificar los picos del hidrograma de escorrentía directa
    # -----------------------------------------------------------
    aux = valleys + [data.index[-1]]
    for i, (v1, v2) in enumerate(zip(aux[:-2], aux[1:])):
        peaks.append(data[serie][v1:v2].idxmax())
        
    # Eliminar última entrada de 'valleys' por ser simplemente el punto final de la serie
    valleys = valleys[:-1]
        
    # Identificar puntos de recesión e inflexión
    # ------------------------------------------
    for i, tp in enumerate(peaks):
        # Buscar el último pico del hidrograma de escorrentía superficial
        # encontrar el índice en fecha de todos los picos
        #if i + 1 < len(valleys):
            #tv1, tv2 = valleys[i], valleys[i+1]
        #else:
            #tv1, tv2 = valleys[i], data.index[-1]
        #idx = np.argwhere(np.diff(np.sign(data.dQ[tv1:tv2]), n=1) != 0).reshape(-1)
        #idx = data[tv1:tv2].index[idx]
        # límite de caudal para encontrar picos
        #Qlim = np.percentile(data.loc[idx, 'Q'], 75) # percentil 75 de los picos
        #Qlim = data.loc[idx, 'Q'].mean() # media de los picos
        #tp = data.loc[idx,:].loc[data.Q > Qlim, :].index[-1]
        
        # Punto final sobre el que buscar el receso
        if i + 1 < len(valleys):
            tf = valleys[i+1]
        elif i + 1 == len(valleys):
            tf = data.index[-1]
        # Encontrar el 'tr' preliminar
        tr = tp
        # 'tr' es el punto final del flujo rápido. Es el primer intervalo a partir del cual la pendiente de la 
        # curva ln(Q)/t es menor que -1/k en al menos 'w' intervalos posteriores
        #for tr in data[tp:tf].index:
            #if (log(data.Q[tr+window]) - log(data.Q[tr])) / (window * At) <= -k**-1:
                #break
        while data[serie][tr] > (data[serie][tf] + (data[serie][tp] - data[serie][tf]) / fraction):
            # Calcular la curva de recesión retroactivamente desde 'tf'
            Qf = data[serie][tf]
            for t in data[tp:tf].index:
                data.loc[t, 'Qrec'] = Qt(Qf, tf, t, k)
            # establecer 'tr' en la intersección del hidrograma y la curva de recesión
            try:
                # Puntos de intersección entre el hidrograma y la curva de recesión
                idx = np.argwhere(np.diff(np.sign(data[serie][tp:tf] - data.Qrec[tp:tf])) != 0).reshape(-1)
                # Se toma 'tr' como el penúltimo punto de intersección
                tr = data[tp:tf].index[idx[-2]]
                #tr = data.loc[tp:tf,:].loc[data.Q <= data.Qrec].index[-2]#.first_valid_index()
                tf = tr + ((tf - tr) / 2).round(freq)
            except:
                tf = tp + ((tf - tp) / 2).round(freq)
            
        # Guardar 'tr' en la lista
        recess.append(tr)
        
        # 'ti' es el punto de inflexión de la curva descendente.
        # el punto de inflexión es el del primer valor positivo o nulo de 'd2Qma'
        try:
            # Encontrar puntos de inflexión del hidrograma que pasan de curvatura - a + entre 'tp' y 'tr'
            idx = np.argwhere(np.diff(np.sign(data.d2Qma[tp:tr]), n=1) > 0).reshape(-1)
            # Se toma 'ti' como el último de dichos puntos
            ti = data[tp:tr].index[idx[-1]]
            #ti = data.loc[tp:tr,:].loc[data['d2Qma'] >= 0].index[0] # ARREGLAR ES EL PASO ANTERIOR!!!!
        except:
            ti = tp + (tr - tp) / 2
        # Guardar 'ti' en la lista
        inflex.append(ti)
    
    # Visualizar
    # ----------
    if plot == True:
        print('Nº de valles:', len(valleys), '\tNº de picos:', len(peaks), '\t\tNº de inflexiones:', len(inflex),
              '\t\tNº de recesiones:', len(recess))
        plt.figure(figsize=(18,3))
        plt.fill_between(data.index, data[serie], alpha=0.3, label=serie)
        plt.plot(data.Qrec, '--k', linewidth=1)
        plt.scatter(valleys, data.loc[valleys, serie], color='green', s=10, label='valles')
        plt.scatter(peaks, data.loc[peaks, serie], color='maroon', s=10, label='picos')
        plt.scatter(recess, data.loc[recess, serie], marker='x', color='black', s=10, label='recesión')
        plt.scatter(inflex, data.loc[inflex, serie], marker='X', color='grey', s=10, label='inflexión')
        plt.ylabel('Q (m³/s)', fontsize=13)
        ymax = ceil(data[serie].max() / 10) * 10
        plt.xlim((data.index[0], data.index[-1]))
        plt.ylim((0,ymax))
        plt.legend(fontsize=12);
    
    # Crear el diccionario de salida
    key_points = {'valley': valleys, 'peak': peaks, 'recession': recess, 'inflexion': inflex}
    
    return key_points


# In[ ]:


def key_points3(data, k, dQ_threshold=0.1, d2Q_threshold=1, window=8, plot=True):
    """Identifica los valles (inicio de un evento de escorrentía superficial) y los picos (caudal máximo de dicho evento)
    en una serie de caudal. Se considera valle al primer intervalo temporal con una pendiente del hidrograma superior a
    'dQ_threshold' o una curvatura superior a 'd2Q_threshold'. Se considera un intervalo como pico si la pendiente en los
    'window' intervalos anteriores es positiva o nula y en los 'window' intervalos posteriores es negativa.
    
    Parámetros:
    -----------
    data:          data frame. Las filas representan pasos temporales y habrá al menos cuatro campos: 'Q' con los datos de 
                   caudal (m³/s), 'dQ' con la pendiente del hidrograma(m³/s²), 'm' con el signo de la pendiente del hidrograma
                   (-), y 'd2Q' con la curvatura del hidrograma (m³/s³)
    dQ_threshold:  float. Valor mínimo de la pendiente del hidrograma a partir del cual se considera que empieza el evento de
                   escorrentía directa.
    d2Q_threshold: float. Valor mínimo de la curvatura del hidrograma a partirl del cual se ocnsidera que empieza el evento 
                   de escorrentía directa.
    window:        integer. Número de pasos temporales que se tendrán en cuenta para encontrar los valles y los picos
    
    Salidas:
    --------
    valleys:       lista. Fechas en las que aparecen los valles.
    peaks:         lista. Fechas en las que ocurren los picos"""
    
    # Listas donde se guardan las fechas de los puntos clave
    valleys, peaks, recess, inflex = [], [], [], []
    # Campos que hace falta crear
    data['Qrec'], data['d2Qma'] = np.nan, np.nan
    
    # intervalo temporal en segundos
    At = (data.index[1] - data.index[0]).total_seconds()
    delta = datetime.timedelta(seconds=At)
    if At == 300:
        freq = '5min'
    elif At == 3600:
        freq = 'H'
    elif At == 86400:
        freq = 'D'
    
    # Identificar los inicios del hidrograma de escorrentía directa
    # -------------------------------------------------------------
    aux = []
    for date in data.index[window:-window]:
        #print(date-window, date-1, date, date+1, date+window)
        if (data.Q[date] <= data.Q[date-window:date-1]).all() & (data.Q[date] <= data.Q[date+1:date+window]).all():
            #print(data.Q[date-window:date-1].min(), data.Q[date], data.Q[date+1:date+window].min())
            aux.append(date)
            
    # De los puntos anteriores, eliminar aquellos que no tengan un pico entre él y el anterior
    for v1, v2 in zip(aux[:-1], aux[1:]):
        if data.Q[v1:v2].max() <= max(data.Q[v1], data.Q[v2]):
            aux.remove(v1)
    valleys = aux
    
    # Visualizar
    # ----------
    if plot == True:
        print('Nº de valles:', len(valleys), '\tNº de picos:', len(peaks), '\t\tNº de inflexiones:', len(inflex),
              '\t\tNº de recesiones:', len(recess))
        plt.figure(figsize=(18,3))
        plt.fill_between(data.index, data.Q, alpha=0.3, label='hidrograma')
        #plt.plot(data.Qrec, '--k', linewidth=1)
        plt.scatter(valleys, data.loc[valleys, 'Q'], color='green', s=10, label='valles')
        #plt.scatter(peaks, data.loc[peaks, 'Q'], color='maroon', s=10, label='picos')
        #plt.scatter(recess, data.loc[recess, 'Q'], marker='x', color='black', s=10, label='recesión')
        #plt.scatter(inflex, data.loc[inflex, 'Q'], marker='X', color='grey', s=10, label='inflexión')
        plt.ylabel('Q (m³/s)', fontsize=13)
        ymax = ceil(data.Q.max() / 10) * 10
        plt.xlim((data.index[0], data.index[-1]))
        plt.ylim((0,ymax))
        plt.legend(fontsize=12);
    
    # Crear el diccionario de salida
    key_points = {'valley': valleys, 'peak': peaks, 'recession': recess, 'inflexion': inflex}
    
    return key_points


# ### 3.2. Métodos de desagregación

# In[1]:


def mml_old(data, A, plot=True, plot_dot=False, xlim=None, title=None, label=None):
    """Desagrega la serie de caudal aforado en río en una serie de caudal rápido (o escorrentía superficial) y una serie de
    caudal lento (o caudal base). Para ello utiliza el método de los mínimos locales. 
    Este método identifica primeramente los mínimos locales como los puntos con caudal mínimo en una ventana centrada de
    ancho 2N+1. El caudal base es la interpolación lineal entre los mínimos locales. La escorrentía superficial es la dife-
    rencia entre el caudal aforado y el caudal base.
    
    Parámetros:
    -----------
    data:      series. Serie de caudal
    A:         float. Área de la cuenca hidrográfica de la estación de aforo (km²)
    plot:      boolean. Si se quiere mostrar el hidrograma desagregado
    plot_dot:  boolena. Si se quieren mostrar en el hidrograma los mínimos locales
    xlim:      list. Fechas de inicio y fin de la gráfica del hidrograma
    title:     string. Título del gráfico. None por defecto.
    
    Salidas:
    --------
    data:      data frame. El 'data frame' de entrada con dos nuevas columnas: 'Qslow' y 'Qquick'
    Si plot == True, se mostrará el hidrograma desagregado"""
    
    Qslow = pd.Series(index=data.index)
    Qquick = pd.Series(index=data.index)
    
    # Calcular el número de días a observar antes y después
    N = int(round(0.8* A**0.2))
    
    # 'Data frame' con los intervalos que representan un mínimo local
    # Iniciar la serie de mínimos locales con el primer intervalo (sea o no mínimo local)
    lows = [data.index[0]]
    for t in data.index[N:-N]:
        Qt = data[t]
        if Qt == data[t-N:t+N].min():
            lows.append(t)
    # Añadir el último punto de la serie
    lows.append(data.index[-1])

    # Calcular la serie de caudal base
    for i in range(len(lows) - 1):
        v1, v2 = lows[i], lows[i+1]
        Q1, Q2 = data[v1], data[v2]
        for t in data[v1:v2].index:
            Qslow[t] = min(Q1 + (Q2 - Q1) * (t - v1) / (v2 - v1), data[t])
    
    # Calcular la serie de caudal lento
    Qquick = data - Qslow
    
    if plot == True:
        # Visualizar
        plt.figure(figsize=(18,3))
        plt.plot(data, '-k', linewidth=0.5, label=label)
        if plot_dot == True:
            plt.scatter(lows.index, lows[serie], color='black', s=8, label='mín. local')
        plt.fill_between(data.index, Qslow, alpha=0.3, label='slow flow')
        plt.fill_between(data.index, Qquick + Qslow, Qslow, alpha=0.3, label='quick flow')
        if xlim == None:
            plt.xlim((data.index[0], data.index[-1]))
            ymax = ceil(data.max() / 10) * 10
        else:
            plt.xlim(xlim)
            ymax = ceil(data.loc[xlim[0]:xlim[1], serie].max() / 10) * 10
        plt.ylim((0,ymax))
        plt.ylabel('Q (m³/s)', fontsize=13)
        plt.title(title, fontsize=14)
        plt.title('Método de los mínimos locales', fontsize=13)
        plt.legend(fontsize=13);
        
    return lows, Qslow, Qquick


# In[4]:


def mml(data, valles, factor=1, plot=False, plot_dot=False, xlim=None, title=None, label=None):
    """Desagrega la serie de caudal total en una serie de caudal rápido (escorrentía superficial + interflujo) y una serie de caudal lento (ocaudal base).
    Se utiliza el método de los mínimos locales. Este método identifica primeramente los mínimos locales como los puntos con caudal mínimo en una ventana centrada de ancho 2N+1. El caudal base es la interpolación lineal entre los mínimos locales. La escorrentía superficial es la diferencia entre el caudal aforado y el caudal base.
    
    Parámetros:
    -----------
    data:      series. Serie de caudal total
    valles:    list. Fechas en las que acontece un mínimo local en la serie
    plot:      boolean. Si se quiere mostrar el hidrograma desagregado
    plot_dot:  boolena. Si se quieren mostrar en el hidrograma los mínimos locales
    xlim:      list. Fechas de inicio y fin de la gráfica del hidrograma
    title:     string. Título del gráfico. None por defecto.
    
    Salidas:
    --------
    Qslow:     series. Serie de caudal lento
    Qquick:    series. Serie de caudal rápido
    Si plot == True, se mostrará el hidrograma desagregado"""
    
    Qslow = pd.Series(index=data.index)
    Qquick = pd.Series(index=data.index)

    # Calcular la serie de caudal base
    for i in range(len(valles) - 1):
        v1, v2 = valles[i], valles[i+1]
        Q1, Q2 = data[v1] * factor, data[v2] * factor
        for t in data[v1:v2].index:
            Qslow[t] = min(Q1 + (Q2 - Q1) * (t - v1) / (v2 - v1), data[t])
    
    # Calcular la serie de caudal lento
    Qquick = data - Qslow
    
    if plot == True:
        # Visualizar
        plt.figure(figsize=(18,3))
        plt.plot(data, '-k', linewidth=0.5, label=label)
        if plot_dot == True:
            plt.scatter(valles, data[valles], color='black', s=8,
                        label='mín. local')
        plt.fill_between(data.index, Qslow, color='steelblue', alpha=0.25,
                         label='slow flow')
        plt.fill_between(data.index, Qquick + Qslow, Qslow, color='steelblue',
                         alpha=0.5, label='quick flow')
        if xlim == None:
            plt.xlim((data.index[0], data.index[-1]))
            ymax = ceil(data.max() / 10) * 10
        else:
            plt.xlim(xlim)
            ymax = ceil(data.loc[xlim[0]:xlim[1]].max() / 10) * 10
        plt.ylim((0,ymax))
        plt.ylabel('Q (m³/s)', fontsize=13)
        plt.title(title, fontsize=14)
        plt.title('Método de los mínimos locales', fontsize=13)
        plt.legend(fontsize=13);
        
    return Qslow, Qquick


# In[12]:


def mlr(data, valley, A, recession=None, plot=True, xlim=None):
    """Se desagrega el hidrograma de un evento de inundación en caudal rápido y lento por medio del método de la línea
    recta.
    
    Parámetros:
    -----------
    data:      series. Serie temporal con el caudal aforado
    valley:    list of datetime. Valor del índice de 'data' para los puntos de inicio del hidrograma de escorrentía 
               superficial. 
    A:         integer. Área de la cuenca vertiente al punto (km²)
    recession: list of datetime. Valor del índice de 'data' para los puntos finales del hidrograma de escorrentía 
               superficial. 
    plot:      boolean. Si se quiere mostrar el hidrograma desagregado
    xlim:      list. Dos valores de las fechas inicial y final del hidrograma a mostrar. Sólo si plot==True
    
    Salidas:
    --------
    Qslow:     series. Serie temporal de caudal lento
    Qquick:    series. Serie temporal de caudal rápido"""
    
    # Crear las series de caudal lento y rápido
    Qslow = pd.Series(index=data.index)
    Qquick = pd.Series(index=data.index)
    
    for i, tv in enumerate(valley): # 'tv' es el paso en el que empieza el flujo rápido
        if recession != None: # si se suministra el fin del hidrograma de escorrentía directa
            tr = recession[i]
            Qv, Qr = data[tv], data[tr]
            
            # Calcular la serie de caudal lento
            # Entre el valle y la recesión
            for t in data[tv:tr].index[1:]:
                Qt = Qv + (t - tv) / (tr - tv) * (Qr - Qv)
                Qslow[t] = min(Qt, data[t]) 
            # Entre la recesión y el siguiente valle
            if i + 1 < len(valley):
                tv2 = valley[i+1]
            else:
                tv2 = data.index[-1]
            aux = mml(data[tr:tv2].copy(), A, serie=serie, plot=False)
            Qslow[tr:tv2] = aux.Qslow[tr:tv2]           
        
        else:
            # 'tv2' es el siguiente valle o el fin de la serie
            if i < len(valley) - 1:
                tv2 = valley[i+1]
            else:
                tv2 = data.index[-1]

            # paso en el que vuelve la recesión del caudal base
            mask = (data.index > tv) & (data <= data[tv])
            tr = data[mask].first_valid_index()
            if (tr == None):# | (tr > tv2):
                tr = tv2

            # calcular la serie de caudal lento
            Qslow[tv:tr] = data[tv]
    
    # calcaular la serie de caudal rápido
    Qquick = data - Qslow
    Qquick[Qquick < 0] = np.nan
    
    if plot == True:
        # Visualizar
        plt.figure(figsize=(18,3))
        plt.plot(data, '-k', linewidth=0.5, label='total flow')
        plt.fill_between(data.index, Qslow, alpha=0.3, label='slow flow')
        plt.fill_between(data.index, Qquick + Qslow, Qslow, alpha=0.3, label='quick flow')
        if xlim == None:
            plt.xlim((data.index[0], data.index[-1]))
        else:
            plt.xlim(xlim)
        ymax = ceil(data.max() / 10) * 10
        plt.ylim((0,ymax))
        plt.ylabel('Q (m³/s)', fontsize=13)
        plt.title('Método de la línea recta', fontsize=13)
        plt.legend(fontsize=13);
        
    return Qslow, Qquick


# In[16]:


def mbf_old(data, valley, peak, k, A, base=None, recession=None, plot=True, xlim=None):
    """Se desagrega el hidrograma de un evento de inundación en caudal rápido y lento por medio del método de la base fija.
    
    Parámetros:
    -----------
    data:      series. Serie temporal de caudal aforado
    valley:    integer. Fila de 'data' en la que se inicia el hidrograma de escorrentía superficial
    peak:      integer. Fila de 'data' en la que ocurre el pico del hidrograma
    k:         float. Constante de deacimiento de la curva de recesión (s)
    A:         integer. Área de la cuenca vertiente al punto (km²)
    serie:     string. Nombre del campo que contiene la serie a desagregar
    base:      integer. Número de pasos temporales entre el pico la desaparición del flujo rápido
    recession: list of datetime. Valor del índice de 'data' para los puntos finales del hidrograma de escorrentía 
               superficial. 
    plot:      boolean. Si se quiere mostrar el hidrograma desagregado
    xlim:      list. Dos valores de las fechas inicial y final del hidrograma a mostrar. Sólo si plot==True
    
    Salidas:
    --------
    Qslow:     series. Serie temporal de caudal lento
    Qquick:    series. Serie temporal de caudal rápido"""
    
    # Crear las series de caudal lento y rápido
    Qslow = data
    Qquick = np.nan
    
    for i, (tv, tp) in enumerate(zip(valley, peak)):
        # Puntos clave del hidrograma
        # 'tv' es el paso en el que empieza el flujo rápido
        # 'tp' es el paso en el que ocurre el pico de caudal
        # 'tr' es el paso en el que desaparece el flujo rápido
        if recession == None:
            tr = tp + base
        else:
            tr = recession[i]
        print(tv, tp, tr)

        # CALCULAR LA SERIE DE FLUJO BASE
        # entre el inicio y el pico
        for t in data[tv:tp].index: 
            Q = Qt(data[tv], tv, t, k)
            Qslow[t] = min(Q, data[t])
        # entre el pico y el fin
        #for t in data[tp:tr].index[1:]: 
            #Q = Qslow[tp] + (data[tr] - Qslow[tp]) * (t - tp) / (tr - tp)
            #Qslow[t] = min(Q, data[t])
        # Entre el fin y el siguiente inicio
        #if i + 1 < len(valley):
            #tv2 = valley[i+1]
        #else:
            #tv2 = data.index[-1]
        #if tr > tv2:
            #tr = tv2
        #l, Qslow[tr:tv2], quick = mml(data[tr:tv2], A, plot=False)
        ##Qslow[tr:tv2] = slow[tr:tv2] 

    # calcular la serie de flujo rápido
    Qquick = data - Qslow
    Qquick[Qquick < 0] = np.nan
    
    if plot == True:
        # Visualizar
        plt.figure(figsize=(18,3))
        plt.plot(data, '-k', linewidth=0.5, label='total flow')
        plt.fill_between(data.index, Qslow, alpha=0.3, label='slow flow')
        plt.fill_between(data.index, Qquick + Qslow, Qslow, alpha=0.3, label='quick flow')
        if xlim == None:
            plt.xlim((data.index[0], data.index[-1]))
        else:
            plt.xlim(xlim)
        ymax = ceil(data.max() / 10) * 10
        plt.ylim((0,ymax))
        plt.ylabel('Q (m³/s)', fontsize=13)
        plt.title('Método de la base fija', fontsize=13)
        plt.legend(fontsize=13);
        
    return Qslow, Qquick


# In[3]:


def mbf(data, valles, k, factor=1, plot=False, plot_dot=False, xlim=None, title=None):
    """"""
    
    # Crear las series de caudal lento y rápido
    Qslow = pd.Series(data=None, index=data.index)
    Qquick = pd.Series(data=None, index=data.index)
    
    # Calcular picos
    picos = []
    for l1, l2 in zip(valles[:-1], valles[1:]):
        picos.append(data[l1:l2].argmax())
    
    # CALCULAR LA SERIE DE FLUJO BASE
    for i, (tv, tp) in enumerate(zip(valles, picos)):
        # Puntos clave del hidrograma
        # 'tv': paso en el que empieza el flujo rápido
        # 'tp': paso en el que ocurre el pico de caudal
        # 'tr': paso en el que desaparece el flujo rápido
        tr = valles[i+1]
        
        # Qslow entre el pico y el fin
        for t in pd.date_range(tp, tr): 
            Q = Qt(data[tr] * factor, tr, t, k)
            Qslow[t] = min(Q, data[t])
        # Qslow entre el inicio y el pico
        for t in pd.date_range(tv, tp):
            Q = data[tv] * factor + (t - tv) * (Qslow[tp] - data[tv] * factor) / (tp - tv)
            Qslow[t] = min(Q, data[t])

    # CALCULAR LA SERIE DE FLUJO RÁPIDO
    Qquick = data - Qslow
    Qquick[Qquick < 0] = np.nan
    
    if plot == True:
        # Visualizar
        plt.figure(figsize=(18,3))
        plt.plot(data, '-k', linewidth=0.5, label='aforado')
        if plot_dot == True:
            plt.scatter(valles.index, valles[serie], color='black', s=8, label='mín. local')
        plt.fill_between(data.index, Qslow, color='steelblue', alpha=0.25, label='lento')
        plt.fill_between(data.index, Qquick + Qslow, Qslow, color='steelblue', alpha=0.5, label='rápido')
        if xlim == None:
            plt.xlim((data.index[0], data.index[-1]))
            ymax = ceil(data.max() / 10) * 10
        else:
            plt.xlim(xlim)
            ymax = ceil(data.loc[xlim[0]:xlim[1], serie].max() / 10) * 10
        plt.ylim((0,ymax))
        plt.ylabel('Q (m³/s)', fontsize=13)
        plt.title(title, fontsize=13)
        plt.title('Método de los mínimos locales', fontsize=13)
        plt.legend(fontsize=13);
        
    return Qslow, Qquick


# In[14]:


def mpv(data, valley, peak, inflexion, recession, k, A, plot=True, xlim=None, w=8):
    """Se desagrega el hidrograma de un evento de inundación en caudal rápido y lento por medio del método de la pendiente
    variable.
    
    Parámetros:
    -----------
    data:      series. Serie temporal de caudal aforado
    valley:    integer. Fila de 'data' en la que se inicia el hidrograma de escorrentía superficial
    peak:      integer. Fila de 'data' en la que ocurre el pico del hidrograma
    inflexion: list of datetime. Valor del índice de 'data' para los puntos de inflexión del hidrograma de 
               escorrentía superficial. 
    recession: list of datetime. Valor del índice de 'data' para los puntos finales del hidrograma de escorrentía 
               superficial. 
    k:         float. Constante de deacimiento de la curva de recesión (s)
    A:         integer. Área de la cuenca vertiente al punto (km²)
    serie:     string. Nombre del campo que contiene la serie a desagregar
    plot:      boolean. Si se quiere mostrar el hidrograma desagregado
    xlim:      list. Dos valores de las fechas inicial y final del hidrograma a mostrar. Sólo si plot==True
    w:         integer. Nº de intervalos en los que se debe cumplir la regla de la curva de recesión.
    
    Salidas:
    --------
    Qslow:     series. Serie temporal de caudal lento
    Qquick:    series. Serie temporal de caudal rápido"""
    
    # Crear las series de caudal lento y rápido
    Qslow = data
    Qquick = np.nan
    
    At = (data.index[1] - data.index[0]).total_seconds() # intervalo temporal en segundos
    
    # Generar la serie de flujo base
    # ------------------------------
    for i, (tv, tp, ti, tr) in enumerate(zip(valley, peak, inflexion, recession)):
        # Caudal en los puntos clave
        Qv = data[tv]
        Qp = Qt(Qv, tv, tp, k)
        Qr = data[tr]
        Qi = Qt(Qr, tr, ti, k)
        # Serie de caudal base
        for t in data[tv:tp].index: # entre el inicio y el pico
            Q = Qt(Qv, tv, t, k)
            Qslow[t] = min(Q, data[t])
        for t in data[tp:ti].index[1:]: # entre el pico y la inflexión
            Q = Qp + (Qi - Qp) * (t - tp) / (ti - tp)
            Qslow[t] = min(Q, data[t])
        for t in data[ti:tr].index: # entre la inflexión y el fin
            Q = Qt(Qi, ti, t, k)
            Qslow[t] = min(Q, data[t])
        # Entre el fin y el siguiente inicio
        if i + 1 < len(valley):
            tv2 = valley[i+1]
        else:
            tv2 = data.index[-1]
        aux = mml(data[tr:tv2].copy(), A, serie=serie, plot=False)
        Qslow[tr:tv2] = aux.Qslow[tr:tv2] 
        
    # Generar la serie de flujo rápido
    # --------------------------------
    Qquick = data - Qslow
    Qquick[Qquick < 0] = np.nan
    
    if plot == True:
        # Visualizar
        plt.figure(figsize=(18,3))
        plt.plot(data, '-k', linewidth=0.5, label=serie)
        plt.fill_between(data.index, Qslow, alpha=0.3, label='slow flow')
        plt.fill_between(data.index, Qquick + Qslow, Qslow, alpha=0.3, label='quick flow')
        if xlim == None:
            plt.xlim((data.index[0], data.index[-1]))
        else:
            plt.xlim(xlim)
        ymax = ceil(data.max() / 10) * 10
        plt.ylim((0,ymax))
        plt.ylabel('Q (m³/s)', fontsize=13)
        plt.title('Método de la pendiente variable', fontsize=13)
        plt.legend(fontsize=13);

    return Qquick, Qslow

