#!/usr/bin/env python
# coding: utf-8

# # Funciones de rendimiento espacial
# _Autor:_    __Jesús Casado__ <br> _Revisión:_ __05/12/2020__
# 
# __Índice__ <br>
# [KGE](#KGE)<br>
# [SPAEF](#SPAEF)<br>
# [Clasificación binario](#Clasificación-binaria)<br>
# [EOF](#EOF)<br>



import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
#plt.style.use('dark_background')
get_ipython().run_line_magic('matplotlib', 'inline')
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec
from sklearn.metrics import f1_score, accuracy_score
import seaborn as sns
sns.set()
custom_style = {'axes.facecolor': 'k',
                'axes.edgecolor': 'gray',
                'axes.labelcolor': 'white',
                'figure.facecolor': 'k',
                'grid.color': 'gray',
                'text.color': 'white',
                'xtick.color': 'white',
                'ytick.color': 'white',
                "xtick.major.size": 0,
                "ytick.major.size": 0}
#sns.set_style(style='whitegrid', rc=custom_style)


# ### KGE



def KGEsp(obs, sim, verbose=False, plot=False):
    """Calcula el coeficiente de eficiencia de Kling-Gupta entre dos mapas (observado y simulado).
    
            KGE = 1 - sqrt[(rho - 1)**2 + (alpha - 1)**2 + (beta - 1)**2]
            
    Donde 'rho' es el coeficiente de correlación de Pearson, 'alpha' es una medida de la variabilidad igual al cociente de las desviaciones típicas simulada y observada, y 'beta' es una medida del sesgo igual al cociente de las medias simulada y observada.
    
    Parámetros:
    -----------
    obs:       array (n,m). Mapa observado
    sim:       array (n,m). Mapa simulado
    verbose:   boolean. Si se quiere mostrar el proceso por pantalla
    plot:      boolean. Si se quieren mostrar gráficos
    
    Salida:
    -------
    kge:        float. Valor de la métrica"""
    
    # 'DataFrame' auxiliar (n,2); 'n' es el nº de celda con dato en ambos mapas
    aux = np.vstack((obs.flatten(), sim.flatten()))
    mask = np.any(np.isnan(aux), axis=0)
    df = pd.DataFrame(data=aux[:,~mask], index=['obs', 'sim']).T
    
    if plot == True:
        sns.jointplot(x='obs', y='sim', data=df, kind='reg')
    
    # coeficiente de correlación
    rho = df.corr().iloc[0,1]
    
    # cociente de las varianzas
    std = df.std()
    alpha = std.sim / std.obs
    # sesgo
    mean = df.mean()
    beta = mean.sim / mean.obs
    
    if verbose == True:
        print('rho = {0:.3f}\talpha = {1:.3f}\tbeta = {2:.3f}'.format(rho, alpha, beta))
        
    # coeficiente de eficiencia de Kling-Gupta
    kge = 1 - np.sqrt((rho - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    if verbose == True:
        print('KGEsp = {0:.3f}'.format(kge))
    
    return kge


# ### SPAEF



def SPAEF(obs, sim, verbose=False, plot=False):
    """Calcula la métrica de rendimiento SPAEF (SPAtial EFicciency) entre dos mapas (observado y simulado).
    
            SPAEF = 1 - sqrt[(rho - 1)**2 + (gamma - 1)**2 + (delta - 1)**2]
            
    Dond 'rho' es el coeficiente de correlación de Pearson, 'gamma' es una medida relativa de la variabilidad igual al cociente entre los coeficientes de variación observados y simulados, y 'delta' es una medida de la variabilidad igual a la intersección entre los histogramas normalizados (N[0,1]) de los dos mapas.
    
    Parámetros:
    -----------
    obs:       array (n,m). Mapa observado
    sim:       array (n,m). Mapa simulado
    verbose:   boolean. Si se quiere mostrar el proceso por pantalla
    plot:      boolean. Si se quieren mostrar gráficos
    
    Salida:
    -------
    spaef:     float. Valor de la métrica"""
    
    # 'DataFrame' auxiliar (n,2); 'n' es el nº de celda con dato en ambos mapas
    aux = np.vstack((obs.flatten(), sim.flatten()))
    mask = np.any(np.isnan(aux), axis=0)
    df = pd.DataFrame(data=aux[:,~mask], index=['obs', 'sim']).T
    
    if plot == True:
        sns.jointplot(x='obs', y='sim', data=df, kind='reg')
    
    # COEFICIENTE DE CORRELACIÓN
    rho = df.corr().iloc[0,1]
    
    # COCIENTE DE LOS COEFICIENTES DE VARIACIÓN
    std = df.std()
    mean = df.mean()
    cv = std / mean
    gamma = cv.sim / cv.obs
    
    # INTERSECCIÓN DE LOS HISTOGRAMAS NORMALIZADOS
    # normalizar los datos
    Zscore = pd.DataFrame(index=df.index, columns=df.columns)
    for col in Zscore.columns:
        Zscore[col] = (df[col] - mean[col]) / std[col]
    # histogramas normalizados
    bins = np.arange(-5, 5.1, 0.25)
    KL = pd.DataFrame(index=bins[:-1], columns=Zscore.columns)
    for col in Zscore.columns:
        KL[col] = np.histogram(Zscore[col].values, bins=bins)[0]
    # calcular la intersección
    delta = KL.min(axis=1).sum() / KL.obs.sum()
    
    if verbose == True:
        print('rho = {0:.3f}\tgamma = {1:.3f}\tdelta = {2:.3f}'.format(rho, gamma, delta))
    
    if plot == True:
        plt.figure()
        sns.displot(Zscore.obs, label='obs');
        sns.displot(Zscore.sim, label='sim')
        plt.legend(fontsize=12);
    
    # CÁLCULO DE SPAEF
    spaef = 1 - np.sqrt((rho - 1)**2 + (gamma - 1)**2 + (delta - 1)**2)
    if verbose == True:
        print('SPAEF = {0:.3f}'.format(spaef))
    
    return spaef


# ### Clasificación binaria



def rendimiento_clasificacion(OBS, SIM, score='f1', average='micro'):
    """Calcula el rendimiento entre una matriz 3D observada y una simulada de datos binarios.
    
    Entradas:
    ---------
    OBS:     raster3D. Observación
    SIM:     raster3D. Simulación
    score:   string. Tipo de métrica a utilizar: 'f1' o 'accuracy'
    average: string. Forma de agregar los datos por clases: 'binary', 'weighted', 'micro' o 'macro'. Sólo se aplica al 'f1_score'
    
    Salidas:
    --------
    rend:    pd.Series. Serie temporal de la función de rendimiento
    """
    
    rend = pd.Series(index=OBS.times, dtype=float)
    for t, time in enumerate(rend.index):
        # 'DataFrame' auxiliar (n,2); 'n' es el nº de celda con dato en ambos mapas
        obs, sim = OBS.data[t,:,:], SIM.data[t,:,:]
        obs = obs.data.flatten()[~obs.mask.flatten()]
        sim = sim.data.flatten()[~sim.mask.flatten()]
        aux = np.vstack((obs, sim))
        mask = np.any(np.isnan(aux), axis=0)
        df = pd.DataFrame(data=aux[:,~mask], index=['obs', 'sim']).T
        # calcular rendimiento en el paso temporal t
        if score == 'f1':
            rend[time] = f1_score(df.obs, df.sim, average=average)
        elif score == 'accuracy':
            rend[time] = accuracy_score(df.obs, df.sim)
        else:
            return 'Función de rendimiento no definida. Escoger entre "f1" o "accuracy".'
        
    return rend


# ### EOF






def rend_espacial(dates, obs, sim, plot=True):
    """Calcula el SPAEF y KGE espacial entre las matrices observadas y simuladas
    
    Parámetros:
    -----------
    dates:     array (t,). Lista de fechas a la que corresponden los mapas
    obs:       array (t,n,m). Mapas observados
    sim:       array (t,n,m). Mapas simulados
    plot:      boolean. Si se quiere mostrar los resultados en forma de gráfico
    
    Salida:
    -------
    rend:      dataframe (t,2). Valores del SPAEF y KGE para cada una de las fechas"""
    
    # comprobar que los datos de entrada son correctos
    if obs.shape != sim.shape:
        print('ERROR. No coinciden las dimensiones de la matriz observada y simulada')
    if len(dates) != obs.shape[0]:
        print('ERROR. No coincide la longitud de "dates" con la primera dimensión de las matrices')
    
    # calcular rendimiento
    rend = pd.DataFrame(index=dates, columns=['KGE', 'SPAEF'])
    for d, date in enumerate(dates):
        rend.loc[date, 'KGE'] = KGEsp(obs[d,:,:], sim[d,:,:])
        rend.loc[date, 'SPAEF'] = SPAEF(obs[d,:,:], sim[d,:,:])
        
    if plot == True:
        fig, ax = plt.subplots(figsize=(8,4))
        lw = 1.2
        ax.plot(rend.KGE, color='lightsteelblue', lw=lw, label='KGE')
        ax.plot(rend.SPAEF, color='indianred', lw=lw, label='SPAEF')
        ax.set(xlim=(dates[0], dates[-1]), ylim=(-2, 1))
        fig.legend(loc=8, bbox_to_anchor=(0.25, -0.025, 0.5, 0.1), ncol=2, fontsize=13)
        plt.tight_layout()
    
    return rend


# In[ ]:




