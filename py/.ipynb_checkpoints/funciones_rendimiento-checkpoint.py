import numpy as np
import pandas as pd


def NSE(observado, simulado):
    """Calcula el coeficiente de eficiencia de Nash-Sutcliffe.
    
    Parámetros:
    -----------
    observado:   series. Serie observada
    simulado:    series. Serie simulada"""
    
    # Eliminar pasos sin dato
    data = pd.concat((observado, simulado), axis=1)
    data.columns = ['obs', 'sim']
    data.dropna(axis=0, how='any', inplace=True)
    # Para la función si no hay datos
    if data.shape[0] == 0:
        return
    # Calcular NSE
    nse = 1 - sum((data.obs - data.sim)**2) / sum((data.obs - data.obs.mean())**2)
    
    return nse


def RMSE(serie1, serie2):
    """Calcula la raíz del error medio cuadrático.
    
    Parámetros:
    -----------
    serie1:    series. Primera del par de series a comparar
    serie2:    series. Segunda del par de series a comparar"""
    
    data = pd.concat((serie1, serie2), axis=1)
    data.columns = ['obs', 'sim']
    # Eliminar pasos sin dato
    data.dropna(axis=0, how='any', inplace=True)
    # Para la función si no hay datos
    if data.shape[0] == 0:
        print('ERROR. Series no coincidentes')
        return 
    # Calcular RMSE
    rmse = np.sqrt(sum((data.obs - data.sim)**2) / data.shape[0])
    
    return rmse


def sesgo(observado, simulado):
    """Calcula el sesgo del hidrograma, es decir, el porcentaje de error en el volumen simulado.
    
    sesgo = (Vsim - Vobs) / Vobs * 100
    
    Parámetros:
    -----------
    observado:   series. Serie observada
    simulado:    series. Serie simulada"""
    
    # Eliminar pasos sin dato
    data = pd.concat((observado, simulado), axis=1)
    data.columns = ['obs', 'sim']
    data.dropna(axis=0, how='any', inplace=True)
    # Para la función si no hay datos
    if data.shape[0] == 0:
        print('No hay valores')
        return 
    # Calcular el sesgo    
    return (data.sim.sum() - data.obs.sum()) / data.obs.sum() * 100

  
def KGE(observado, simulado, sa=1, sb=1, sr=1):
    """Calcula el coeficiente de eficiencia de Kling-Gupta.
    
    Parámetros:
    -----------
    observado:   series. Serie observada
    simulado:    series. Serie simulada
    sa, sb, sr: integer. Factores de escala de los tres términos del KGE: alpha, beta y coeficiente de correlación, respectivamente
    
    Salida:
    -------
    KGE:        float. Eficienica de Kling-Gupta"""
    
    # Eliminar pasos sin dato
    data = pd.concat((observado, simulado), axis=1)
    data.columns = ['obs', 'sim']
    data.dropna(axis=0, how='any', inplace=True)
    # Para la función si no hay datos
    if data.shape[0] == 0:
        return

    # calcular cada uno de los términos del KGE
    alpha = data.sim.std() / data.obs.std()
    beta = data.sim.mean() / data.obs.mean()
    r = np.corrcoef(data.obs, data.sim)[0, 1]
    
    # Cacular KGE
    ED = np.sqrt((sr * (r - 1))**2 + (sa * (alpha - 1))**2 + (sb * (beta - 1))**2)
    KGE = 1 - ED
    
    return KGE


def matriz_confusion(obs, sim):
    """Calcula la matriz de confunsión del acierto en la ocurrencia o ausencia de precipitación diaria.
    
    Parámetros:
    -----------
    obs:       series. Serie observada
    sim:       series. Serie simulada"""
    
    data = pd.concat((obs, sim), axis=1)
    data.columns = ['obs', 'sim']
    # convertir días de lluvia en 1
    data[data > 0] = 1
    # Eliminar pasos sin dato
    data.dropna(axis=0, how='any', inplace=True)
    # Para la función si no hay datos
    if data.shape[0] == 0:
        print('ERROR. Series no coincidentes')
    # días con lluvia en la observación
    data1 = data[data.obs ==1]
    # días secos en la observación
    data0 = data[data.obs == 0]
    # calcular acierto
    acierto00 = sum(data0.sim == 0) / data0.shape[0]
    acierto01 = sum(data0.sim == 1) / data0.shape[0]
    acierto10 = sum(data1.sim == 0) / data1.shape[0]
    acierto11 = sum(data1.sim == 1) / data1.shape[0]
    acierto = [[acierto00, acierto01],
               [acierto10, acierto11]]
    acierto = pd.DataFrame(acierto, index=['obs0', 'obs1'], columns=['sim0', 'sim1'])
    
    return acierto

    
