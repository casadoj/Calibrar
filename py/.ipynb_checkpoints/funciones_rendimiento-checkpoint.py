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