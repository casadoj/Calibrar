def NSE(data, c_obs='obs', c_sim='sim'):
    """Calcula el coeficiente de eficiencia de Nash-Sutcliffe.
    
    Parámetros:
    -----------
    data:      data frame. """
    
    # Eliminar pasos sin dato
    data.dropna(axis=0, how='any', inplace=True)
    # Para la función sin no hay datos
    if data.shape[0] == 0:
        return
    # Calcular NSE
    nse = 1 - sum((data[c_obs] - data[c_sim])**2) / sum((data[c_obs] - data[c_obs].mean())**2)
    
    return nse