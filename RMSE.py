def RMSE(data, c_obs='obs', c_sim='sim'):
    """Calcula la raíz del error medio cuadrático."""
    
    # Eliminar pasos sin dato
    data.dropna(axis=0, how='any', inplace=True)
    # Para la función sin no hay datos
    if data.shape[0] == 0:
        return
    # Calcular RMSE
    rmse = np.sqrt(sum((data[c_obs] - data[c_sim])**2) / data.shape[0])
    
    return rmse