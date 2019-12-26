def sesgo(data, c_obs='obs', c_sim='sim'):
    """Calcula el sesgo del hidrograma, es decir, el porcentaje de error en el volumen simulado."""
    
    # Eliminar pasos sin dato
    data.dropna(axis=0, how='any', inplace=True)
    # Para la función sin no hay datos
    if data.shape[0] == 0:
        return
    # Calcular la serie de volumen a partir de la de caudal
    V = data.copy()
    for date in V.index:
        days = monthrange(date.year, date.month)[1]
        seconds = 24 * 3600
        V.loc[date,:] = data.loc[date,:] * days * seconds / 1e6
    # Calcular el sesgo
    try:
        sesgo = (V['sim'].sum() - V['obs'].sum()) / V['obs'].sum() * 100
    except:
        sesgo = np.nan
    return sesgo