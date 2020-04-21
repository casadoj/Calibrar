def rendimiento(observed, simulated, aggregation='mean', start=None, end=None):
    """Sobre un par de series diarias (observada y simulada) genera la serie mensual según el método de agregración definido.
    Posteriormente, sobre estas dos series, calcula tres criterios de rendimiento: el coeficiente de eficiencia de Nash-
    Sutcliffe (NSE), el sesgo en volumen (%V) y la raíz del error cuadrático medio (RMSE) para cada cuenca en las series
    y los coloca en un 'data frame' que posteriormente exporta.
    
    Parámetros:
    -----------
    observed:    dataframe. Serie observada; cada fila representa una fila y cada columna una estación.
    simulated:   dataframe. Serie simulada; cada fila representa una fila y cada columna una estación.
    aggregation: string. Tipo de agregación de los datos diarios a mensuales.
    skipmonths:  integer. Número de meses a evitar en el cálculo de los criterios por considerarse calentamiento.
    nmonths:     integer. Número de meses a tener en cuenta a partir de 'skipmonths'.
    
    Salidas:
    --------
    obs_m:       dataframe. Serie agregada mensual de las observaciones. Sin recortar
    sim_m:       dataframe. Serie agregada mensual de la simulación. Sin recortar
    performance: dataframe. Valores de los tres criterios para cada estación y el periodo de estudio."""
    
    # Calcular las series mensuales
    # -----------------------------
    obs_m = observed.groupby(by=[observed.index.year, observed.index.month]).aggregate([aggregation])
    sim_m = simulated.groupby(by=[simulated.index.year, simulated.index.month]).aggregate([aggregation])
    # Corregir los índices
    dates = []
    for date in obs_m.index:
        dates.append(datetime.datetime(year=date[0], month=date[1], day=monthrange(date[0], date[1])[1]))
    obs_m.index = dates
    obs_m.index.name = 'Fecha'
    dates = []
    for date in sim_m.index:
        dates.append(datetime.datetime(year=date[0], month=date[1], day=monthrange(date[0], date[1])[1]))
    sim_m.index = dates
    sim_m.index.name = 'Fecha'
    del dates
    #return obs_m, sim_m
    # Corregir los nombres de las columnas
    obs_m.columns = obs_m.columns.levels[0]
    sim_m.columns = sim_m.columns.levels[0]
    
    # Calcular el rendimiento
    # -----------------------
    # Definir las estaciones que están en ambas series
    stns = []
    for stn in observed.columns:
        if stn in simulated.columns: stns.append(stn)
    # Recortar las series a las fechas indicadas
    st = pd.datetime(start[0], start[1], monthrange(start[0], start[1])[1]).date()
    en = pd.datetime(end[0], end[1], monthrange(end[0], end[1])[1]).date()
    dates = pd.date_range(st, en, freq='M')
    #obs_m, sim_m = obs_m.loc[st:en, :], sim_m.loc[st:en, :]
    # Crear el 'data frame' donde se guardará el rendimiento de cada cuenca
    performance = pd.DataFrame(index=['NSE', '%V', 'RMSE'], columns=stns)
    performance.index.name = 'criterio'
    # Calcular el rendimiento de cada cuenca
    for stn in stns:
        # 'data frame' con la observación y la simulación de una estación
        data = pd.DataFrame(index=dates, columns=['obs', 'sim'])
        data['obs'], data['sim'] = obs_m.loc[dates, stn].values, sim_m.loc[dates, stn].values
        # Calcular criterios
        performance.loc['NSE', stn] = NSE(data, 'obs', 'sim')
        performance.loc['%V', stn] = sesgo(data, 'obs', 'sim')
        performance.loc['RMSE', stn] = RMSE(data, 'obs', 'sim')
        #del data

    return obs_m, sim_m, performance
