#!/usr/bin/env python
# coding: utf-8

# # Modelo de nieve
# _Autor:_    __Jesús Casado__ <br> _Revisión:_ __05/12/2020__
# 
# __Índice__ <br>
# [`degreeDayMethod`](#degreeDayMethod)<br>
# [`sonwCover`](#snowCover)<br>
# [`remuestrearMODIS`](#remuestrearMODIS)<br>

import numpy as np
import numpy.ma as ma
from funciones_raster import *

# ### degreeDayMethod

# In[2]:


def degreeDayMethod(PCP, TMP, RAD=None, Ts=0., Tb=0., DDF1=2., DDF2=4., verbose=True):
    """Calcula la acumulación de nieve y su fusión por el método del índice grado-día.
    
    Entradas:
    ---------
    PCP:     raster3D. Objeto con los datos de la serie temporal (t,x,y) de mapas de precipitación (mm)
    TMP:     raster3D. Objeto con los datos de la serie temporal (t,x,y) de mapas de temperatura (media o mínima) (°C)
    RAD:     raster3D. raster3D. Objeto con los datos de la serie temporal (12,x,y)  de mapas de radiación difusa normalizados. Si es None, no se utiliza la radiación para modificar los índices grado-día
    Ts:      float. Umbral de temperatura por debajo del cual se asume que la precipitación es en forma de nieve
    Tb:      float. Umbral de temperatura por encima del cual se produce derretimiento
    DDF1:    float. Índice grado-día (mm/°C) para tiempo seco
    DDF2:    float. Índice grado-día (mm/°C) para tiempo lluvioso
    verbose: boolean. Mostrar por pantalla el proceso
    
    Salidas:
    --------
    swe:     array (t,x,y). Serie temporal de mapas de equivalente agua-nieve (mm)
    sm:      array (t,x,y). Serie temporal de mapas de fusión de nieve (mm)
    """
    
    # comprobaciones iniciales
    if PCP.data.shape != TMP.data.shape:
        return 'ERROR. Las arrays climáticos no tienen las mismas dimensiones.'
    if RAD is not None:
        if RAD.data.shape[0] != 12:
            return 'ERROR. El array de radiación debe tener 12 capas (meses) en el eje 0.'
        if RAD.data.shape[1:] != PCP.data.shape[1:]:
            return 'ERROR. La dimensión espacial (ejes 1 y 2) del array de radiación no coincide con la de los arrays climáticos.'
    
    # extrar datos
    pcp, tmp, dates = PCP.data, TMP.data, PCP.times
    if RAD is not None:
        rad = RAD.data
    
    if isinstance(pcp, np.ma.MaskedArray):
        # array 3D en blanco
        arr3D = ma.masked_array(np.zeros(pcp.shape), pcp.mask)
        # array 2D en blanco
        arr2D = ma.masked_array(np.zeros(pcp[0,:,:].shape), pcp[0,:,:].mask)
    else:
        # array 3D en blanco
        arr3D = np.array(np.zeros(pcp.shape))
        # array 2D en blanco
        arr2D = np.array(np.zeros(pcp[0,:,:].shape))
    
    # arrays donde guardar los resultados
    swe = arr3D.copy()  # equivalente agua-nieve (snow-water equivalent)
    sm = arr3D.copy()   # fusión de la nieve (snowmelt)
    

    for i, date in enumerate(dates):
        if verbose:
            print('Paso {0:<3} de {1:<3}:\t{2}'.format(i+1, len(dates), date), end='\r')
            
        # extraer mes
        month = date.month

        # extraer mapas con los datos
        pcp_i = pcp[i,:,:]   # precipitación
        tmp_i = tmp[i,:,:]   # temperatura
        swe_0 = swe[i-1,:,:]  # equivalente agua-nieve inicial

        # precipitación en forma de nieve
        snowfall = arr2D.copy()
        maskT = tmp_i < Ts
        snowfall[maskT] = pcp_i[maskT]

        # fusión de la nieve potencial (snowmeltP)
        snowmeltP = arr2D.copy()
        # en celdas sin lluvia
        dry = pcp_i <= 0
        if RAD is None:
            snowmeltP[dry] = DDF1 * (tmp_i[dry] - Tb)
        else:
            snowmeltP[dry] = DDF1 * rad[month-1,:,:][dry] * (tmp_i[dry] - Tb)
        # en celdas con lluvia
        rain = pcp_i > 0
        if RAD is None:
            snowmeltP[rain] = DDF2 * (tmp_i[rain] - Tb)
        else:
            snowmeltP[rain] = DDF2 * rad[month-1,:,:][rain] * (tmp_i[rain] - Tb)
        # convertir en 0 valores negativos (cuando la temperatura es inferior a Tb)
        snowmeltP[snowmeltP < 0] = 0

        # equivalente de agua-nieve disponible
        swe_i = swe_0 + snowfall

        # fusión de la nieve real (snowmelt_)
        snowmeltR = np.minimum(swe_i, snowmeltP)
        sm[i,:,:] = snowmeltR.copy()

        # equivalente agua-nieve final
        swe_i -= snowmeltR
        swe[i,:,:] = swe_i.copy()
    
    # enmascarar los array de resultados
    if isinstance(pcp, np.ma.MaskedArray) is False:
        mask2D = np.all(np.isnan(pcp), axis=0)
        # máscara 3D a partir de la anterior
        mask3D = np.zeros(pcp.shape, dtype=bool)
        mask3D[:,:,:] = mask2D[np.newaxis,:,:]
        
        swe = np.ma.masked_array(swe, mask3D)
        sm = np.ma.masked_array(sm, mask3D)
        
    SWE = raster3D(swe, PCP.X, PCP.Y, dates, units='mm', variable='snow water equivalent', label='SWE', crs=PCP.crs)
    SM = raster3D(sm, PCP.X, PCP.Y, dates, units='mm', variable='snowmelt', label='SM', crs=PCP.crs)
        
    return SWE, SM


# ### snowCover

# In[ ]:


def snowCover(SWE, threshold=1):
    """Convierte un raster3D con datos de equivalente agua-nieve en un raster3D binario de cobertura de nieve
    
    Entradas:
    ---------
    SWE:       raster3D. Equivalente agua-nieve
    threshold: float. Valor de SWE a partir del que se asume que la celda está cubierta por nieve
    
    Salida:
    -------
    SWEbin:    raster3D. Cobertura de nieve
    """
    
    # extraer información
    swe = SWE.data.copy()
    data, mask = swe.data, swe.mask
    
    # reclasificar SWE en un mapa binario
    data[data >= 1] = 1
    data[data < 1] = 0
    
    # generar raster3D de salida
    data = np.ma.masked_array(data, mask)
    SWEbin = raster3D(data, SWE.X, SWE.Y, SWE.times, units='-', variable='modelled snow cover',
                      label='SCmod', crs=SWE.crs)
    
    return SWEbin


# ### remuestrearMODIS

# In[ ]:


def remuestrearMODIS(OBS, SIM, func='max'):
    """Agrega los resultados de la simulación (resolución diaria) a la frecuencia de la observación (8 días).
    
    Entradas:
    ---------
    OBS:     raster3D. Observación
    SIM:     raster3D. Simulación
    func:    string. Función con la que agregar los datos: 'max', 'min', 'mean'
    
    Salida:
    -------
    SIM_:    raster3D. Simulación agregada
    """
    
    # extraer información
    dataObs = OBS.data.copy()
    timesObs = OBS.times
    dataSim = SIM.data.copy()
    timesSim = SIM.times

    data = np.zeros(dataObs.shape) * np.nan
    timef = None
    i = 0
    while i < len(timesObs):
        # fechas de inicio y fin del periodo
        timeo = timef
        timef = timesObs[i]
        # convertir fechas en posición en el eje 0 del array
        if i != 0:
            to = np.where(timesSim == timeo)[0][0]
        else:
            to = None
        tf = np.where(timesSim == timef)[0][0] + 1
        # aplicar función al periodo entre to y tf
        if func == 'max':
            data[i,:,:] = np.nanmax(dataSim[to:tf,:,:], axis=0)
        elif func == 'mean':
            data[i,:,:] = np.nanmean(dataSim[to:tf,:,:], axis=0)
        elif func == 'min':
            data[i,:,:] = np.nanmin(dataSim[to:tf,:,:], axis=0)
            
        i += 1

    data = np.ma.masked_array(data, dataObs.mask)
    
    return raster3D(data, SIM.X, SIM.Y, timesObs, SIM.units, SIM.variable, SIM.label,
                    crs=SIM.crs)


# In[ ]:




