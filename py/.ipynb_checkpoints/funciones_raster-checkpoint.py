#!/usr/bin/env python
# coding: utf-8

# # Funciones raster
# _Autor:_    __Jesús Casado__ <br> _Revisión:_ __05/12/2020__
# 
# __Índice__ <br>
# [__Clases__](#Clases)<br>
# [`raster2D`](#raster2D)<br>
# [__Funciones__](#Funciones)<br>
# [`interpolarNN`](#interpolarNN)<br>
# [`recortarRaster`](#recortarRaster)<br>
# [`read_ascii`](#read_ascii)<br>
# [`write_ascii`](#write_ascii)<br>
# [`corregirRaster`](#corregirRaster)<br>

# In[1]:


import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import re
from pyproj import Transformer, CRS
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier


# ```Python
# def seriesIVERCAM(file):
#     """Importa los archivos csv con la serie temporal de mapas climáticos generados en IVERCAM.
#     
#     Entradas:
#     ---------
#     file:    string. Nombre del archivo
#     
#     Salidas:
#     --------
#     arr:    array (dates,X,Y). Matriz con los datos
#     X:      array. Coordenadas X del mapa
#     Y:      array. Coordenadas Y del mapa
#     dates:  array. Fechas de cada mapa
#     """
#     
#     # importar csv
#     data = pd.read_csv(file)
# 
#     # coordenadas X e Y
#     XX = data.X
#     X = np.sort(np.unique(XX))
#     YY = data.Y
#     Y = np.sort(np.unique(YY))[::-1]
#     nrow, ncol = len(Y), len(X)
#     
#     # fechas
#     data_ = data.drop(['X', 'Y'], axis=1)
#     dates = list(data_.columns)
#     ntime = len(dates)
#     
#     # array 3D
#     arr = np.empty((ntime, nrow, ncol))
#     arr[:,:,:] = np.nan
#     for n in range(data_.shape[0]):
#         i = np.where(Y == YY[n])[0][0]
#         j = np.where(X == XX[n])[0][0]
#         arr[:,i,j] = data_.iloc[n,:]
# 
#     arr = ma.masked_where(np.isnan(arr), arr)
#     
#     return arr, X, Y, dates
# 
# # precipitación
# pcp, Xp, Yp, Tp = seriesIVERCAM('pcp_pred_1000.csv')
# 
# plt.figure()
# im = plt.imshow(pcp.mean(axis=0))
# cb = plt.colorbar(im, shrink=.75)
# plt.axis('off');
# 
# # temperatura
# tmp, Xt, Yt, Tt = seriesIVERCAM('Tmed_pred_1000_IDWz.csv')
# 
# plt.figure()
# im = plt.imshow(tmp.mean(axis=0))
# cb = plt.colorbar(im, shrink=.75)
# plt.axis('off');
# ```

# ## Clases
# ### raster2D

# In[2]:


class raster2D:
    def __init__(self, data, X, Y, crs, noData=-9999):
        """Clase que contiene los datos que represetan matrices 2D con las dimensiones coordenadas Y, coordenadas X.
        
        Entradas:
        ---------
        data:    array (times, Y, X). Matriz con los datos de la predicción de HARMONIE
        X:       array (X,). Coordenadas X (m o grados) de las columnas de la matriz 'data'
        Y:       array (Y,). Coordenadas Y (m o grados) de las files de la matriz 'data'
        crs:     string o callable. Sistema de coordenadas de referencia. Los datos originales están en 'epsg:4258'
        noData:  int. Valor de dato faltante en el ascii original
        """
        
        self.data = data
        self.X = X
        self.Y = Y
        self.crs = crs
        self.noData = noData
        
        # crear algunos atributos
        cellsize = np.diff(X).mean()
        self.cellsize = cellsize
        self.attributes = [len(X), len(Y), X.min(), Y.min(), cellsize, noData]
        self.extent = [X.min(), X.max() + cellsize, Y.min(), Y.max() + cellsize]


# ### raster3D

# In[3]:


class raster3D:
    def __init__(self, data, X, Y, times, units=None, variable=None, label=None, crs=None):
        """Clase que contiene los datos que represetan matrices 3D con las dimensiones, tiempo, coordenadas Y, coordenadas X.
        
        Entradas:
        ---------
        data:    array (times, Y, X). Matriz con los datos de la predicción de HARMONIE
        X:       array (X,). Coordenadas X (m o grados) de las columnas de la matriz 'data'
        Y:       array (Y,). Coordenadas Y (m o grados) de las files de la matriz 'data'
        times:   array (times,). Fecha y hora de cada uno de los pasos temporales de 'data'
        units:   string. Unidades de la variable. P.ej. 'mm'
        variable: string. Descripción de la variable. P.ej. 'precipitación total'
        label:    string. Etiqueta de la variable. P.ej. 'APCP'
        crs:      string o callable. Sistema de coordenadas de referencia. Los datos originales están en 'epsg:4258'
        """
        
        self.data = data
        self.X = X
        self.Y = Y
        self.times = times
        self.units = units
        self.variable = variable
        self.label = label
        self.crs = crs
        
    def cellsize(self):
        
        return np.diff(self.X).mean()
    
    def extent(self):
        
        X, Y = self.X, self.Y
        cellsize = self.cellsize()
        return [X.min(), X.max() + cellsize, Y.min(), Y.max() + cellsize]
    
    
    def extraer(self, start, end, axis=0, inplace=False):
        """Se extrae una selección del raster3D en uno de los axis.

        Entradas:
        ---------
        self:    objeto de clase raster3D
        start:   valor mínimo dentro del recorte
        end:     valor máximo dentro del recorte
        axis:    int. Eje del raster a recortar

        Salidas:
        --------
        raster:  raster3D.
        """

        def of(array, start, end):
            """Posición inicial y final de 'start' y 'end' dentro del array."""

            o = np.where(array == start)[0][0]
            f = np.where(array == end)[0][0] + 1

            return o, f

        data, X, Y, times = self.data, self.X, self.Y, self.times

        if axis == 0:
            o, f = of(times, start, end)
            data = self.data[o:f,:,:]
            times = times[o:f]
        elif axis == 1:
            o, f = of(Y, start, end)
            data = self.data[:,o:f,:]
            Y = Y[o:f]
        elif axis == 2:
            o, f = of(X, start, end)
            data = self.data[:,:,o:f]
            X = X[o:f]

        if inplace:
            self.data, self.X, self.Y, self.times = data, X, Y, times
        else:
            return raster3D(data, X, Y, times, self.units, self.variable, self.label, self.crs)
    
    def enmascararNaN(self):
        """Enmascara los datos en aquellas celdas con todo NaN en la serie temporal.      
        """
                
        # crear máscara
        mask2D = np.all(np.isnan(self.data), axis=0)
        mask3D = np.zeros(self.data.shape, dtype=bool)
        mask3D[:,:,:] = mask2D[np.newaxis,:,:]

        self.data = np.ma.masked_array(self.data, mask3D)
        
        
    def recortar(self, poligono, buffer=None, inplace=False):
        """Recorta los datos de raster3D según el polígono.

        Entradas:
        ---------
        self:      class raster3D
        poligono:  geopandas.GeoDataframe. Polígono con el que recortar los mapas
        buffer:    float. Distancia a la que hacer una paralela al polígono antes del recorte
        inplace:   boolean. Si se quiere sobreescribir el resultado sobre self o no
        
        Salidas:
        --------
        Si 'inplace == False':
            modis: class raster3D
        """
        
        # extraer información de 'HARMONIE'
        X, Y = self.X, self.Y
        data = self.data

        # buffer
        if buffer is not None:
            poligono = poligono.buffer(buffer)

        # definir crs del polígono
        if self.crs != poligono.crs:
            poligono = poligono.to_crs(self.crs)

        # extensión de la cuenca
        left, bottom, right, top = poligono.bounds.loc[0,:]
        # buffer
        if buffer is not None:
            left -= buffer
            bottom -= buffer
            right += buffer
            top += buffer

        # recortar según la extensión de la cuenca
        maskC = (X >= left) & (X <= right)
        maskR = (Y >= bottom) & (Y <= top)
        data = data[:,maskR,:][:,:,maskC]
        X = X[maskC] + self.cellsize() / 2
        Y = Y[maskR] + self.cellsize() / 2

        # GeoDataFrame de puntos de la malla raster3D
        XX, YY = np.meshgrid(X.flatten(), Y.flatten())
        points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(XX.flatten(), YY.flatten(), crs=self.crs))

        # máscara del polígono
        inp, res = poligono.sindex.query_bulk(points.geometry, predicate='intersects')
        mask = np.isin(np.arange(len(points)), inp)
        mask2D = mask.reshape(XX.shape)

        # máscara 3D a partir de la anterior
        mask3D = np.zeros(data.shape, dtype=bool)
        mask3D[:,:,:] = mask2D[np.newaxis,:,:]
        
        # recortar mapa al área del polígono
    #     data_ma = np.ma.masked_array(data, ~mask3D)
        data_ma = data.copy()
        data_ma[~mask3D] = np.nan

        # eliminar filas y columnas sin datos
        maskR = np.isnan(data_ma.sum(axis=0)).all(axis=1)
        maskC = np.isnan(data_ma.sum(axis=0)).all(axis=0)
        data_ma = data_ma[:,~maskR,:][:,:,~maskC]
    #     data_ma = np.ma.masked_invalid(data_ma)
        Y = Y[~maskR] - self.cellsize() / 2
        X = X[~maskC] - self.cellsize() / 2
        
        if inplace:
            self.data = data_ma
            self.X = X
            self.Y = Y
            self.mask3D = mask3D
        else:
            # crear diccionario con los resultados  
            modis = raster3D(data_ma, X, Y, self.times, self.units, self.variable, self.label, crs=self.crs)
            modis.mask3D = mask3D
            return modis
        

    def reproyectar(self, crsOut, cellsize, n_neighbors=1, weights='distance', p=2,
                    snap=None, inplace=False):
        """Proyecta la malla de raster3D desde su sistema de coordenadas original (sinusoidal) al sistema deseado en una malla regular de tamaño definido.

        Entradas:
        ---------
        self:        class raster3D
        crsOut:      CRS. Sistema de coordenadas de referencia al que se quieren proyectar los datos. P.ej. 'epsg:25830'
        cellsize:    float. Tamaño de celda de la malla a generar
        n_neighbors: int. Nº de celdas cercanas a utilizar en la interpolación
        weights:     str. Tipo de ponderación en la interpolación
        p:           int. Exponente de la ponderación
        inplace:   boolean. Si se quiere sobreescribir el resultado sobre self o no

        Salida:
        -------
        Si 'inplace == False':
            harmonie: class HARMONIE
        """

        # extraer información de HARMONIE
        data = self.data
        Y = self.Y
        X = self.X
        times = self.times
        crsIn = self.crs

        # matrices de longitud y latitud de cada una de las celdas
        XX, YY = np.meshgrid(X, Y)

        # transformar coordendas y reformar en matrices del mismo tamaño que el mapa diario
        transformer = Transformer.from_crs(crsIn, crsOut) 
        Xorig, Yorig = transformer.transform(XX.flatten(), YY.flatten())
        XXorig = Xorig.reshape(XX.shape)
        YYorig = Yorig.reshape(YY.shape)

        # definir límites de la malla a interpolar
        if snap is None:
            xmin, xmax, ymin, ymax = Xorig.min(), Xorig.max(), Yorig.min(), Yorig.max()
            # redondear según el tamaño de celda
            xmin = int(np.floor(xmin / cellsize) * cellsize)
            xmax = int(np.ceil(xmax / cellsize) * cellsize)
            ymin = int(np.floor(ymin / cellsize) * cellsize)
            ymax = int(np.ceil(ymax / cellsize) * cellsize)

            # coordenadas X e Y de la malla a interpolar
            Xgrid = np.arange(xmin, xmax + cellsize, cellsize)
            Ygrid = np.arange(ymin, ymax + cellsize, cellsize)[::-1]
        else:
            crsOut = snap.crs
            Xgrid = snap.X
            Ygrid = snap.Y

        # matrices de X e Y de cada una de las celdas de la malla a interpolar
        XXgrid, YYgrid = np.meshgrid(Xgrid, Ygrid)

        # interpolar mapas en la malla
        data_ = np.empty((len(times), len(Ygrid), len(Xgrid)), dtype=float)
        for t, time in enumerate(times):
            print('Paso {0} de {1}:\t{2}'.format(t+1, len(times), time), end='\r')
            data_[t,:,:] = interpolarNN(XXorig, YYorig, data[t,:,:], XXgrid, YYgrid,
                                        n_neighbors=n_neighbors,  weights=weights, p=p)
        
        if inplace:
            self.data = data_
            self.X = Xgrid
            self.Y = Ygrid
            self.crs = crsOut
        else:
            # crear nueva instancia de clase HARMONIE
            modis = raster3D(data_, Xgrid, Ygrid, self.times, crs=crsOut)#, self.units, self.variable, self.label, crsOut)
            return modis
        
    def plot(self, time=None, ax=None, **kwargs):
        """
        """
        
        cmap = kwargs.get('cmap', 'viridis')
        figsize = kwargs.get('figsize', None)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        if time is None:
            ax.imshow(self.data.mean(axis=0), extent=self.extent(), cmap=cmap)
        else:
            if isinstance(time, int):
                t = time
            else:
                t = np.where(self.times == time)[0][0]

            ax.imshow(self.data[t,:,:], extent=self.extent(), cmap=cmap)
        
        ax.set_aspect('equal')
        ax.axis('off')


# ## Funciones
# ### interpolarNN

# In[4]:


def interpolarNN(XXorig, YYorig, mapa, XXgrid, YYgrid, n_neighbors=1, weights='distance', p=1):
    """Interpolar un mapa desde una malla original a otra malla regular. Se utiliza el algoritno de vencinos cercanos.
    Utilizando como pesos 'distance' y como exponente 'p=2' es el método de la distancia inversa al cuadrado.
    
    Entradas:
    ---------
    XXorig:      np.array (r1, c1). Coordenadas X de los puntos del mapa de origen
    YYorig:      np.array (r1, cw). Coordenadas Y de los puntos del mapa de origen
    mapa:        np.array (r1, c1). Valores de la variable en los puntos del mapa de origen
    XXgrid:      np.array (r2, c2). Coordenadas X de los puntos del mapa objetivo
    YYgrid:      np.array (r2, c2). Coordenadas Y de los puntos del mapa objetivo
    n_neighbors: int. Nº de vecinos cercanos, es decir, los puntos a tener en cuenta en la interpolación de cada celda de la malla.
    weights:     str. Tipo de ponderación: 'uniform', 'distance'
    p:           int. Exponente al que elevar 'weights' a la hora de ponderar
    
    Salida:
    -------
    pred:         np.array (r2, c2). Valores de la varible interpolados en la mall objetivo
    """
    
    # AJUSTE
    # ......
    # target array
    if isinstance(mapa, np.ma.MaskedArray):
        y = mapa.data.flatten().astype(float)
    else:
        y = mapa.flatten().astype(float)
    mask = np.isnan(y)
    y = y[~mask]
    # feature matrix
#     XXorig, YYorig = np.meshgrid(Xorig, Yorig)
    X = np.vstack((XXorig.flatten(), YYorig.flatten())).T
    X = X[~mask,:]
    # definir y ajustar el modelo
    neigh = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, p=p).fit(X, y)
    
    # PREDICCIÓN
    # ..........
    # feauture matrix
#     XXgrid, YYgrid = np.meshgrid(Xgrid, Ygrid)
    X_ = np.vstack((XXgrid.flatten(), YYgrid.flatten())).T
    # predecir
    pred = neigh.predict(X_).reshape(XXgrid.shape)
    
    return pred


# ### recortarRaster

# In[5]:


def recortarRaster(data, X, Y, polygon, crs=None, buffer=None):
    """Recorta los datos 3D según un shapefile de polígono.
    
    Entradas:
    ---------
    data:    array (T,Y,X). Serie temporal de mapas
    X:       array (X,). Coordenadas X de las columnas del mapa
    Y:       array (Y,). Coordenadas Y de las filas del mapa
    polygon: geopandas.Series. Polígono con el que recortar los datos
    crs:     Sistema de coordenadas de X e Y
    buffer:  boolean. Distancia a la que hacer el buffer. Por defecto es None y no se hace
    
    Salidas:
    --------
    data_:   array (T,Y_,X_). Serie temporal de mapas recortados
    X_:      array. Coordenadas X de las columnas del mapa 'data_'
    Y_:      array. Coordenadas Y de las filas del mapa 'data_' 
    """
    
    # crear buffer de la cuenca
    if buffer is not None:
        mask_shp = polygon.buffer(buffer)
    else:
        mask_shp = polygon
        
    # GeoDataFrame de puntos de la malla HARMONIE reproyectada
    XX, YY = np.meshgrid(X, Y)
    if crs is None:
        crs = cuenca.crs
    points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(XX.flatten(), YY.flatten(),
                                                          crs=crs))

    # máscara con el área de la CHC
    inp, res = mask_shp.sindex.query_bulk(points.geometry, predicate='intersects')
    mask1D = np.isin(np.arange(len(points)), inp)
    mask2D = mask1D.reshape(XX.shape)
    
    if len(data.shape) == 2:
        # recortar 'data' a la cuenca
        data_ = data.copy()
        data_[~mask2D] = np.nan
        
        # eliminar filas y columnas sin datos
        maskR = np.isnan(data_).all(axis=1)
        maskC = np.isnan(data_).all(axis=0)
        data_ = data_[~maskR,:][:,~maskC]
        
    elif len(data.shape) == 3:
        # máscara 3D a partir de la anterior
        mask3D = np.zeros(data.shape, dtype=bool)
        mask3D[:,:,:] = mask2D[np.newaxis,:,:]

        # recortar 'data' a la cuenca
        data_ = data.copy()
        data_[~mask3D] = np.nan

        # eliminar filas y columnas sin datos
        maskR = np.isnan(data_.sum(axis=0)).all(axis=1)
        maskC = np.isnan(data_.sum(axis=0)).all(axis=0)
        data_ = data_[:,~maskR,:][:,:,~maskC]
    
    data_ = np.ma.masked_invalid(data_)
    Y_ = Ygrid[~maskR]
    X_ = Xgrid[~maskC]
    
    return data_, X_, Y_



# ### read_ascii

# In[6]:


def read_ascii(filename, datatype='float', crs=None):
    """Import an ASCII file. Data is saved as a 2D numpy array and the attributes as integers or floating point numbers.

    Parameters:
    -----------
    filename:     string. Name (including path and extension) of the ASCII file

    Output:
    -------
    Results are given as methods of the function
        attributes:   list. A list of six attributes:
        ncols:        int. Number of columns
        nrows:        int. Number of rows
        xllcorner:    float. X coordinate of the left lower corner
        yllcorner:    float. Y coordinate of the left lower corner
        cellsize:     int. Spatial discretization
        NODATA_value: float. Value representing no data
        data:         naddary[nrows,ncols]. The data in the map"""

    with open(filename, 'r+') as file:
        # import all the lines in the file
        asc = file.readlines()
        # extract attributes
        ncols = int(asc[0].split()[1])
        nrows = int(asc[1].split()[1])
        xllcorner = float(asc[2].split()[1])
        yllcorner = float(asc[3].split()[1])
        cellsize = float(asc[4].split()[1])
        NODATA_value = float(asc[5].split()[1])
        #attributes = [ncols, nrows, xllcorner, yllcorner, cellsize, NODATA_value]
        # extract data
        data = np.zeros((nrows, ncols))
        for i in range(nrows):
            data[i, :] = asc[i + 6].split()
        data[data == NODATA_value] = np.nan
        data = np.ma.masked_invalid(data)
        data = data.astype(datatype)
    file.close()
    
    # guardar objeto como raster2D
    X = np.arange(xllcorner, xllcorner + ncols * cellsize, cellsize)
    Y = np.arange(yllcorner, yllcorner + nrows * cellsize, cellsize)[::-1]
    if crs is not None:
        crs = CRS.from_epsg(crs)
    raster = raster2D(data, X, Y, crs, noData=NODATA_value)

    return raster


# ### write_ascii

# In[9]:


def write_ascii(filename, raster, format='%.0f '):
    """Export a 2D numpy array and its corresponding attributes as an ascii raster. It may also create the '.proj' file that defines the coordinate system.

    Parameters:
    -----------
    filename:     string. Name (including path and extension) of the ASCII file
    data:         narray. 2D array with the data to be exported
    attributes:   narray[6x1]. Array including the following information: ncols, nrows, xllcorner, yllcorner, cellsize, NODATA_value
    format:       string. Format in which the values in 'data' will be exported
    epsg:       srting. EGSG code (only figures) of the reference coordinate system. If 'None', it doesn't create the 'proj' file

    Output:
    -------
    An .asc raster file. Optionally, the associated .prj file
    """
    
    # extract info from raster
    aux = raster.data.copy()
    attributes = raster.attributes

    # unmask data if masked
    if np.ma.is_masked(aux):
        np.ma.set_fill_value(aux, attributes[5])
        aux = aux.filled()

    # convert NaN to NODATA_value
    aux[np.isnan(aux)] = attributes[5]

    # export ascii
    with open(filename, 'w+') as ascfile:
        # write attributes
        ascfile.write('ncols\t\t{0:<8}\n'.format(attributes[0]))
        ascfile.write('nrows\t\t{0:<8}\n'.format(attributes[1]))
        ascfile.write('xllcorner\t{0:<8}\n'.format(attributes[2]))
        ascfile.write('yllcorner\t{0:<8}\n'.format(attributes[3]))
        ascfile.write('cellsize\t{0:<8}\n'.format(attributes[4]))
        ascfile.write('NODATA_value\t{0:<8}\n'.format(attributes[5]))
        # write data
        for i in range(aux.shape[0]):
            #values = df.iloc[i, 6:].tolist()
            values = aux[i, :].tolist()
            ascfile.writelines([format % item  for item in values])
            ascfile.write("\n")
    ascfile.close()

    # export proj file
    try:
        wkt = raster.crs.to_wkt()
        # create the .prj file
        with open(filename[:-4] + '.prj', 'w') as projfile:
            projfile.write(wkt)
        projfile.close()
    except:
        pass


# ### corregirRaster

# In[8]:


def corregirRaster(array, X=None, Y=None):
    """Elimina filas y columnas sin ningún dato
    
    Entradas:
    ---------
    array:   array (m,n)
    X:       array (n,). Coordenada X de las columnas
    Y:       array (m,). Coordenada Y de las filas
    
    Salidas:
    --------
    array:   array (m',n')
    X:       array (n',). Coordenada X de las columnas
    Y:       array (m',). Coordenada Y de las filas
    """
    
    # recortar filas y columnas vacías
    maskC = np.all(np.isnan(array), axis=0)
    maskR = np.all(np.isnan(array), axis=1)
    array = array[~maskR,:][:,~maskC]
    if (X is not None) & (Y is not None):
        X = X[~maskC]
        Y = Y[~maskR]
        return array, X, Y
    else:
        return array

