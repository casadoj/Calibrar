#!/usr/bin/env python
# coding: utf-8

# # Funciones raster
# _Autor:_    __Jesús Casado__ <br> _Revisión:_ __05/12/2020__
# 
# __Índice__ <br>
# [__Clases__](#Clases)<br>
# [`MODIS`](#MODIS)<br>
# 
# [__Funciones__](#Funciones)<br>
# [`MODIS_extract`](#MODIS_extract)<br>
# [`video`](#video)<br>
# [`video2`](#video2)<br>



import os
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
import subprocess
import re
from pyproj import Transformer, CRS
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.animation as animation
from IPython.display import HTML


# ## Clases



# sistema de proyección de MODIS
# https://spatialreference.org/ref/sr-org/modis-sinusoidal/
sinusoidal = CRS.from_proj4('+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs ')


# ### MODIS



class MODIS:
    def __init__(self, data, X, Y, times, units=None, variable=None, label=None, crs=sinusoidal):
        """Clase que contiene la información relevante de las predicciones del modelo HARMONIE para una variable concreta.
        
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
        
        
    def recortar(self, poligono, buffer=None, inplace=False):
        """Recorta los datos de MODIS según el polígono.

        Entradas:
        ---------
        self:      class MODIS
        poligono:  geopandas.GeoDataframe. Polígono con el que recortar los mapas
        buffer:    float. Distancia a la que hacer una paralela al polígono antes del recorte
        inplace:   boolean. Si se quiere sobreescribir el resultado sobre self o no
        
        Salidas:
        --------
        Si 'inplace == False':
            modis: class MODIS
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

        # GeoDataFrame de puntos de la malla MODIS
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
            modis = MODIS(data_ma, X, Y, self.times, self.units, self.variable, self.label, crs=self.crs)
            modis.mask3D = mask3D
            return modis
        

    def reproyectar(self, crsOut, cellsize, n_neighbors=1, weights='distance', p=2,
                    snap=None, inplace=False):
        """Proyecta la malla de MODIS desde su sistema de coordenadas original (sinusoidal) al sistema deseado en una malla regular de tamaño definido.

        Entradas:
        ---------
        self:        class MODIS
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
            modis = MODIS(data_, Xgrid, Ygrid, self.times, crs=crsOut)#, self.units, self.variable, self.label, crsOut)
            return modis


# ```Python
#     def enmascarar(self, mask3D, inplace=False):
#         """Recorta los datos de HARMONIE según el polígono 
# 
#         Entradas:
#         ---------
#         self:      class HARMONIE
#         poligono:  geopandas.GeoDataframe. Polígono con el que recortar los mapas
#         crs:       str or callable. Sistema de coordenadas de referencia. Si es None, se toma el de 'poligono'
#         buffer:    float. Distancia a la que hacer una paralela al polígono antes del recorte
#         inplace:   boolean. Si se quiere sobreescribir el resultado sobre self o no
#         
#         Salidas:
#         --------
#         Si 'inplace == False':
#             harmonie: class HARMONIE
#         """
# 
#         # extraer información de 'HARMONIE'
#         X, Y, crs = self.X, self.Y, self.crs
#         data = self.data
# 
#         # recortar mapa al área del polígono
#     #     data_ma = np.ma.masked_array(data, ~mask3D)
#         data_ma = data.copy()
#         data_ma[~mask3D] = np.nan
# 
#         # eliminar filas y columnas sin datos
#         maskR = np.isnan(data_ma.sum(axis=0)).all(axis=1)
#         maskC = np.isnan(data_ma.sum(axis=0)).all(axis=0)
#         data_ma = data_ma[:,~maskR,:][:,:,~maskC]
#     #     data_ma = np.ma.masked_invalid(data_ma)
#         Y = Y[~maskR]
#         X = X[~maskC]
#         
#         if inplace:
#             self.data = data_ma
#             self.X = X
#             self.Y = Y
#             self.mask3D = mask3D
#         else:
#             # crear diccionario con los resultados  
#             harmonie = HARMONIE(data_ma, X, Y, self.times, self.units, self.variable,
#                                  self.label, self.crs)
#             harmonie.mask3D = mask3D
#             return harmonie
#     
#     
#     def agregar(self, freq, func='mean', inplace=False):
#         """Agrega los datos de HARMONIE a la resolución temporal deseada
#         
#         Entradas:
#         ---------
#         self:        class HARMONIE
#         freq:    float. Frecuencia temporal (h) a la que se quiere agregar la serie
#         func:    string. Tipo de agregación: 'mean' o 'sum'
#         inplace:   boolean. Si se quiere sobreescribir el resultado sobre self o no
# 
#         Salida:
#         -------
#         Si 'inplace == False':
#             harmonie: class HARMONIE
#         """
#         
#         # extraer datos de HARMONIE
#         data = self.data
#         times = self.times
#         
#         # pasada de HARMONIE en formato datetime
#         time_p = times[0]        
# 
#         # array
#         for i, ho in enumerate(np.arange(0, 48, freq)):
#             time_o = time_p + timedelta(hours=int(ho))
#             time_f = time_o + timedelta(hours=freq)
#             to = np.where(times == time_o)[0][0]
#             try:
#                 tf = np.where(np.array(times) == time_f)[0][0] + 1
#             except:
#                 tf = None
#             if func == 'sum':
#                 temp = data[to:tf,:,:].sum(axis=0)[np.newaxis,:,:]
#             elif func == 'mean':
#                 temp = data[to:tf,:,:].mean(axis=0)[np.newaxis,:,:]
#             if 'dataAg' in locals():
#                 dataAg = np.vstack((dataAg, temp))
#                 times_.append(time_o)
#             else:
#                 dataAg = temp
#                 times_ = [time_o]
#         times_ = np.array(times_)
#         
#         if inplace:
#             self.data = dataAg
#             self.times = times_
#         else:
#             # crear nueva instancia de clase HARMONIE
#             harmonie = HARMONIE(dataAg, self.X, self.Y, times_, self.units, self.variable, self.label, self.crs)
#             return harmonie
#     
#     
#     def mapa(self, cuencas, provincias=None, simbologia=None, export=True, **kwargs):
#         """Crea una figura con los mapas diarios de la predicción de HARMONIE.
#         
#         Entradas:
#         ---------
#         self:       class HARMONIE
#         cuencas:    geodataframe. Polígonos de las cuencas
#         provincias: geodataframe. Polígonos de las provincias
#         simbologia: dataframe. Tabla con la simbología a utilizar en el mapa. Debe contener las columnas 'Dato', 'Red', 'Green', 'Blue' y 'alpha' correspondientes a los umbrales de cada color, la combinación de colores RGP y la transparencia
#         export:     boolean. Si se quiere exportar la figura como pdf
#         
#         kwargs:
#                     figsize:    list (,). Tamaño de la figura. Por defecto (16, 7)
#                     color:      string. Color con el que definir los límites de las provincias. Por defecto es blanco
#                     dateformat: string. Formato de las fechas en la figura. Por defecto '%d-%m-%Y %H:%M'
#                     extent:     tuple [O, E, S, N]. Límites espaciales de la figura. Por defecto 'None'
#                     rutaExport: string. Carpeta donde guardar el pdf. Por defecto '../output/'
#         
#         Salidas:
#         --------
#         Mapas diarios
#         Si 'export == True', se exporta un pdf con el nombre 'label_pasada.pdf'. Por ejemplo, 'APCP_2020113000.pdf'
#         """
# 
#         # extraer kwargs
#         figsize = kwargs.get('figsize', (16, 7))
#         color = kwargs.get('color', 'w')
#         dateformat = kwargs.get('dateformat', '%d-%m-%Y %H:%M')
#         extent = kwargs.get('extent', None)
#         rutaExport = kwargs.get('rutaExport', '../output/')
# 
#         # definir colores y umbrales
#         if simbologia is not None:
#             colors = simbologia[['Red', 'Green', 'Blue', 'alpha']].values / 255
#             cmap = ListedColormap(colors)
#             if self.units == '°C':
#                 ini = -30
#             elif self.units == 'mm':
#                 ini = 0
#             thresholds = np.insert(simbologia.Dato.values, 0, ini)
#             norm = BoundaryNorm(thresholds, cmap.N, clip=True)
# 
#         # extraer datos de HARMONIE
#         data, X, Y = self.data, self.X, self.Y
#         times = self.times
#         freq = np.diff(times).mean().total_seconds() / 3600
# 
#         # centroides de las cuencas
#         cuencas['center'] = cuencas.centroid
#         centroids = cuencas.copy()
#         centroids.set_geometry('center', inplace=True)
# 
#         # definir figura
#         fig, axes = plt.subplots(nrows=2, figsize=figsize)
# 
#         # generar mapas
#         for t, (ax, fecha_ini) in enumerate(zip(axes, times)):
# 
#             fecha_fin = fecha_ini + timedelta(hours=freq)
# 
#             # título y encabezado de la figura
#             if t == 0:
#                 if fecha_ini.month in [11, 12, 1, 2, 3]:
#                         UTC = 'UTC+01'
#                 else:
#                     UTC = 'UTC+02'
#                 ax.text(.5, 1.2, 'Pronóstico de {0} del modelo HARMONIE-Arome de la AEMET'.format(self.variable),
#                          weight='bold', horizontalalignment='center', transform=ax.transAxes)
#                 ax.text(.5, 1.1, 'Predicción del {0} ({1})'.format(times[0].strftime('%d-%m-%Y %Hh'), UTC),
#                         horizontalalignment='center', transform=ax.transAxes)
# 
#             # título del gráfico
#             ax.text(.5, .95, '{0} al {1}'.format(fecha_ini.strftime(dateformat),
#                                                    fecha_fin.strftime(dateformat)),
#                     horizontalalignment='center', weight='bold', fontsize=11, transform=ax.transAxes)
# 
#             # capas vectoriales
#             if provincias is not None:
#                 provincias.plot(color='lightgray', edgecolor='lightgray', ax=ax, zorder=1)
#             cuencas.boundary.plot(color=color, lw=1., ax=ax, zorder=4)
# 
#             # mapa HARMONIE
#             im = ax.imshow(data[t,::-1], cmap=cmap, norm=norm, extent=[X.min(), X.max(), Y.min(), Y.max()], zorder=2)
# 
#             # precipitación areal
#             for x, y, label in zip(centroids.geometry.x, centroids.geometry.y, cuencas['d{0}'.format(t+1)]):
#                 if label >= 50:
#                     c, fg = 'orange', 'k'
#                 elif label >= 90:
#                     c, fg = 'yellow', 'k'
#                 else:
#                     c, fg = 'k' , 'w'
#                 ax.text(x, y, '{0:.1f}'.format(label), horizontalalignment='center', fontsize=10,
#                         color=c, path_effects=[pe.withStroke(linewidth=2, foreground=fg)],# weight='bold',
#                         zorder=6)#, weight='bold')
# 
#             ax.set_aspect('equal')
#             if extent is not None:
#                 ax.set(xlim=(extent[0], extent[1]), ylim=(extent[2], extent[3]))
#             ax.axis('off');
# 
#         # barra de la leyenda
#         cax = fig.add_axes([0.26, 0.085, 0.5, 0.015])
#         cb = fig.colorbar(im, ticks=thresholds, orientation='horizontal', cax=cax)
#         cb.ax.set_xticklabels([int(x) if int(x) == x else x for x in thresholds])
#         cb.ax.tick_params(size=0)
#         cb.set_label('{0} ({1})'.format(self.variable, self.units))
# 
#         # exportar mapas en formato PDF
#         if export:
#             pdfFile = rutaExport + '{0}_{1}.pdf'.format(self.label, times[0].strftime('%Y%m%d%H'))
#             print('Exportando archivo {0}'.format(pdfFile))
#             plt.savefig(pdfFile, dpi=600, bbox_inches='tight')
#         
#     
#     def video(self, cuencas, provincias=None, simbologia=None, fps=2, dpi=100, export=False, **kwargs):
#         """Crea un vídeo con la predicción de HARMONIE.
#         
#         Entradas:
#         ---------
#         self:       class HARMONIE
#         cuencas:    geodataframe. Polígonos de las cuencas
#         provincias: geodataframe. Polígonos de las provincias
#         simbologia: dataframe. Tabla con la simbología a utilizar en el mapa. Debe contener las columnas 'Dato', 'Red', 'Green', 'Blue' y 'alpha' correspondientes a los umbrales de cada color, la combinación de colores RGP y la transparencia
#         fps:        int. 'frames per seconds'. Número de pasos temporales a mostrar por segundo
#         dpi:        int. 'dots per inch'. Resolución del mapa
#         export:     boolean. Si se quiere exportar la figura como pdf
#         
#         kwargs:
#                     figsize:    list (,). Tamaño de la figura. Por defecto (16, 7)
#                     color:      string. Color con el que definir los límites de las provincias. Por defecto es blanco
#                     dateformat: string. Formato de las fechas en la figura. Por defecto '%d-%m-%Y %H:%M'
#                     extent:     tuple [O, E, S, N]. Límites espaciales de la figura. Por defecto 'None'
#                     rutaExport: string. Carpeta donde guardar el pdf. Por defecto '../output/'
#         
#         Salidas:
#         --------
#         Mapas diarios
#         Si 'export == True', se exporta un pdf con el nombre 'label_pasada.mp4'. Por ejemplo, 'APCP_2020113000.mp4'
#         """
#         
#         # extraer datos de HARMONIE
#         data, X, Y, times = self.data, self.X, self.Y, self.times
#         data = data[:,::-1,:]
# 
#         # extraer kwargs
#         figsize = kwargs.get('figsize', (16, 3.75))#(17, 3.5)
#         color = kwargs.get('color', 'w')
#         dateformat = kwargs.get('dateformat', '%d-%m-%Y %H:%M')
#         extent = kwargs.get('extent', None)
#         rutaExport = kwargs.get('rutaExport', '../output/')
# 
#         # definir colores y umbrales
#         if simbologia is not None:
#             colors = simbologia[['Red', 'Green', 'Blue', 'alpha']].values / 255
#             cmap = ListedColormap(colors)
#             if self.units == '°C':
#                 ini = -30
#             elif self.units == 'mm':
#                 ini = 0
#             thresholds = np.insert(simbologia.Dato.values, 0, ini)
#             norm = BoundaryNorm(thresholds, cmap.N, clip=True)
# 
#         # definir configuración del gráfico en blanco
#         fig, ax = plt.subplots(figsize=figsize)
# 
#         # título y encabezado de la figura
#         mm = times[0].month
#         if int(mm) in [11, 12, 1, 2, 3]:
#                 UTC = 'UTC+01'
#         else:
#             UTC = 'UTC+02'
#         ax.text(.5, 1.1, 'Pronóstico de {0} del modelo HARMONIE-Arome de la AEMET'.format(self.variable),
#                  weight='bold', horizontalalignment='center', transform=ax.transAxes)
#         ax.text(.5, 1.025, 'Predicción del {0} ({1})'.format(times[0].strftime('%d-%m-%Y %Hh'), UTC), horizontalalignment='center', transform=ax.transAxes)
# 
#         # título del gráfico
#         title = ax.text(.5, 0.9, '', horizontalalignment='center', weight='bold', fontsize=11, transform=ax.transAxes)
#         ax.text(.5, -.16, '{0} ({1})'.format(self.variable, self.units),
#                 horizontalalignment='center', fontsize=11, transform=ax.transAxes)
# 
#         # capas vectoriales
#         if provincias is not None:
#             provincias.plot(color='lightgray', edgecolor='lightgray', ax=ax, zorder=1)
#         cuencas.boundary.plot(color='w', lw=1., ax=ax, zorder=3)
# 
#         # mapa HARMONIE
#         im = ax.imshow(np.zeros(data.shape[1:]), cmap=cmap, norm=norm,
#                        extent=[X.min(), X.max(), Y.min(), Y.max()], zorder=2)
# 
#         ax.set_aspect('equal')
#         ax.set(xlim=(extent[0], extent[1]), ylim=(extent[2], extent[3]))
#         ax.axis('off');
# 
#         # barra de la leyenda
#         cax = fig.add_axes([.3, 0.09, 0.4, 0.025])
#         cb = fig.colorbar(im, ticks=thresholds, orientation='horizontal', cax=cax)
#         cb.ax.set_xticklabels([int(x) if int(x) == x else x for x in thresholds])
#         cb.ax.tick_params(size=0)
#         #cb.set_label('{0} ({1})'.format(variables[var]['variable'], variables[var]['units']))
# 
#         def updatefig(i, *args):
#             """Función que define los zdatos a mostrar  el título en cada iteración"""
#             title.set_text(times[i].strftime(dateformat))
#             im.set_array(data[i,:,:])
#             return im,
# 
#         # genera la animación iterando sobre 'updatefig' un número 'frames' de veces
#         ani = animation.FuncAnimation(fig, updatefig, frames=data.shape[0], interval=1000/fps,
#                                       blit=True)
#         # guardar vídeo
#         if export:
#             mp4File = rutaExport + '{0}_{1}.mp4'.format(self.label, times[0].strftime('%Y%m%d%H'))
#             print('Exportando archivo {0}'.format(mp4File))
#             ani.save(mp4File, fps=fps, extra_args=['-vcodec', 'libx264'], dpi=dpi)
# 
#         # ver vídeo en el 'notebook'
#         return HTML(ani.to_html5_video())
# ```

# ## Funciones
# ### MODIS_extract

# In[157]:


def MODIS_extract(path, product, var, tiles, factor=None, dateslim=None, extent=None, verbose=True):
    """Extrae los datos de MODIS para un producto, variable y fechas dadas, transforma las coordenadas y recorta a la zona de estudio.
    
    Entradas:
    ---------
    path:       string. Ruta donde se encuentran los datos de MODIS (ha de haber una subcarpeta para cada producto)
    product:    string. Nombre del producto MODIS, p.ej. MOD16A2
    var:        string. Variable de interés dentro de los archivos 'hdf'
    factor:     float. Factor con el que multiplicar los datos para obtener su valor real (comprobar en la página de MODIS para el producto y variable de interés)
    tiles:      list. Hojas del producto MODIS a tratar
    dateslim:   list. Fechas de inicio y fin del periodo de estudio en formato YYYY-MM-DD. Si es 'None', se extraen los datos para todas las fechas disponibles
    clip:       string. Ruta y nombre del archivo ASCII que se utilizará como máscara para recortar los datos. Si es 'None', se extraen todos los datos
    coordsClip: pyproj.CRS. Proj del sistema de coordenadas al que se quieren transformar los datos. Si en 'None', se mantiene el sistema de coordenadas sinusoidal de MODIS
    verbose:    boolean. Si se quiere mostrar en pantalla el desarrollo de la función
    
    Salidas:
    --------
    Como métodos:
        data:    array (Y, X) o (dates, Y, X). Mapas de la variable de interés. 3D si hay más de un archivo (más de una fecha)
        Xcoords: array (2D). Coordenadas X de cada celda de los mapas de 'data'
        Ycoords: array (2D). Coordenadas Y de cada celda de los mapas de 'data'
        dates:   list. Fechas a las que corresponde cada uno de los maapas de 'data'
    """    
    
    if os.path.exists(path + product + '/') is False:
        os.makedirs(path + product + '/')
    os.chdir(path + product + '/')
    
    # SELECCIÓN DE ARCHIVOS
    # ---------------------
    if dateslim is not None:
        # convertir fechas límite en datetime.date
        start = datetime.strptime(dateslim[0], '%Y-%m-%d').date()
        end = datetime.strptime(dateslim[1], '%Y-%m-%d').date()
    
    dates, files = {tile: [] for tile in tiles}, {tile: [] for tile in tiles}
    for tile in tiles:
        # seleccionar archivos del producto para las hojas y fechas indicadas
        for file in [f for f in os.listdir() if (product in f) & (tile in f)]:
            year = file.split('.')[1][1:5]
            doy = file.split('.')[1][5:]
            date = datetime.strptime(' '.join([year, doy]), '%Y %j').date()
            if dateslim is not None:
                if (date>= start) & (date <= end):
                    dates[tile].append(date)
                    files[tile].append(file)
            else:
                dates[tile].append(date)
                files[tile].append(file)
    # comprobar que el número de archivos es igual en todas las hojas
    if len(set([len(dates[tile]) for tile in tiles])) > 1:
        print('¡ERROR! Diferente número de fechas en las diferentes hojas')
        MODIS_extract.files = files
        MODIS_extract.dates = dates
        return 
    else:
        dates = np.sort(np.unique(np.array([date for tile in tiles for date in dates[tile]])))
        if verbose:
            print('Seleccionar archivos')
            print('nº de archivos (fechas): {0:>3}'.format(len(dates)), end='\n\n')

    # ATRIBUTOS MODIS
    # ---------------
    if verbose:
        print('Generar atributos globales')
    # extraer atributos para cada hoja
    attributes = pd.DataFrame(index=tiles, columns=['ncols', 'nrows', 'Xo', 'Yf', 'Xf', 'Yo'])
    for tile in tiles:
        attributes.loc[tile,:] = hdfAttrs(files[tile][0])

    # extensión total
    Xo = np.min(attributes.Xo)
    Yf = np.max(attributes.Yf)
    Xf = np.max(attributes.Xf)
    Yo = np.min(attributes.Yo)
    # nº total de columnas y filas
    colsize = np.mean((attributes.Xf - attributes.Xo) / attributes.ncols)
    ncols = int(round((Xf - Xo) / colsize, 0))
    rowsize = np.mean((attributes.Yf - attributes.Yo) / attributes.nrows)
    nrows = int(round((Yf - Yo) / rowsize, 0))
    if verbose == True:
        print('dimensión:\t\t({0:}, {1:})'.format(ncols, nrows))
        print('esquina inf. izqda.:\t({0:>10.2f}, {1:>10.2f})'.format(Xo, Yo))
        print('esquina sup. dcha.:\t({0:>10.2f}, {1:>10.2f})'.format(Xf, Yf), end='\n\n')

    # coordenadas x de las celdas
    Xmodis = np.linspace(Xo, Xf, ncols)
    # coordenadas y de las celdas
    Ymodis = np.linspace(Yf, Yo, nrows)
        
    # CREAR MÁSCARAS
    # --------------
    if extent is not None:
        if verbose == True:
            print('Crear máscaras')
            
        # crear máscara según la extensión
        left, right, bottom, top =  extent
        maskCols = (Xmodis >= left) & (Xmodis <= right)
        maskRows = (Ymodis >= bottom) & (Ymodis <= top)
        
        # recortar coordenadas
        Xmodis = Xmodis[maskCols]
        Ymodis = Ymodis[maskRows]
        
        if verbose == True:
            print('dimensión:\t\t({0:>4}, {1:>4})'.format(len(Ymodis), len(Xmodis)))
            print('esquina inf. izqda.:\t({0:>10.2f}, {1:>10.2f})'.format(Xmodis.min(), Ymodis.min()))
        print('esquina sup. dcha.:\t({0:>10.2f}, {1:>10.2f})'.format(Xmodis.max(), Ymodis.max()),
              end='\n\n')

    # IMPORTAR DATOS
    # --------------
    if verbose:
        print('Importar datos')
        
    for d, date in enumerate(dates):
        dateStr = str(date.year) + str(date.timetuple().tm_yday).zfill(3)

        for t, tile in enumerate(tiles):
            print('Fecha {0:>2} de {1:>2}: {2}\t||\tTile {3:>2} de {4:>2}: {5}'.format(d + 1, len(dates), date,
                                                                                       t + 1, len(tiles), tile), end='\r')
            
            # localización de la hoja dentro del total de hojas
            nc, nr, xo, yf, xf, yo = attributes.loc[tile, :]
            i = int(round((Yf - yf) / (rowsize * attributes.nrows[t]), 0))
            j = int(round((Xf - xf) / (colsize * attributes.ncols[t]), 0))

            # archivo de la fecha y hoja dada
            file = [f for f in files[tile] if dateStr in f][0]
            # cargar archivo 'hdf'
            hdf = Dataset(file, format='hdf4')
            # extraer datos de la variable
            tmp = hdf[var][:]
            tmp = tmp.astype(float)
            tmp[tmp.mask] = np.nan
            hdf.close()
            # guardar datos en un array global de la fecha
            if t == 0:
                dataD = tmp.copy()
            else:
                if (i == 1) & (j == 0):
                    dataD = np.concatenate((dataD, tmp), axis=0)
                elif (i == 0) & (j == 1):
                    dataD = np.concatenate((dataD, tmp), axis=1)
            del tmp
        
        # recortar array de la fecha con la máscara
        if extent is not None:
            dataD = dataD[maskRows, :][:, maskCols]
            
        # guardar datos en un array total
        if d == 0:
            data = dataD.copy()
        else:
            data = np.dstack((data, dataD))
        del dataD
    print()
    
    # multiplicar por el factor de escala (si existe)
    if factor is not None:
        data *= factor
        
    # reordenar el array (tiempo, Y, X)
    if len(data.shape) == 3:
        tmp = np.ones((data.shape[2], data.shape[0], data.shape[1])) * np.nan
        for t in range(data.shape[2]):
            tmp[t,:,:] = data[:,:,t]
        data = tmp.copy()
        del tmp

    # GUARDAR RESULTADOS
    # ------------------
    modis = raster3D(data, Xmodis, Ymodis, dates, crs=sinusoidal)
    
    return modis


# ### video

# In[1]:


def video(raster, cuenca, cmap, norm, DEM=None, fps=2, dpi=600, export=False, **kwargs):
    """Crea un vídeo con la predicción de HARMONIE.

    Entradas:
    ---------
    raster:     class raster3D
    cuenca:     geodataframe. Polígonos de las cuencas
    cmap:       
    norm:
    DEM:        class raster2D. Mapa de elevación
    fps:        int. 'frames per seconds'. Número de pasos temporales a mostrar por segundo
    dpi:        int. 'dots per inch'. Resolución del mapa
    export:     boolean. Si se quiere exportar la figura como mp4

    kwargs:
                figsize:    list (,). Tamaño de la figura. Por defecto (16, 7)
                color:      string. Color con el que definir los límites de las provincias. Por defecto es blanco
                dateformat: string. Formato de las fechas en la figura. Por defecto '%d-%m-%Y %H:%M'
                extent:     tuple [O, E, S, N]. Límites espaciales de la figura. Por defecto 'None'
                rutaExport: string. Carpeta donde guardar el pdf. Por defecto '../output/'

    Salidas:
    --------
    Mapas diarios
    Si 'export == True', se exporta un pdf con el nombre 'label_pasada.mp4'. Por ejemplo, 'APCP_2020113000.mp4'
    """

    # extraer datos del raster
    data, X, Y, times = raster.data, raster.X, raster.Y, raster.times

    # extraer kwargs
    figsize = kwargs.get('figsize', (16, 3.75))#(17, 3.5)
    color = kwargs.get('color', 'k')
    dateformat = kwargs.get('dateformat', '%d-%m-%Y')
    extent = kwargs.get('extent', raster.extent())
    rutaExport = kwargs.get('rutaExport', '../output/')

    # definir configuración del gráfico en blanco
    fig, ax = plt.subplots(figsize=figsize)

    # título del gráfico
    title = ax.text(.5, 1.05, '', horizontalalignment='center', fontsize=13,
                    transform=ax.transAxes)

    # capas vectoriales
    cuenca.boundary.plot(color=color, lw=1., ax=ax, zorder=3)

    # dem de fondo
    ax.imshow(DEM.data, extent=DEM.extent, cmap='pink', zorder=1)

    # mapa
    im = ax.imshow(np.zeros(data.shape[1:]), cmap=cmap, norm=norm, extent=raster.extent(),
                   zorder=2)

    ax.set_aspect('equal')
    ax.set(xlim=(extent[0], extent[1]), ylim=(extent[2], extent[3]))
    ax.axis('off');

    # barra de la leyenda
    snow_patch = mpatches.Patch(color=cmap.colors[1], label='snow')
    fig.legend(handles=[snow_patch], loc=8, fontsize=13)

    def updatefig(i, *args):
        """Función que define los zdatos a mostrar  el título en cada iteración"""
        title.set_text(times[i].strftime(dateformat))
        im.set_array(data[i,:,:])
        return im,

    # genera la animación iterando sobre 'updatefig' un número 'frames' de veces
    ani = animation.FuncAnimation(fig, updatefig, frames=data.shape[0], interval=1000/fps,
                                  blit=True)
    # guardar vídeo
    if export:
        mp4File = rutaExport + '{0}_{1}.mp4'.format(raster.label, times[0].strftime('%Y%m%d%H'))
        print('Exportando archivo {0}'.format(mp4File))
        ani.save(mp4File, fps=fps, extra_args=['-vcodec', 'libx264'], dpi=dpi)

    # ver vídeo en el 'notebook'
    return HTML(ani.to_html5_video())


# ### video2

# In[116]:


def video2(raster1, raster2, cuenca, cmap, norm, DEM=None, fps=2, dpi=600, export=None, **kwargs):
    """Crea un vídeo con la predicción de HARMONIE.

    Entradas:
    ---------
    raster1:    class raster3D
    raster2:    class raster3D
    cuenca:     geodataframe. Polígonos de las cuencas
    cmap:
    norm:
    DEM:        class raster2D. Mapa de elevación
    fps:        int. 'frames per seconds'. Número de pasos temporales a mostrar por segundo
    dpi:        int. 'dots per inch'. Resolución del mapa
    export:     boolean. Si se quiere exportar la figura como pdf

    kwargs:
                figsize:    list (,). Tamaño de la figura. Por defecto (16, 7)
                color:      string. Color con el que definir los límites de las provincias. Por defecto es blanco
                dateformat: string. Formato de las fechas en la figura. Por defecto '%d-%m-%Y %H:%M'
                extent:     tuple [O, E, S, N]. Límites espaciales de la figura. Por defecto 'None'
                rutaExport: string. Carpeta donde guardar el pdf. Por defecto '../output/'

    Salidas:
    --------
    Mapas diarios
    Si 'export == True', se exporta un pdf con el nombre 'label_pasada.mp4'. Por ejemplo, 'APCP_2020113000.mp4'
    """

    # extraer datos del raster
    data1, X, Y, times = raster1.data, raster1.X, raster1.Y, raster1.times
    data2 = raster2.data

    # extraer kwargs
    figsize = kwargs.get('figsize', (16, 3.75))#(17, 3.5)
    color = kwargs.get('color', 'k')
    dateformat = kwargs.get('dateformat', '%d-%m-%Y')
    extent = kwargs.get('extent', raster1.extent())
    rutaExport = kwargs.get('rutaExport', '../output/')
    labels = kwargs.get('labels', ['', ''])

    # definir configuración del gráfico en blanco
    fig, ax = plt.subplots(ncols=2, figsize=figsize, sharex=True, sharey=True)

    # título del gráfico
    title = ax[0].text(1.05, 1.0, '', horizontalalignment='center', fontsize=13, transform=ax[0].transAxes)

    # capas vectoriales
    ax[0].text(.5, 1.05, labels[0], horizontalalignment='center', fontsize=14, transform=ax[0].transAxes)
    cuenca.boundary.plot(color=color, lw=1., ax=ax[0], zorder=3)
    # dem de fondo
    ax[0].imshow(DEM.data, extent=DEM.extent, cmap='pink', zorder=1)
    # mapa
    im1 = ax[0].imshow(np.zeros(data1.shape[1:]), cmap=cmap, norm=norm, extent=raster1.extent(),
                       zorder=2)
    ax[0].set_aspect('equal')
    ax[0].set(xlim=(extent[0], extent[1]), ylim=(extent[2], extent[3]))
    ax[0].axis('off');

    # capas vectoriales
    ax[1].text(.5, 1.05, labels[1], horizontalalignment='center', fontsize=14, transform=ax[1].transAxes)
    cuenca.boundary.plot(color=color, lw=1., ax=ax[1], zorder=3)
    # dem de fondo
    ax[1].imshow(DEM.data, extent=DEM.extent, cmap='pink', zorder=1)
    # mapa
    im2 = ax[1].imshow(np.zeros(data2.shape[1:]), cmap=cmap, norm=norm, extent=raster2.extent(),
                       zorder=2)
    ax[1].set_aspect('equal')
    ax[1].set(xlim=(extent[0], extent[1]), ylim=(extent[2], extent[3]))
    ax[1].axis('off');

    # barra de la leyenda
    snow_patch = mpatches.Patch(color=cmap.colors[1], label='snow')
    fig.legend(handles=[snow_patch], loc=8, fontsize=13)

    def updatefig(i, *args):
        """Función que define los zdatos a mostrar  el título en cada iteración"""
        title.set_text(times[i].strftime(dateformat))
        im1.set_array(data1[i,:,:])
        im2.set_array(data2[i,:,:])
        return im1, im2

    # genera la animación iterando sobre 'updatefig' un número 'frames' de veces
    ani = animation.FuncAnimation(fig, updatefig, frames=data1.shape[0], interval=1000/fps,
                                  blit=True)
    # guardar vídeo
    if export is not None:
        if export.endswith('mp4'):
            print('Exportando archivo {0}'.format(export))
            ani.save(export, fps=fps, extra_args=['-vcodec', 'libx264'], dpi=dpi)
        else:
            print('Formato de archivo incorrecto; ha de ser mp4.')

    # ver vídeo en el 'notebook'
    return HTML(ani.to_html5_video())


# In[ ]:




