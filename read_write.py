
# Funciones de lectura y escritura de diferentes tipos de archivo en Python

import numpy as np
import pandas as pd
from math import floor, ceil
import os
from pyproj import Proj


def read_ascii(filename, datatype='float'):
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
        attributes = [ncols, nrows, xllcorner, yllcorner, cellsize, NODATA_value]
        # extract data
        data = np.zeros((nrows, ncols))
        for i in range(nrows):
            data[i, :] = asc[i + 6].split()
        data[data == NODATA_value] = np.nan
        data = np.ma.masked_invalid(data)
        data = data.astype(datatype)
    file.close()

    read_ascii.attributes = attributes
    read_ascii.data = data

#     return read_ascii


def write_ascii(filename, data, attributes, format='%.0f ', epsg=None):
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

    aux = data.copy()

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
    if epsg != None:
        import requests
        # access projection information
        wkt = requests.get("http://spatialreference.org/ref/epsg/{0}/prettywkt/".format(epsg)).text
        # remove spaces between charachters
        wkt = wkt.replace(" ", "")
        # place all the text on one line
        wkt = wkt.replace('\n', '')
        # create the .prj file
        with open(filename[:-4] + '.prj', 'w') as projfile:
            projfile.write(wkt)
        projfile.close()


def correct_ascii(file, format='%.0f '):
    """Elimina filas y/o columnas del ascii con todo NaN

    Parámetros:
    -----------
    file:      string. Ruta, nombre y extensión del archivo ascii
    format:    string. Formato en el que exportar los valores del ascii

    Salida:
    -------
    Sobreescribe el archivo ascii con la correción"""


    # importar archivo
    read_ascii(file)
    data = read_ascii.data
    atr = read_ascii.attributes
    x = np.arange(atr[2], atr[2] + atr[0] * atr[4], atr[4])
    y = np.arange(atr[3], atr[3] + atr[1] * atr[4], atr[4])[::-1]
    # eliminar filas y columnas vacías
    maskRow = np.isnan(data).all(axis=1)
    maskCol = np.isnan(data).all(axis=0)
    data = data[~maskRow, :]
    data = data[:, ~maskCol]
    x = x[~maskCol]
    y = y[~maskRow]
    # corregir atributos
    atr[0], atr[1] = data.shape[1], data.shape[0]
    atr[2], atr[3] = x[0], y[-1]
    # exportar ascii corregido
    write_ascii(file, data, atr, format=format)

    # guardar corrección
    correct_ascii.data = data
    correct_ascii.attributes = atr
