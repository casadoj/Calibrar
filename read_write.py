
# Funciones de lectura y escritura de diferentes tipos de archivo en Python

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('seaborn-whitegrid')
import pandas as pd
from math import floor, ceil
import os


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
        cellsize = int(asc[4].split()[1])
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
        
    return read_ascii


def write_ascii(filename, data, attributes, format='%.0f '):
    """Export a 2D numpy array and its corresponding attributes as an ascii raster.
    
    Parameters:
    -----------
    filename:     string. Name (including path and extension) of the ASCII file
	data:         narray. 2D array with the data to be exported
	attributes:   narray[6x1]. Array including the following information: ncols, nrows, xllcorner, yllcorner, cellsize, NODATA_value
    format:       string. Format in which the values in 'data' will be exported
    
	Output:
    -------
    An .asc raster file"""
    
    aux = data.copy()
    
    # unmask data if masked
    if np.ma.is_masked(aux):
        np.ma.set_fill_value(aux, attributes[5])
        aux = aux.filled()
    
    # convert NaN to NODATA_value
    aux[np.isnan(aux)] = attributes[5]
    
    # export ascii
    with open(filename, 'w+') as file:
        # write attributes
        file.write('ncols\t\t{0:<8}\n'.format(attributes[0]))
        file.write('nrows\t\t{0:<8}\n'.format(attributes[1]))
        file.write('xllcorner\t{0:<8}\n'.format(attributes[2]))
        file.write('yllcorner\t{0:<8}\n'.format(attributes[3]))
        file.write('cellsize\t{0:<8}\n'.format(attributes[4]))
        file.write('NODATA_value\t{0:<8}\n'.format(attributes[5]))
        # write data
        for i in range(aux.shape[0]):
            #values = df.iloc[i, 6:].tolist()
            values = aux[i, :].tolist()
            file.writelines([format % item  for item in values])
            file.write("\n")
            
    file.close()
