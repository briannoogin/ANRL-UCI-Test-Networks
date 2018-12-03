import numpy as np
import os
import pandas as pd

def load_data(path):
    file = open(path,'r')
    data = []
    for line in file:
        data.append(line.split())
    file.close()
    labels = []
    # take away the headers
    data = data[1:len(data)] 
    # convert from string to float
    data = [[float(dataPoint) for dataPoint in row] for row in data]
     # extract the classes and remove the classes column
    for index in range(len(data)):
        labels.append(data[index][-1])
        del data[index][-1]
    return [data,labels]
def deleteZeros(path):
     f = pd.read_table(path, header=None, delim_whitespace=True)
     f= f[f[23] != 0]
     np.savetxt(r'mHealth.log', f.values, fmt='%d')
     print(f[23].value_counts())
     return f
load_data('mHealth.log')