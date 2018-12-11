import numpy as np
import os
import pandas as pd
import sys

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
    return np.asarray(data),np.asarray(labels)
def deleteZeros(path):
     f = pd.read_table(path, header=None, delim_whitespace=True)
     f= f[f[23] != 0]
     np.savetxt(r'mHealth.log', f.values, fmt='%d')
     #dataframe.sort_index(by='count', ascending=[True])
     print(f[23].value_counts().sort_index(ascending=[True]).tolist())
     return f
for i in range(1,11):
    print("Subject " + str(i))
    deleteZeros('MHEALTHDATASET/mHealth_subject' + str(i) + '.log')