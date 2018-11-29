import numpy as np
import os
import pandas as pd

def load_data(path, omit=False):
    f = pd.read_table(path, header=None, delim_whitespace=True)
    train, test = [], []
    
    num_data = len(f)
    for x in range(0, num_data):
        # skip data that has -200
        skip_flag = False
        
        # all pieces of data are used besides str name, and origin (last 2 indices)
        row_of_data = [f.loc[x][i] for i in range(1,15)]
        
        for j in row_of_data:
            if float(j) == -200:
                skip_flag = True 
        
        # if any piece of data is unknown, we don't add it to our dataset
        if skip_flag and omit:
            continue
        
        # append y_ to test, and x to train
        test.append(f.loc[x][0])
        train.append(row_of_data)

    return train, test
def deleteZeros(path):
     f = pd.read_table(path, header=None, delim_whitespace=True)
     f= f[f[23] != 0]
     return f
deleteZeros('mHealth_subject1.log')