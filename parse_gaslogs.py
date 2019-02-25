import numpy as np
import pandas as pd
import sys
import os
import shutil

# counts all classes
def countClasses(path):
     f = pd.read_table(path, header=None, delim_whitespace=True)
     # np.savetxt(path, f.values, fmt='%s')
     #dataframe.sort_index(by='count', ascending=[True])
     print(f[0].value_counts().sort_index(ascending=[True]).tolist())
     return f

def parseFilesAndCount():
    for i in range(1,11):
        with open("parsed_gas/batch" + str(i) + ".dat", mode="w") as out:
            with open("Gas_sensor_dataset/dataset/batch" + str(i) + ".dat") as log:
                for line in log:
                    log_samples = line.split()
                    out.write("".join(num.ljust(18) for num in log_samples))
                    out.write("\n");

        print("Batch " + str(i))
        countClasses("Gas_sensor_dataset/dataset/batch" + str(i) + ".dat")

def combine_data(path):
    filenames = os.listdir(path)
    with open('./combined_batches.dat','wb') as wfd:
        for f in filenames:
            with open(path + f,'rb') as fd:
                shutil.copyfileobj(fd, wfd)

combine_data("parsed_gas/")
countClasses('./combined_batches.dat')
