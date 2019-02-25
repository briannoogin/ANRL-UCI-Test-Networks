import numpy as np
import pandas as pd
import sys

#ARGS:
#1: INPUT FILE
#2: FORMATTED OUTPUT FILE

for i in range(1,11):
    with open("parsed_gas/batch" + str(i) + ".dat", mode="w") as out:
        with open("Gas_sensor_dataset/dataset/batch" + str(i) + ".dat") as log:
            for line in log:
                log_samples = line.split()
                out.write("".join(num.ljust(18) for num in log_samples))
                out.write("\n");

    # deletes all examples with 0 as label
    def countClasses(path):
         f = pd.read_table(path, header=None, delim_whitespace=True)
         # np.savetxt(path, f.values, fmt='%s')
         #dataframe.sort_index(by='count', ascending=[True])
         print(f[0].value_counts().sort_index(ascending=[True]).tolist())
         return f

    print("Batch " + str(i))
    countClasses("Gas_sensor_dataset/dataset/batch" + str(i) + ".dat")
