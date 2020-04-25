
"""
@author: Boyang Xia
"""
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

""" This script plot the bar charts:
    Number of Occurrence of EACH IP addresses.
"""
#%%

"Read the file: "
data = pd.read_csv('coursework1.csv')

"Data Pre-processing: "
sourceip = data['sourceIP'].tolist()
destip = data['destIP'].tolist()
classification = data['classification'].tolist()

"Convert the csv into list and re-order the sequence"
s = pd.Series(sourceip).value_counts().reset_index().values.tolist()
d = pd.Series(destip).value_counts().reset_index().values.tolist()
c = pd.Series(classification).value_counts().reset_index().values.tolist()

#%%

def genehist(entry):
    "genehist(d) OR genehist(s) OR genehist(c)"
    "Generate Histogram of Each IP's Number of Occurrence"
    length = range(len(entry))
    
    "Store IP addresses in store_x; Store number of occurrence in store_y"
    store_x, store_y = [], []

    for k in length:
        store_x.append(entry[k][0])
        store_y.append(entry[k][1])
    
    "Plot bar charts"
    plt.figure(figsize=(20,7))
    plt.bar(np.arange(len(store_x)), store_y, align='center', alpha=0.5)
    plt.xticks(np.arange(len(store_x)), tuple(store_x), rotation=90)

    plt.ylabel('Number of Occurrence')
    # plt.title('Histogram of DestIP')
    plt.show()
    
    return

#%%
genehist(s)
genehist(d)
# genehist(c)
