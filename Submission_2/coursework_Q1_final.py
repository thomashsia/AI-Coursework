
"""
@author: Boyang Xia
"""
import pandas as pd

""" This script counts 
    the number of distinct IP addresses 
    and Classification labels.
"""
#%%
"Read the File:"
data = pd.read_csv('coursework1.csv')
# data = pd.read_csv('coursework2.csv')

"Convert the File to List:"
sourceip = data['sourceIP'].tolist()
destip = data['destIP'].tolist()
classfication = data['classification'].tolist()

#%%
"Counting Function"
def count(entry):
    store = []
    S = range(0, len(entry))
    
    for s in S:
        store.append(entry[s])
    entry_list = set(store)
    entry_count = len(entry_list)
    
    return entry_count

#%%

print('SourceIP: ', count(sourceip))
print('DestIP: ', count(destip))
print('Classification: ', count(classfication))