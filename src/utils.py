import os
import numpy as np

def parse_test_files(path):
    for dirs, subdirs, files in os.walk(path):
        if dirs == path and files:
            unique_files = set([i.split('.')[0] for i in files])
            for basename in unique_files:
                temp = [i for i in files if basename in i]
                base = os.path.join(path, basename)
                os.makedirs(base, exist_ok = True) ;
                for i in temp: os.rename(os.path.join(path,i), os.path.join(base, i))
        elif subdirs:
            print('Files are already partitioned')
            print(os.listdir(dirs))
            break
        else:
            print('There are no files to partition')
            break

def ceil10(x):
    return np.ceil(x / 10) * 10

def floor10(x):
    return np.floor(x / 10) * 10