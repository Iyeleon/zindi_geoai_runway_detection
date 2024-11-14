import os
import random
import numpy as np
import tensorflow as tf

def set_seed(seed=42):
    'Set seed for reproducibility'
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Seed set to: {seed}")

def parse_test_files(path):
    '''Reorganizes all test files into folders if not alreay done'''
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
    '''Get ceiling values rounded to the nearest 10'''
    return np.ceil(x / 10) * 10

def floor10(x):
    '''Get floor values rounded to the nearest 10'''
    return np.floor(x / 10) * 10