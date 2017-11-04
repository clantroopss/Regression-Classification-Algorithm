import sys
import re
import pandas as pd
import numpy as np
import config
import algorithm
from sklearn.utils import shuffle


def main():

    for name, data_set in config.datasets.items():
        if data_set['header_present']:
            ds = pd.read_csv(data_set['location'], sep=data_set['sep'])
        else:
            ds = pd.read_csv(data_set['location'], sep=data_set['sep'], header=None)

        for chunk in config.chunk_size:
            if chunk < len(ds.index):
                print "chunk size", chunk
                if name == 'skin_nonskin':
                    ds = shuffle(ds, n_samples=ds.shape[0], random_state=0)
                ds_chunk = ds.head(chunk)
                print "dataset", name
                algorithm.execute_algorithm(ds_chunk, name, chunk,"regression")
                algorithm.execute_algorithm(ds_chunk, name, chunk, "classification")


if __name__ == "__main__":
    main()
