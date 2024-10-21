import pandas as pd
import sklearn as sk
import sklearn.model_selection as skms
import matplotlib.pyplot as plt
import os
import pickle

import time


source_filename = 'movie_reviews.xlsx'
cache_filename = source_filename + ".pkl"
split_factor = 0.8

def main():
    start_time = time.time()
    df = read_data()

    print("Frame size", df.shape[0])
    train, test = split_data(df)


    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.2f} seconds")


# Read data from the Excel file.
# The file is large, and this process is slow
# since I will be running this a number of times, I decided to cache the data frame
def read_data():
    if os.path.exists(cache_filename):
        with open(cache_filename, 'rb') as f:
            df = pickle.load(f)
        print('Loaded data from cache')
    else:
        df = pd.read_excel(source_filename)
        with open(cache_filename, 'wb') as f:
            pickle.dump(df, f)
        print('Loaded data from Excel and cached it')
    print(df.head())
    return df

def split_data(df):
    train, test = skms.train_test_split(df, test_size=1-split_factor)
    print('Train size: ', train.shape[0])
    print('Test size: ', test.shape[0])
    return train, test

if __name__ == '__main__':
    main()