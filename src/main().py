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
    start_time = time.time() # start measuring time

    trainingData, trainingLabels, testData, testLabels = task1()

    print(f"Execution time: {time.time() - start_time:.2f} seconds")     # calculate elapsed


    task2(trainingData, trainingLabels, testData, testLabels)

def task1():
    printHeader("#", "Task 1")

    df = read_data()
    print("Total row count", df.shape[0])
    trainingData, trainingLabels, testData, testLabels = split_data(df)

    #where positive
    print("Positive Labels in training data: " + str(trainingLabels[trainingLabels == 'positive'].shape[0]))
    print("Negative Labels in training data: " + str(trainingLabels[trainingLabels == 'negative'].shape[0]))
    print("Positive Labels in test data: " + str(testLabels[testLabels == 'positive'].shape[0]))
    print("Negative Labels in test data: " + str(testLabels[testLabels == 'negative'].shape[0]))

    return trainingData, trainingLabels, testData, testLabels

def task2(trainingData, trainingLabels, testData, testLabels):
    printHeader("#", "Task 2")

# Read data from the Excel file.
# The file is large, and this process is slow
# since I will be running this a number of times, I decided to cache the data frame
def read_data():
    if os.path.exists(cache_filename):
        with open(cache_filename, 'rb') as f:
            df = pickle.load(f)
        print('Loaded data from cache\n')
    else:
        df = pd.read_excel(source_filename)
        with open(cache_filename, 'wb') as f:
            pickle.dump(df, f)
        print('Loaded data from Excel and cached it')
    return df


    # this query creates a boolean mask for the rows that have the value 'train' or 'test' in the 'Split' column
    # the mask is then used to filter the rows that will be used for training
    # it returns the text column and the sentiment column as separate arrays
def split_data(df):
    trainingData = df[df['Split'] == 'train']['Review']
    trainingLabels = df[df['Split'] == 'train']['Sentiment']
    testData = df[df['Split'] == 'test']['Review']
    testLabels = df[df['Split'] == 'test']['Sentiment']
    return trainingData, trainingLabels, testData, testLabels

def printHeader(char, text):
    print('\n'+ char * 80)
    print(text.center(80, ' '))
    print(char * 80 + '\n')


if __name__ == '__main__':
    main()