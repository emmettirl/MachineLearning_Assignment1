import pandas as pd
import sklearn as sk
import sklearn.model_selection as skms
import matplotlib.pyplot as plt
import os
import pickle

import time

from pandas import value_counts

source_filename = 'movie_reviews.xlsx'
cache_filename = source_filename + ".pkl"

def main():
    start_time = time.time() # start measuring time

# Task 1
    trainingData, trainingLabels, testData, testLabels = task1()

# bundling into a test and training data frame
    trainingDf = pd.DataFrame({'Review': trainingData, 'Sentiment': trainingLabels})
    testDf = pd.DataFrame({'Review': testData, 'Sentiment': testLabels})

# Task 2
    minimumWordLength, minimumWordOccurence = 3, 15
    trainWordList, testWordList = task2(trainingDf, testDf, minimumWordLength, minimumWordOccurence)

# Task 3

    task3()


    print(f"Execution time: {time.time() - start_time:.2f} seconds")     # calculate elapsed


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

def task2(trainingDf, testDf, minimumWordLength, minimumWordOccurence):
    printHeader("#", "Task 2")

    train_wordList = extract_features(trainingDf, minimumWordLength, minimumWordOccurence)
    print(f"Number of unique words in training data, longer than {minimumWordLength} characters, which occure more than {minimumWordOccurence} times: " + str(len(train_wordList)))

    test_wordList = extract_features(testDf, minimumWordLength, minimumWordOccurence)
    print(f"Number of unique words in test data, longer than {minimumWordLength} characters, which occure more than {minimumWordOccurence} times: " + str(len(test_wordList)))

    return train_wordList, test_wordList

def task3():
    printHeader("#", "Task 3")

def extract_features(data, minimumWordLength, minimumWordOccurence):
    # replace all dashes with spaces
    data["Review"] = data["Review"].replace("-", ' ', regex=True)
    # remove all non-alphanumeric characters
    pattern = r'[^\w\s]'
    data["Review"] = data["Review"].replace(pattern, '', regex=True)
    # convert to lowercase
    data["Review"] = data["Review"].str.lower()
    # split the words
    data["Review"] = data["Review"].str.split()
    # explode the words into separate rows
    word_data = pd.DataFrame({"words": data["Review"].explode()})

    word_data = word_data[word_data["words"].str.len() >= minimumWordLength]
    word_vc = word_data['words'].value_counts().reset_index()
    word_vc.columns = ['Word', 'Count']
    word_vc = word_vc[word_vc["Count"] >= minimumWordOccurence]

    wordlist = word_vc['Word'].tolist()
    return wordlist


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