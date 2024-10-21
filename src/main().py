import pandas as pd
import os
import pickle

import time

source_filename = 'movie_reviews'
file_extension = '.xlsx'
main_cache_filename = source_filename + ".pkl"
cache_folder = 'cache'

minimumWordLength, minimumWordOccurrence = 10, 20

# Create a cache folder if it does not exist
if not os.path.exists(cache_folder):
    os.makedirs(cache_folder)

def main():
    start_time = time.time() # start measuring time

# Task 1
    trainingData, trainingLabels, testData, testLabels = task1()

# bundling into a test and training data frame
    trainingDf = pd.DataFrame({'Review': trainingData, 'Sentiment': trainingLabels})
    testDf = pd.DataFrame({'Review': testData, 'Sentiment': testLabels})

# Task 2
    trainWordList, testWordList = task2(trainingDf, testDf, minimumWordLength, minimumWordOccurrence)

# Task 3

    task3(trainingDf, testDf, trainWordList, testWordList, minimumWordLength, minimumWordOccurrence)


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
    print(f"Number of unique words in training data, longer than {minimumWordLength} characters, which occur more than {minimumWordOccurence} times: " + str(len(train_wordList)))

    test_wordList = extract_features(testDf, minimumWordLength, minimumWordOccurence)
    print(f"Number of unique words in test data, longer than {minimumWordLength} characters, which occur more than {minimumWordOccurence} times: " + str(len(test_wordList)))

    return train_wordList, test_wordList

def task3(trainingDf, testDf, trainWordList, testWordList, minimumWordLength, minimumWordOccurence):
    printHeader("#", "Task 3")
    train_word_counts_df = get_word_counts_df(trainingDf, trainWordList, minimumWordLength, minimumWordOccurence, "train_word_counts_df")
    print(train_word_counts_df)
    test_word_counts_df = get_word_counts_df(testDf, testWordList, minimumWordLength, minimumWordOccurence, "test_word_counts_df")
    print(test_word_counts_df)


def get_word_counts_df(df, wordList, minimumWordLength, minimumWordOccurence, cache_name):
    words_cache_file =  source_filename + "-" + cache_name + "-" + str(minimumWordLength) + "-"+ str(minimumWordOccurence) + ".pkl"

    cache_filepath = os.path.join(cache_folder, words_cache_file)

    # this task is expensive, so I will cache the results, taking into account the minimum word length and minimum word occurrence may change
    if os.path.exists(cache_filepath):

        with open(cache_filepath, 'rb') as f:
            word_counts_df = pickle.load(f)
        print(f'Loaded data from cache: {cache_filepath}\n')

    else:

        word_counts = {word: 0 for word in wordList}

        positive_reviews = df[df['Sentiment'] == 'positive']
        total_reviews = positive_reviews.shape[0]

        start_time = time.time()
        for i, review in enumerate(positive_reviews['Review']):
            for word in wordList:
                if word in review:
                    word_counts[word] += 1
            progressbar(i, total_reviews, start_time)

        word_counts_df = pd.DataFrame(list(word_counts.items()), columns=['Word', 'Positive_Review_Count'])

        with open(cache_filepath, 'wb') as f:
            pickle.dump(word_counts_df, f)
        print(f'Saved data to cache: {cache_filepath}\n')

    return word_counts_df


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
    cache_path = os.path.join(cache_folder, main_cache_filename)
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            df = pickle.load(f)
        print('Loaded data from cache\n')
    else:
        df = pd.read_excel(source_filename + file_extension)
        with open(cache_path, 'wb') as f:
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

# # Function to display a progress bar so the user knows the program is still running and how far along it is
def progressbar(i, upper_range, start_time):
    # Calculate the percentage of completion
    percentage = (i / (upper_range - 1)) * 100
    # Calculate the number of '█' characters to display
    num_blocks = int(percentage)
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    # Estimate remaining time
    if percentage > 0:
        estimated_total_time = elapsed_time / (percentage / 100)
        remaining_time = estimated_total_time - elapsed_time
    else:
        remaining_time = 0
    # Create the progress bar string
    progress_string = f'\r{("█" * num_blocks)}{("_" * (100 - num_blocks))} {percentage:.2f}% Elapsed: {elapsed_time:.2f}s Remaining: {remaining_time:.2f}s'
    if i == upper_range - 1:
        print(progress_string)
    else:
        print(progress_string, end='', flush=True)

if __name__ == '__main__':
    main()
