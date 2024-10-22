import math
import pandas as pd
import os
import pickle
import time
import sklearn.model_selection

source_filename = 'movie_reviews'
file_extension = '.xlsx'
main_cache_filename = source_filename + ".pkl"

cache_folder = 'cache'

minimumWordLength, minimumWordOccurrence = 12, 50


########################################################################################################################
# Task 1
########################################################################################################################
def task1(df):
    print_header("#", "Task 1")

    print_divider("-")

    print("Total row count", df.shape[0])
    print_divider("-")
    training_data, training_labels, test_data, test_labels = split_data(df)

    # where positive
    print("Positive Labels in training data: " + str(training_labels[training_labels == 'positive'].shape[0]))
    print("Negative Labels in training data: " + str(training_labels[training_labels == 'negative'].shape[0]))
    print_divider("-")
    print("Positive Labels in test data: " + str(test_labels[test_labels == 'positive'].shape[0]))
    print("Negative Labels in test data: " + str(test_labels[test_labels == 'negative'].shape[0]))

    training_df = pd.DataFrame({'Review': training_data, 'Sentiment': training_labels})
    test_df = pd.DataFrame({'Review': test_data, 'Sentiment': test_labels})

    return training_df, test_df


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

# data is being split based on the value in the 'Split' column, rather than a random split factor
def split_data(df):
    training_data = df[df['Split'] == 'train']['Review']
    training_labels = df[df['Split'] == 'train']['Sentiment']
    test_data = df[df['Split'] == 'test']['Review']
    test_labels = df[df['Split'] == 'test']['Sentiment']
    return training_data, training_labels, test_data, test_labels


########################################################################################################################
# Task 2
########################################################################################################################
def task2(training_df, test_df, minimum_word_length, minimum_word_occurrence):
    print_header("#", "Task 2")

    train_word_list = extract_features(training_df, minimum_word_length, minimum_word_occurrence)
    print(
        f"Number of unique words in training data, longer than {minimum_word_length} characters, which occur more than {minimum_word_occurrence} times: " + str(
            len(train_word_list)))

    test_word_list = extract_features(test_df, minimum_word_length, minimum_word_occurrence)
    print(
        f"Number of unique words in test data, longer than {minimum_word_length} characters, which occur more than {minimum_word_occurrence} times: " + str(
            len(test_word_list)))

    return train_word_list, test_word_list


def extract_features(data, minimum_word_length, minimum_word_occurrence):
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

    word_data = word_data[word_data["words"].str.len() >= minimum_word_length]
    word_vc = word_data['words'].value_counts().reset_index()
    word_vc.columns = ['Word', 'Count']
    word_vc = word_vc[word_vc["Count"] >= minimum_word_occurrence]

    wordlist = word_vc['Word'].tolist()
    return wordlist


########################################################################################################################
# Task 3
########################################################################################################################
def task3(training_df, test_df, train_word_list, test_word_list, minimum_word_length, minimum_word_occurrence):
    print_header("#", "Task 3")
    train_word_counts_pos = word_in_review_occurrences(training_df, train_word_list, minimum_word_length, minimum_word_occurrence,
                                                   "train_word_counts_pos", 'positive')
    print(train_word_counts_pos.sort_values(by='review_count', ascending=False))
    print_divider("-")

    train_word_counts_neg = word_in_review_occurrences(training_df, train_word_list, minimum_word_length, minimum_word_occurrence,
                                                   "train_word_counts_neg", 'negative')
    print(train_word_counts_neg.sort_values(by='review_count', ascending=False))
    print_divider("-")

    test_word_counts_pos = word_in_review_occurrences(test_df, test_word_list, minimum_word_length, minimum_word_occurrence,
                                                  "test_word_counts_pos", 'positive')
    print(test_word_counts_pos.sort_values(by='review_count', ascending=False))
    print_divider("-")

    test_word_counts_neg = word_in_review_occurrences(test_df, test_word_list, minimum_word_length, minimum_word_occurrence,
                                                  "test_word_counts_neg", 'negative')
    print(test_word_counts_neg.sort_values(by='review_count', ascending=False))
    print_divider("-")

    return train_word_counts_pos, train_word_counts_neg, test_word_counts_pos, test_word_counts_neg


def word_in_review_occurrences(df, word_list, minimum_word_length, minimum_word_occurrence, cache_name, sentiment):
    words_cache_file = source_filename + "-" + cache_name + "-" + str(minimum_word_length) + "-" + str(
        minimum_word_occurrence) + ".pkl"
    cache_filepath = os.path.join(cache_folder, words_cache_file)

    # this task is expensive, so I will cache the results, taking into account the minimum word length and minimum word occurrence may change
    if os.path.exists(cache_filepath):

        with open(cache_filepath, 'rb') as f:
            word_counts_df = pickle.load(f)
        print(f'Loaded data from cache: {cache_filepath}\n')

    else:
        word_counts = {word: 0 for word in word_list}
        reviews = df[df['Sentiment'] == sentiment]
        total_reviews = reviews.shape[0]
        start_time = time.time()

        for i, review in enumerate(reviews['Review']):
            for word in word_list:
                if word in review:
                    word_counts[word] += 1
            progressbar(i, total_reviews, start_time)

        word_counts_df = pd.DataFrame(list(word_counts.items()), columns=['Word', 'review_count'])

        with open(cache_filepath, 'wb') as f:
            pickle.dump(word_counts_df, f)
        print(f'Saved data to cache: {cache_filepath}\n')

    return word_counts_df


########################################################################################################################
# Task 4
########################################################################################################################
def task4(training_df, test_df, train_word_counts_pos, train_word_counts_neg, test_word_counts_pos, test_word_counts_neg):
    print_header("#", "Task 4")

    # To calculate conditional probability, I need to find the probability a word is in a review,
    # and the probability a review is positive,
    # then divide the former by the latter.

    # calculate priors
    training_positive_prior = calculatePriors(training_df, 'positive')
    training_negative_prior = calculatePriors(training_df, 'negative')
    test_positive_prior = calculatePriors(test_df, 'positive')
    test_negative_prior = calculatePriors(test_df, 'negative')

    print("Priors")
    print(f"Training DataProbability. Positive: {training_positive_prior:.2f}, Negative: {training_negative_prior:.2f}")
    print(f"Test Data Probability. Positive: {test_positive_prior:.2f}, Negative: {test_negative_prior:.2f}")

    print_divider("-")

    # calculate conditional probability
    train_word_counts_pos = calculateConditionalProbability(train_word_counts_pos, training_df.shape[0], training_positive_prior)
    train_word_counts_neg = calculateConditionalProbability(train_word_counts_neg, training_df.shape[0], training_negative_prior)
    test_word_counts_pos = calculateConditionalProbability(test_word_counts_pos, test_df.shape[0], test_positive_prior)
    test_word_counts_neg = calculateConditionalProbability(test_word_counts_neg, test_df.shape[0], test_negative_prior)

    print(f" Number of training Reviews: {training_df.shape[0]}")
    print(f" Number of test Reviews: {test_df.shape[0]}")
    print_divider("-")

    print("Training Data")
    print("\nPositive")
    print (train_word_counts_pos)
    print("\nNegative")
    print (train_word_counts_neg)
    print_divider("-")

    print("Test Data")
    print("\nPositive")
    print (test_word_counts_pos)
    print("\nNegative")
    print (test_word_counts_neg)

    return (
        training_positive_prior, training_negative_prior, test_positive_prior, test_negative_prior,
        train_word_counts_pos, train_word_counts_neg, test_word_counts_pos, test_word_counts_neg
    )


def calculatePriors(df, sentiment):
    return df[df['Sentiment'] == sentiment].shape[0] / df.shape[0]


def calculateConditionalProbability(word_counts_df, review_count, sentiment_probability):
    alpha = 1
    # Calculate the probability of each word being in a review
    word_counts_df['probability'] = word_counts_df['review_count'] + alpha / review_count + (alpha * 2)

    # Calculate the conditional probability of each word being in a review, given the review is positive, with laplace smoothing
    word_counts_df['conditional_probability'] = word_counts_df['probability'] + alpha / sentiment_probability + (2*alpha)
    return word_counts_df


########################################################################################################################
# Task 5
########################################################################################################################
def task5(training_positive_prior, training_negative_prior, test_positive_prior, test_negative_prior,
          train_word_counts_pos, train_word_counts_neg, test_word_counts_pos, test_word_counts_neg,
          test_df, training_df):
    print_header("#", "Task 5")

    train_predicitons_cache_name = f"{source_filename}-train-predictions-{minimumWordLength}-{minimumWordOccurrence}.pkl"
    train_predicitons = getPredictionsCache(train_predicitons_cache_name)
    if train_predicitons == None:
        train_predicitons = bayesianPredictor(training_df, train_word_counts_pos, train_word_counts_neg, training_positive_prior, training_negative_prior)
        cachePredictions(train_predicitons_cache_name, train_predicitons)
    training_df['Prediction'] = train_predicitons

    print("Training Data Predictions")
    print(training_df[['Review', 'Prediction']])
    print_divider("-")

    test_predictions_cache_name = f"{source_filename}-test-predictions-{minimumWordLength}-{minimumWordOccurrence}.pkl"
    test_predicitons = getPredictionsCache(test_predictions_cache_name)
    if test_predicitons == None:
        test_predicitons = bayesianPredictor(test_df, test_word_counts_pos, test_word_counts_neg, test_positive_prior, test_negative_prior)
        cachePredictions(test_predictions_cache_name, test_predicitons)
    test_df['Prediction'] = test_predicitons

    print("Test Data Predictions")
    print(test_df[['Review', 'Prediction']])

    return training_df, test_df


def getPredictionsCache(cache_name):
    cache_filepath = os.path.join(cache_folder, cache_name)

    # this task is expensive, so I will cache the results, taking into account the minimum word length and minimum word occurrence may change
    if os.path.exists(cache_filepath):

        with open(cache_filepath, 'rb') as f:
            predictions = pickle.load(f)
        print(f'Loaded data from cache: {cache_filepath}\n')
        return predictions

    else:
        return


def cachePredictions(cache_name, predicitons):
    cache_filepath = os.path.join(cache_folder, cache_name)

    with open(cache_filepath, 'wb') as f:
        pickle.dump(predicitons, f)
    print(f'\nSaved data to cache: {cache_filepath}\n')


def calculate_log_likelihood(review, word_counts, prior):
    log_likelihood = math.log(prior)
    for word in review:
        if word in word_counts['Word'].values:
            word_prob = word_counts[word_counts['Word'] == word]['conditional_probability'].values[0]
            log_likelihood += math.log(word_prob)
    return log_likelihood


def bayesianPredictor(df, word_counts_pos, word_counts_neg, positive_prior, negative_prior):
    predictions = []
    starttime = time.time()
    i = 0

    for review in df['Review']:
        log_likelihood_pos = calculate_log_likelihood(review, word_counts_pos, positive_prior)
        log_likelihood_neg = calculate_log_likelihood(review, word_counts_neg, negative_prior)

        if log_likelihood_pos > log_likelihood_neg:
            predictions.append('positive')
        else:
            predictions.append('negative')
        i += 1
        progressbar(i, df['Review'].shape[0], starttime)

    return predictions


########################################################################################################################
# Task 6
########################################################################################################################
def task6(df):
    print_header("#", "Task 6")
    n = 5
    shuffle = True
    test_size = 0.2
    randomState = 42

    kfold =  sklearn.model_selection.KFold(n_splits=n, shuffle=shuffle, random_state=randomState)


    folds = kfold.split(df)

    for fold in folds:
        train, test = sklearn.model_selection.train_test_split(df, test_size=test_size, random_state=randomState)

        print (f"Train: {train}, \n\nTest: {test}")


########################################################################################################################
# Task 7
########################################################################################################################
def task7():
    print_header("#", "Task 7")


########################################################################################################################
# Output Formatting
########################################################################################################################
def print_header(char, text):
    print('\n' + char * 80)
    print(text.center(80, ' '))
    print(str(char * 80) + '\n')


def print_divider(char):
    print("\n" + char * 80 + "\n")


# Function to display a progress bar so the user knows the program is still running and how far along it is
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
    progress_string = f'\r{("█" * num_blocks)}{("_" * (100 - num_blocks))} {percentage:.2f}% | Elapsed: {elapsed_time:.2f}s | Remaining: {remaining_time:.2f}s'
    if i == upper_range - 1:
        print(progress_string)
    else:
        print(progress_string, end='', flush=True)


########################################################################################################################
# Main
########################################################################################################################

def main():
    start_time = time.time()  # start measuring time

    # Create a cache folder if it does not exist
    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)

    df = read_data()

    # Task 1
    training_df, test_df = task1(df)

    # Task 2
    (train_word_list, test_word_list) \
        = task2(training_df, test_df, minimumWordLength, minimumWordOccurrence)

    # Task 3
    (train_word_counts_pos, train_word_counts_neg, test_word_counts_pos, test_word_counts_neg) \
        = task3(training_df, test_df, train_word_list, test_word_list, minimumWordLength, minimumWordOccurrence)

    # Task 4
    (training_positive_prior, training_negative_prior, test_positive_prior, test_negative_prior,
     train_word_counts_pos, train_word_counts_neg, test_word_counts_pos, test_word_counts_neg) \
        = task4(training_df, test_df, train_word_counts_pos, train_word_counts_neg, test_word_counts_pos,
                test_word_counts_neg)

    # Task 5
    training_df, test_df = task5(training_positive_prior, training_negative_prior, test_positive_prior,
                                 test_negative_prior,
                                 train_word_counts_pos, train_word_counts_neg, test_word_counts_pos,
                                 test_word_counts_neg,
                                 test_df, training_df)

    # Task 6
    task6(df)

    # Task 7
    task7()

    # Print elapsed time
    print_header("#", "Execution Time")
    print(f"Execution time: {time.time() - start_time:.2f} seconds")


if __name__ == '__main__':
    main()
