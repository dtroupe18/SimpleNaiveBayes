"""
1. Title: Pima Indians Diabetes Database

2. Sources:
   (a) Original owners: National Institute of Diabetes and Digestive and
                        Kidney Diseases
   (b) Donor of database: Vincent Sigillito (vgs@aplcen.apl.jhu.edu)
                          Research Center, RMI Group Leader
                          Applied Physics Laboratory
                          The Johns Hopkins University
                          Johns Hopkins Road
                          Laurel, MD 20707
                          (301) 953-6231
   (c) Date received: 9 May 1990

3. Past Usage:
    1. Smith,~J.~W., Everhart,~J.~E., Dickson,~W.~C., Knowler,~W.~C., \&
       Johannes,~R.~S. (1988). Using the ADAP learning algorithm to forecast
       the onset of diabetes mellitus.  In {\it Proceedings of the Symposium
       on Computer Applications and Medical Care} (pp. 261--265).  IEEE
       Computer Society Press.

       The diagnostic, binary-valued variable investigated is whether the
       patient shows signs of diabetes according to World Health Organization
       criteria (i.e., if the 2 hour post-load plasma glucose was at least
       200 mg/dl at any survey  examination or if found during routine medical
       care).   The population lives near Phoenix, Arizona, USA.

       Results: Their ADAP algorithm makes a real-valued prediction between
       0 and 1.  This was transformed into a binary decision using a cutoff of
       0.448.  Using 576 training instances, the sensitivity and specificity
       of their algorithm was 76% on the remaining 192 instances.

4. Relevant Information:
      Several constraints were placed on the selection of these instances from
      a larger database.  In particular, all patients here are females at
      least 21 years old of Pima Indian heritage.  ADAP is an adaptive learning
      routine that generates and executes digital analogs of perceptron-like
      devices.  It is a unique algorithm; see the paper for details.

5. Number of Instances: 768

6. Number of Attributes: 8 plus class

7. For Each Attribute: (all numeric-valued)
   1. Number of times pregnant
   2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
   3. Diastolic blood pressure (mm Hg)
   4. Triceps skin fold thickness (mm)
   5. 2-Hour serum insulin (mu U/ml)
   6. Body mass index (weight in kg/(height in m)^2)
   7. Diabetes pedigree function
   8. Age (years)
   9. Class variable (0 or 1)

8. Missing Attribute Values: Yes

9. Class Distribution: (class value 1 is interpreted as "tested positive for
   diabetes")

   Class Value  Number of instances
   0            500
   1            268

10. Brief statistical analysis:

    Attribute number:    Mean:   Standard Deviation:
    1.                     3.8     3.4
    2.                   120.9    32.0
    3.                    69.1    19.4
    4.                    20.5    16.0
    5.                    79.8   115.2
    6.                    32.0     7.9
    7.                     0.5     0.3
    8.                    33.2    11.8


Gaussian Probability Distribution "http://mathworld.wolfram.com/NormalDistribution.html"
Gaussian Probability Distribution "https://www.physics.ohio-state.edu/~gan/teaching/spring04/Chapter3.pdf"
"""

import csv
import random
import math
import numpy as np


def load_csv(filename):
    """

    :param filename: name of csv file
    :return: data set as a 2 dimensional list where each row in a list
    """
    lines = csv.reader(open(filename, 'r'))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset


# data = load_csv('pima-indians-diabetes.data.csv')
# print(data)

def split_dataset(dataset, ratio):
    """
    split dataset into training and testing
    :param dataset: Two dimensional list
    :param ratio: Percentage of data to go into the training set
    :return: Training set and testing set
    """
    size_of_training_set = int(len(dataset) * ratio)
    train_set = []
    test_set = list(dataset)

    while len(train_set) < size_of_training_set:
        index = random.randrange(len(test_set))
        train_set.append(test_set.pop(index))

    return [train_set, test_set]


# training_set, testing_set = split_dataset(data, 0.67)
# print(training_set)
# print(testing_set)


def separate_by_label(dataset):
    """

    :param dataset: two dimensional list of data values
    :return: dictionary where labels are keys and
    values are the data points with that label
    """
    separated = {}
    for x in range(len(dataset)):
        row = dataset[x]
        if row[-1] not in separated:
            separated[row[-1]] = []
        separated[row[-1]].append(row)

    return separated


# separated = separate_by_label(data)
# print(separated)
# print(separated[1])
# print(separated[0])


def calc_mean(lst):
    return sum(lst) / float(len(lst))


def calc_standard_deviation(lst):
    avg = calc_mean(lst)
    variance = sum([pow(x - avg, 2) for x in lst]) / float(len(lst) - 1)

    return math.sqrt(variance)


# numbers = [1, 2, 3, 4, 5]
# print(calc_mean(numbers))
# print(calc_standard_deviation(numbers))


def summarize_data(lst):
    """
    Calculate the mean and standard deviation for each attribute
    :param lst: list
    :return: list with mean and standard deviation for each attribute
    """

    summaries = [(calc_mean(attribute), calc_standard_deviation(attribute)) for attribute in zip(*lst)]
    del summaries[-1]
    return summaries


# summarize_me = [[1, 20, 0], [2, 21, 1], [3, 22, 0]]
# print(summarize_data(summarize_me))


def summarize_by_label(data):
    """
    Method to summarize the attributes for each label

    :param data:
    :return: dict label: [(atr mean, atr stdv), (atr mean, atr stdv)....]
    """
    separated_data = separate_by_label(data)
    summaries = {}
    for label, instances in separated_data.items():
        summaries[label] = summarize_data(instances)
    return summaries

# fake_data = [[1, 20, 1], [2, 21, 0], [3, 22, 1], [4,22,0]]
# fake_summary = summarize_by_label(fake_data)


def calc_probability(x, mean, standard_deviation):
    """

    :param x: value
    :param mean: average
    :param standard_deviation: standard deviation
    :return: probability of that value given a normal distribution
    """
    # e ^ -(y - mean)^2 / (2 * (standard deviation)^2)
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(standard_deviation, 2))))
    # ( 1 / sqrt(2π) ^ exponent
    return (1 / (math.sqrt(2 * math.pi) * standard_deviation)) * exponent


# x = 57
# mean = 50
# stand_dev = 5
# print(calc_probability(x, mean, stand_dev))


def calc_label_probabilities(summaries, input_vector):
    """
    the probability of a given data instance is calculated by multiplying together
    the attribute probabilities for each class. The result is a map of class values
    to probabilities.

    :param summaries:
    :param input_vector:
    :return: dict
    """
    probabilities = {}
    for label, label_summaries in summaries.items():
        probabilities[label] = 1
        for i in range(len(label_summaries)):
            mean, standard_dev = label_summaries[i]
            x = input_vector[i]
            probabilities[label] *= calc_probability(x, mean, standard_dev)

    return probabilities

# fake_input_vec = [1.1, 2.3]
# fake_probabilities = calc_label_probabilities(fake_summary, fake_input_vec)
# print(fake_probabilities)


def predict(summaries, input_vector):
    """
    Calculate the probability of a data instance belonging
    to each label. We look for the largest probability and return
    the associated class.
    :param summaries:
    :param input_vector:
    :return:
    """
    probabilities = calc_label_probabilities(summaries, input_vector)
    best_label, best_prob = None, -1
    for label, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = label

    return best_label

# summaries = {'A': [(1, 0.5)], 'B': [(20, 5.0)]}
# inputVector = 1.1
# print(predict(summaries, inputVector))


def get_predictions(summaries, test_set):
    """
     Make predictions for each data instance in our
     test dataset
    """

    predictions = []
    for i in range(len(test_set)):
        result = predict(summaries, test_set[i])
        predictions.append(result)

    return predictions

# summaries = {'A': [(1, 0.5)], 'B': [(20, 5.0)]}
# testSet = [1.1, 19.1]
# predictions = get_predictions(summaries, testSet)
# print(predictions)


def get_accuracy(test_set, predictions):
    """
    Compare predictions to class labels in the test dataset
    and get our classification accuracy
    """
    correct = 0
    for i in range(len(test_set)):
        if test_set[i][-1] == predictions[i]:
            correct += 1

    return (correct / float(len(test_set))) * 100


# fake_testSet = [[1, 1, 1, 'a'], [2, 2, 2, 'a'], [3, 3, 3, 'b']]
# fake_predictions = ['a', 'a', 'a']
# fake_accuracy = get_accuracy(fake_testSet, fake_predictions)
# print(fake_accuracy)

def main(filename, split_ratio):
    data = load_csv(filename)
    training_set, testing_set = split_dataset(data, split_ratio)
    print("Size of Training Set: ", len(training_set))
    print("Size of Testing Set: ", len(testing_set))

    # create model
    summaries = summarize_by_label(training_set)

    # test mode
    predictions = get_predictions(summaries, testing_set)
    accuracy = get_accuracy(testing_set, predictions)
    print('Accuracy: {0}%'.format(accuracy))


main('pima-indians-diabetes.data.csv', 0.70)

