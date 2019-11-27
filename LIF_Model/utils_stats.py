from csv import reader
import random as rand
from math import sqrt

#Load csv file
def load_csv(filename):
    dataset = list()
    with open(filename,'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

#convert string column to float
def str_column_to_float(dataset,column):
    for row in dataset:
        row[column] = float(row[column].strip())

#convert string column to integer
def str_column_to_int(dataset,column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i,value in enumerate(unique):
        lookup[value] = i
        for row in dataset:
            row[column] = lookup[row[column]]
    return lookup

def cross_validation_split(dataset,n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset)/n_folds)
    for _ in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = rand.randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
        return dataset_split

def euclidean_distance(row1,row2):
    #importar funcion sqrt de libreria math
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i])**2
    return sqrt(distance)

import numpy as np
row1 = np.array([4.4,5.0,6.2])
row2 = np.array([5,4.5,5,4])