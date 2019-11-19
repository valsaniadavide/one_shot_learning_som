import os, shutil
import random

import pandas as pd
import numpy as np
from models.som.SOMTest import show_som, show_confusion
from utils.utils import from_csv_with_filenames, from_csv_visual_10classes, from_npy_visual_data
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, Normalizer
from utils.constants import Constants
from sklearn.metrics.pairwise import euclidean_distances


def get_class_input_indexes(class_list, classes):
    """
    Function that given the number of total classes and the corresponding label of examples return a list of list
    where the inner list represent the list of the indexes of the examples labelled with the corresponding class (index
    of the outer list)

    :param class_list: list of labels associated to examples
    :param classes: number of total classes
    :return: list of list of examples indexes per class
    """
    class_indexes = []
    for input_class in classes:
        indexes = [i for i, x in enumerate(class_list) if x == input_class]
        class_indexes.append(indexes)
    return class_indexes


def get_random_classes(xs, ys, classes, n_class_examples_train, n_class_examples_test, class_to_exclude=-1):
    """
    Function that return 'n_class_examples_train' examples per class used for training and 'n_class_examples_test'
    examples per class used for test

    :param xs: list of examples
    :param ys: labels of examples
    :param classes: number of total classes associated to examples
    :param n_class_examples_train: number of examples required for train
    :param n_class_examples_test: number of examples required for test (if -1 are returned all remaining
                                    examples after train filtering)
    :return: list of training examples with corresponding label (new_xs_train, new_ys_train) and list of test
                examples with corresponding label (new_xs_test, new_ys_test)
    """
    new_xs_train, new_ys_train = [], []
    new_xs_test, new_ys_test = [], []
    class_indexes = get_class_input_indexes(ys, classes)
    for i in range(0, n_class_examples_train):
        for class_elements in class_indexes:
            # random.seed(20)
            if len(class_elements) > 0:
                index_random_element = random.choice(class_elements)
                class_elements.remove(index_random_element)
                new_xs_train.append(xs[index_random_element])
                new_ys_train.append(ys[index_random_element])
    if n_class_examples_test != -1:
        for i in range(0, n_class_examples_test):
            for class_elements in class_indexes:
                # random.seed(10)
                index_random_element = random.choice(class_elements)
                class_elements.remove(index_random_element)
                new_xs_test.append(xs[index_random_element])
                new_ys_test.append(ys[index_random_element])
    else:
        for class_elements in class_indexes:
            for index in range(0, len(class_elements)):
                # if index != class_to_exclude:
                new_xs_test.append(xs[class_elements[index]])
                new_ys_test.append(ys[class_elements[index]])

    return new_xs_train, new_ys_train, new_xs_test, new_ys_test


def get_examples_of_class(xs, ys, classes, class_to_extract):
    """
    Function that extract all examples of a class from a dataset in input
    :param xs: dataset in input
    :param ys: corresponding labels to dataset
    :param classes: numeric list of classes
    :param class_to_extract: number of the class to extract
    :return: list of examples extracted (with labels associated) and others remaining inputs
    """
    xs_others = []
    ys_others = []
    classes_indexes = get_class_input_indexes(ys, classes)
    ext_xs = []
    ext_ys = []
    for index_class_element in classes_indexes[class_to_extract]:
        ext_xs.append(xs[index_class_element])
        ext_ys.append(ys[index_class_element])
    classes_indexes.pop(class_to_extract)
    for class_elements in classes_indexes:
        for index_element in class_elements:
            xs_others.append(xs[index_element])
            ys_others.append(ys[index_element])
    return ext_xs, ext_ys, np.array(xs_others), np.array(ys_others)


def import_data(visual_data_path, audio_data_path, segmented=False):
    """
    Function that loads data from paths
    :param visual_data_path: visual data paths
    :param audio_data_path: audio data paths
    :return: visual and audio data loaded from path
    """
    a_xs, a_ys, filenames_audio = from_csv_with_filenames(audio_data_path)
    filenames_visual = []
    v_xs, v_ys = [], []
    if segmented:
        v_xs, v_ys, _ = from_npy_visual_data(visual_data_path)
    else:
        v_xs, v_ys, filenames_visual = from_csv_visual_10classes(visual_data_path)
        v_ys = [int(y) - 1000 for y in v_ys]
    a_ys = [int(y) - 1000 for y in a_ys]
    v_xs = StandardScaler().fit_transform(v_xs)
    a_xs = StandardScaler().fit_transform(a_xs)
    v_ys = np.array(v_ys)
    a_ys = np.array(a_ys)
    return v_xs, v_ys, a_xs, a_ys, filenames_visual, filenames_audio


def print_charts(som, xs, ys, label_classes, suffix, title, subpath='test_data'):
    show_som(som, xs, ys, label_classes,
             title, dark=False, suffix=suffix, subpath=subpath)
    show_confusion(som, xs, ys, title=title, suffix=suffix, subpath=subpath)


def clean_folders(path):
    try:
        for the_file in os.listdir(path):
            file_path = os.path.join(path, the_file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
    except Exception as e:
        pass


def get_min_max_mean_input_feature(xs):
    df = pd.DataFrame(xs)
    data_stat = pd.DataFrame()
    data_stat['min_value'] = df.min(axis=0)
    data_stat['max_value'] = df.max(axis=0)
    data_stat['mean_value'] = df.mean(axis=0)
    data_stat['variance_value'] = df.var(axis=0)
    return data_stat


def compute_inputs_distances(xs):
    """
    Function that compute the distances between inputs passed as param
    :param xs: list of input
    :return: distances between couples of inputs
    """
    elements = list.copy(xs)
    number_of_elements = len(elements)
    compactness = 0
    for index in range(0, len(xs)):
        element = xs[index]
        elements.remove(element)
        if len(elements) > 0:
            element_distances = np.sum(euclidean_distances(np.array(element).reshape(1, -1), elements))
            compactness += element_distances
    return compactness / number_of_elements


def inputs_compactness(xs, ys):
    """
    Function that compute compactness for each class
    :param xs: list of inputs
    :param ys: list of labels
    :return: list of compactness values for each class
    """
    classes = Constants.classes
    intra_classes_distances = []
    inter_classes_distances = []
    for _class in classes:
        ext_xs, ext_ys, others_xs, others_ys = get_examples_of_class(xs, ys, classes, _class)
        intra_classes_distances.append(compute_inputs_distances(ext_xs))
        inter_classes_distances.append(compute_inputs_distances(others_xs))
    return np.divide(intra_classes_distances, inter_classes_distances)


def train_som_and_get_weight(som_v=None, som_a=None, v_xs=None, a_xs=None):
    weights_v, weights_a = [], []
    if som_v is not None and v_xs is not None:
        print('--> Training SOM (Visual)')
        som_v.train(v_xs, pca_initialization_weights=True, verbose=True)
        weights_v = som_v.get_weights()
    if som_a is not None and a_xs is not None:
        print('--> Training SOM (Audio)')
        som_a.train(a_xs, pca_initialization_weights=True, verbose=True)
        weights_a = som_a.get_weights()
    print('<<< Done')
    return weights_v, weights_a, som_v, som_a


def safe_create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def write_md_file(file_name, mean_accuracy, mean_accuracy_a, mean_variance, mean_variance_a, scaler, dimensions, alpha,
                  sigma, n_iters, splits_v, splits_a, test_name, precisions, recall, f1_scores):
    f = open(file_name, "a+")
    f.write('\n## {}\n'.format(test_name))
    f.write(
        '\t - Scale type = {} \n \t - Dimension = {}\n \t - Alpha = {}\n \t - Sigma = {}\n \t - Iterations = {}\n \t - KFold visual = {}\n \t - KFold audio = {}\n'.format(
            scaler, dimensions, alpha, sigma, n_iters, splits_v, splits_a))
    f.write("| 					 | Source Visual | Source Audio |\n")
    f.write("|-------------------| ------------- | ------------ |\n")
    f.write("| __Mean Accuracy__ |  {} 			 | {}		    |\n".format(round(mean_accuracy, 2),
                                                                                round(mean_accuracy_a, 2)))
    f.write("| __Mean Variance__ |  {} 		 	 | {}    	    |\n".format(round(mean_variance, 4),
                                                                                 round(mean_variance_a, 4)))
    f.write("| __Mean Precision__ |  {} 		 	 | {}    	    |\n".format(round(precisions[0], 4),
                                                                                  round(precisions[1], 4)))
    f.write(
        "| __Mean Recall__ |  {} 		 	 | {}    	    |\n".format(round(recall[0], 4), round(recall[1], 4)))
    f.write("| __Mean F1 Score__ |  {} 		 	 | {}    	    |\n".format(round(f1_scores[0], 4),
                                                                                 round(f1_scores[1], 4)))
    f.close()
