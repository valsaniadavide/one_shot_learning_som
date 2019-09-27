import random


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


def get_random_classes(xs, ys, classes, n_class_examples_train, n_class_examples_test):
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
            random.seed(10)
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
                new_xs_test.append(xs[class_elements[index]])
                new_ys_test.append(ys[class_elements[index]])

    return new_xs_train, new_ys_train, new_xs_test, new_ys_test
