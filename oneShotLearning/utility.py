import random


def get_class_input_indexes(class_list, classes):
    class_indexes = []
    for input_class in classes:
        indexes = [i for i, x in enumerate(class_list) if x == input_class]
        class_indexes.append(indexes)
    return class_indexes


def get_random_classes(xs, ys, classes, n_class_examples_train, n_class_examples_test):
    new_xs_train, new_ys_train = [], []
    new_xs_test, new_ys_test = [], []
    class_indexes = get_class_input_indexes(ys, classes)
    for i in range(0, n_class_examples_train):
        for class_elements in class_indexes:
            index_random_element = random.choice(class_elements)
            class_elements.remove(index_random_element)
            new_xs_train.append(xs[index_random_element])
            new_ys_train.append(ys[index_random_element])
    for i in range(0, n_class_examples_test):
        for class_elements in class_indexes:
            index_random_element = random.choice(class_elements)
            class_elements.remove(index_random_element)
            new_xs_test.append(xs[index_random_element])
            new_ys_test.append(ys[index_random_element])

    return new_xs_train, new_ys_train, new_xs_test, new_ys_test
