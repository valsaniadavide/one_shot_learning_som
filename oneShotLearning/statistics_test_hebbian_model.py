import json
import os
import numpy as np
import seaborn as sb

from utils.constants import Constants
import matplotlib.pyplot as plt


def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def plot_results_on_bar(data_results, labels_results, labels_threshold, data_type='Training'):
    width = 0.10  # the width of the bars
    (row, columns) = np.shape(data_results)
    fig, ax = plt.subplots(figsize=(15, 5))
    rects = []
    x = np.arange(columns)  # the label locations
    position = 0
    colors = sb.color_palette('bright', n_colors=row)
    for i in range(0, row):
        rect = ax.bar(x + position, data_results[i], width, label=labels_threshold[i].replace('_', ' ').split(' ', 1)[1], color=colors[i])
        rects.append(rect)
        position += width
    labels_results = [string.split(' ', 3)[3] for string in labels_results]
    ax.set_ylabel('Accuracy')
    ax.set_title('{} Set Accuracy Performance of Hebbian Models'.format(data_type))
    ax.set_xticks(x + .25)
    ax.set_xticklabels(labels_results)
    ax.legend()
    for rect in rects:
        autolabel(rect, ax)
    fig.tight_layout()
    plt.savefig(os.path.join(Constants.statistics_path, 'plots_{}_accuracy_results.png'.format(data_type.lower())))
    plt.show()

if __name__ == '__main__':
    threshold = range(60, 90, 5)
    result = {}
    for value_threshold in threshold:
        name_folder = os.path.join('one_shot_results_threshold_{}'.format(value_threshold))
        name_object_test = 'test_threshold_{}'.format(value_threshold)
        result[name_object_test] = {}
        result[name_object_test]['value_threshold'] = value_threshold
        labels_result = []
        for class_index in Constants.classes:
            class_label = Constants.label_classes[class_index]
            with open(
                    os.path.join(Constants.statistics_path, name_folder,
                                 '{}_one_shot_class_{}'.format(class_index + 1, class_label),
                                 'results_class_{}.md'.format(class_label))) as f:
                file_split = [line.split('\n') for line in f]
                file_split = [line[0][1:-1].split('|') for line in file_split if 'Accuracy' in line[0]]
                for element in file_split:
                    if result[name_object_test].get(element[0]) is None:
                        result[name_object_test][element[0]] = [float(element[1])]
                    else:
                        result[name_object_test][element[0]].append(float(element[1]))
    with open(os.path.join(Constants.statistics_path, 'results_tests.json'), 'w') as f:
        json.dump(result, f, indent=True)

    train_results, test_results = [], []
    labels_train, labels_test = [], []
    labels_threshold = []

    for key, value in result.items():
        threshold_val = value.pop('value_threshold', None)
        train_result, test_result = [], []
        labels_threshold.append(key)
        for key_inner, value_inner in value.items():
            mean = round(np.mean(value_inner), 2)
            if 'Training' in key_inner:
                train_result.append(mean), labels_train.append(key_inner)
            else:
                test_result.append(mean), labels_test.append(key_inner)
        train_results.append(train_result), test_results.append(test_result)

    train_results, test_results = np.array(train_results), np.array(test_results)
    labels_train, labels_test = list(dict.fromkeys(labels_train)), list(dict.fromkeys(labels_test))
    plot_results_on_bar(train_results, labels_train, labels_threshold, data_type='Training')
    plot_results_on_bar(test_results, labels_test, labels_threshold, data_type='Test')


