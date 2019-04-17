import numpy as np
import os
from utils.utils import from_csv_with_filenames, from_csv_visual_10classes, from_csv_visual_100classes
from utils.constants import Constants
import matplotlib
import matplotlib.pyplot as plt
from collections import OrderedDict
import argparse
from sklearn.cluster import KMeans, MiniBatchKMeans

random_seed = 42


def get_prototypes(xs, ys):
    prototype_dict = {unique_y: [] for unique_y in set(ys)}
    for i, x in enumerate(xs):
        prototype_dict[ys[i]].append(x)
    prototype_dict = {k: np.array(prototype) for k, prototype in prototype_dict.items()}
    result = dict()
    for y in set(sorted(ys)):
        result[y] = np.mean(prototype_dict[y], axis=0)
    return result

def average_prototype_distance_matrix(xs, ys):
    """
        Relies on all y in ys being in the interval [0, number of classes) - take care of
        handling that if importing it from another script.
        As a general rule, x in xs and y in ys are not ordered by class but by speaker
        That is why I am not using models.som.SOMTest.classPrototype, which
        has the opposite assumption.
    """
    prototype_distance_matrix = np.zeros((len(set(ys)), len(set(ys))))
    # compute prototypes in dictionary d
    prototype_dict = {unique_y: [] for unique_y in set(ys)}
    for i, x in enumerate(xs):
        prototype_dict[ys[i]].append(x)
    prototype_dict = {k: np.array(prototype) for k, prototype in prototype_dict.items()}
    for y in set(ys):
        prototype_dict[y] = np.mean(prototype_dict[y], axis=0)
    prototypes = np.asarray(list(prototype_dict.values())).T
    for i, x in enumerate(xs):
        prototype_distance_matrix[ys[i]][:] += np.mean(np.absolute(prototypes - x.reshape((-1, 1))), axis=0).T
    print(prototype_distance_matrix)
    fig, ax = plt.subplots()
    ax.matshow(prototype_distance_matrix, cmap=plt.get_cmap('Greys'))
    for (i, j), distance in np.ndenumerate(prototype_distance_matrix):
        ax.text(j, i, '{:0.2f}'.format(distance), ha='center', va='center')
    plt.show(True)

def examples_distance(xs, i1, i2):
    return np.linalg.norm(xs[i1]-xs[i2])

def cluster_compactness(xs, ys, num_classes):
    print('Fitting clustering model...')
    model = KMeans(num_classes, max_iter=10000, tol=1e-30, random_state=random_seed)
    #model = KMeans(n_clusters = num_classes,max_iter=100,n_init=50,algorithm='elkan') # giorgia
    cluster_ys = model.fit_predict(xs)
    print('Done. Computing occurrences...')
    # columns: clusters; rows: classes
    occurrences_matrix = np.zeros((num_classes, num_classes))
    cluster_belonging_dict = {y: [] for y in range(num_classes)}
    for i, c_y in enumerate(cluster_ys):
        occurrences_matrix[c_y][ys[i]] += 1
        cluster_belonging_dict[c_y].append(i)
    occurrences_matrix /= np.sum(occurrences_matrix, axis=0) # normalize column-wise
    occurrences_matrix = np.sort(occurrences_matrix, axis=0)
    #print(occurrences_matrix[:][-1])
    print('Computing intra-cluster distance...')
    intra_cluster_distance = [0 for i in range(num_classes)]
    inter_cluster_distance = 0
    for i in range(num_classes):
        intra_temp = 0
        for pos, j in enumerate(cluster_belonging_dict[i]):
            x1 = xs[j]
            for k in cluster_belonging_dict[i][pos+1:]:
                x2 = xs[k]
                intra_temp += np.linalg.norm(x1-x2)
        intra_cluster_distance[i] = intra_temp / len(cluster_belonging_dict[i])
    for i, x1 in enumerate(xs):
        for x2 in xs[i+1:]:
            inter_cluster_distance += np.linalg.norm(x1-x2)
    inter_cluster_distance /= len(xs)
    plt.ylim(0, 1)
    plt.bar(range(num_classes), intra_cluster_distance/inter_cluster_distance)
    plt.show()
    mean = np.mean(intra_cluster_distance/inter_cluster_distance)
    var = np.var(intra_cluster_distance/inter_cluster_distance)
    print(intra_cluster_distance/inter_cluster_distance)
    print('Mean {}; variance {}'.format(mean, var))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze representation quality.')
    parser.add_argument('--csv-path', metavar='csv_path', type=str, required=True, help='The csv file with the extracted representations.')
    parser.add_argument('--classes100', action='store_true',
                        help='Specify whether you are analyzing \
                        a file with representations from 100 classes, as the loading functions are different.',
                        default=False)
    parser.add_argument('--is-audio', action='store_true', default=False,
                        help='Specify whether the csv contains audio representations, as the loading functions are different.')
    parser.add_argument('--cluster', action='store_true', default=False)

    args = parser.parse_args()

    if not args.classes100:
        num_classes = 10
        if not args.is_audio:
            xs, ys = from_csv_visual_10classes(args.csv_path)
        else:
            xs, ys, _ = from_csv_with_filenames(args.csv_path)
        ys = [int(y)-1000 for y in ys] # see comment in average_prototype_distance_matrix
    else:
        num_classes = 100
        if not args.is_audio:
            xs, ys = from_csv_visual_100classes(args.csv_path)
        else:
            xs, ys, _ =  from_csv_with_filenames(args.csv_path)

    if not args.cluster:
        average_prototype_distance_matrix(xs, ys)
    else:
        cluster_compactness(xs, ys, num_classes)
