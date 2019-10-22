import math
import operator
import os

from minisom import MiniSom
from sklearn.metrics import accuracy_score

from models.som.SOM import SOM
from matplotlib import pyplot as plt
import seaborn as sb
import numpy as np
import matplotlib.patches as m_patches

from utils.constants import Constants


class SelfOrganizingMap(object):
    """
    Class that implement the Adapter pattern that allow
    to use the HebbianModel with SOM class and MiniSOM library
    """

    def __init__(self, n, m, dim, n_iterations=50, learning_rate=0.3, sigma=4, data='audio',
                 neighborhood_function='gaussian'):
        """
        Initalization of the SOM

        :param m: number of rows
        :param n: number of columns
        :param dim: dimension of the inputs
        :param n_iterations: number of training iterations
        :param learning_rate: learning rate used to train the SOM
        :param sigma: neighborhood function value
        :param data: type of data 'audio' or 'visual'
        :param neighborhood_function: type of neighborhood function to apply to the train
        """
        self._miniSOM = MiniSom(n, m, dim, sigma=sigma,
                                learning_rate=learning_rate, neighborhood_function=neighborhood_function)
        self._som = SOM(m, n, dim, alpha=learning_rate, sigma=sigma, n_iterations=n_iterations, batch_size=1, data=data)
        self._n = n
        self._m = m
        self._dim = dim
        self._n_iterations = n_iterations
        self._learning_rate = learning_rate
        self._sigma = sigma
        self._neighborhood_function = neighborhood_function

    def train(self, xs, verbose=True, pca_initialization_weights=True):
        """
        Function that train the SOM

        :param xs: training set
        :param verbose: if true print the status of the training at each step
        """
        if pca_initialization_weights:
            self._miniSOM.pca_weights_init(xs)
        self._miniSOM.train_random(xs, self._n_iterations, verbose=verbose)
        weights = self._miniSOM.get_weights().transpose(2, 0, 1).reshape(np.shape(xs)[1], -1).transpose()
        print(np.shape(weights))
        self._set_weights(weights)
        self._som.set_weights(weights)

    def _set_weights(self, weights):
        self._weightages = weights

    def plot_som(self, xs, ys, type_dataset='train'):
        """
        Function that plot the SOM

        :param xs: list of elements
        :param ys: label of the elements
        :param type_dataset: label 'train' or 'test' used to pathfile
        """
        plt.figure(figsize=(15, 12))
        classes = Constants.label_classes
        colors = sb.color_palette('bright', n_colors=len(classes))
        labels_map = self._miniSOM.labels_map(xs, ys)

        for bmu, value in labels_map.items():
            for class_label, count in value.items():
                size = 60 / 2 + np.log(1 + count ** 2) * 60
                plt.scatter(bmu[0] + .5, bmu[1] + .5, s=size, color=colors[class_label], alpha=0.8,
                            edgecolors=colors[class_label])

        plt.axis([0, self._miniSOM.get_weights().shape[0], 0, self._miniSOM.get_weights().shape[1]])
        img_path = os.path.join(Constants.PLOT_FOLDER, 'temp', 'som_mapping_{}.png'.format(type_dataset))
        patch_list = []
        for i in range(len(classes)):
            patch = m_patches.Patch(color=colors[i], label=classes[i])
            patch_list.append(patch)

        plt.legend(handles=patch_list, loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 25})
        plt.tight_layout()
        plt.savefig(img_path)
        plt.show()
        plt.close()

    def plot_u_matrix(self):
        """
        Function that plot the Unified Distance Matrix
        """
        umatrix = self._miniSOM.distance_map()
        plt.figure(figsize=(15, 12))
        img_path = os.path.join(Constants.PLOT_FOLDER, 'temp', 'u-matrix.png')
        plt.imshow(umatrix, origin='lower', interpolation='spline36')
        plt.colorbar()
        plt.savefig(img_path)
        plt.show(block=True)

    def accuracy(self, xs, ys):
        """
        Function that compute the accuracy of the dataset passed as param

        :param xs: data
        :param ys: labels associated to data
        :return: the accuracy score
        """
        labels_map = self._miniSOM.labels_map(xs, ys)
        bmu_inputs = [labels_map[self._miniSOM.winner(x)] for x in xs]
        print(bmu_inputs)
        results = [max(bmu.items(), key=operator.itemgetter(1))[0] for bmu in bmu_inputs]
        accuracy_result = accuracy_score(ys, results)
        print('Accuracy = {}'.format(accuracy_result))
        return accuracy_result

    def get_activations(self, xs):
        return self._som.get_activations(xs)
        # actvs = []
        # print(np.shape(self._som._weightages))
        # weights = self._som._weightages.tolist()
        # for i in range(0, np.shape(weights)[0]):
        #     actv = self._miniSOM.quantization_error_from_neuron(xs, weights[i])
        #     actvs.append(math.exp(-(np.sum(actv) / np.shape(actv)[0]) / 0.6))
        # return old_som_acts, []
