import math
import operator
import os

from matplotlib.gridspec import GridSpec
from minisom import MiniSom
from sklearn.metrics import accuracy_score

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from models.som.SOM import SOM
from matplotlib import pyplot as plt

import seaborn as sb
import numpy as np
import matplotlib.patches as m_patches
import matplotlib

from utils.constants import Constants

FIG_SIZE = (12, 8)


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
        matplotlib.rcParams.update({'font.size': 13})

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
        plt.figure(figsize=FIG_SIZE)
        plt.xticks(np.arange(0, self._n, step=1))
        plt.yticks(np.arange(0, self._m, step=1))
        plt.title('Input\'s BMUs activations')
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

        plt.legend(handles=patch_list, loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 16})
        plt.tight_layout()
        plt.savefig(img_path)
        plt.show()
        plt.close()

    def plot_u_matrix(self):
        """
        Function that plot the Unified Distance Matrix
        """
        umatrix = self._miniSOM.distance_map()
        plt.figure(figsize=FIG_SIZE)
        img_path = os.path.join(Constants.PLOT_FOLDER, 'temp', 'u-matrix.png')
        plt.imshow(umatrix, origin='lower', interpolation='spline36')
        plt.xticks(np.arange(0, self._n, step=1))
        plt.yticks(np.arange(0, self._m, step=1))
        plt.title('Unified Distance Matrix')
        plt.colorbar()
        plt.savefig(img_path)
        plt.show(block=True)

    def accuracy(self, xs, ys, type='audio'):
        """
        Function that compute the accuracy of the dataset passed as param

        :param xs: data
        :param ys: labels associated to data
        :return: the accuracy score
        """
        labels_map = self._miniSOM.labels_map(xs, ys)
        bmu_inputs = [labels_map[self._miniSOM.winner(x)] for x in xs]
        results = [max(bmu.items(), key=operator.itemgetter(1))[0] for bmu in bmu_inputs]
        accuracy_result = accuracy_score(ys, results)
        print('Accuracy SOM {} = {}'.format(type, accuracy_result))
        return accuracy_result

    def get_activations(self, xs):
        return self._som.get_activations(xs)

    def plot_activation_frequencies(self, xs):
        plt.figure(figsize=FIG_SIZE)
        plt.title('BMUs Activation Frequencies')
        frequencies = self._miniSOM.activation_response(xs)
        plt.pcolor(frequencies, cmap='Blues')
        plt.xticks(np.arange(0, self._n, step=1))
        plt.yticks(np.arange(0, self._m, step=1))
        plt.colorbar()
        plt.show()

    def neuron_locations(self):
        positions = []
        for i in range(self._m):
            for j in range(self._n):
                positions.append((i, j))
        return positions

    def _draw_pie(self, dists, xpos, ypos, size=None, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        # for incremental pie slices
        cumsum = np.cumsum(dists)
        cumsum = cumsum / cumsum[-1]
        pie = [0] + cumsum.tolist()

        for r1, r2 in zip(pie[:-1], pie[1:]):
            angles = np.linspace(2 * np.pi * r1, 2 * np.pi * r2)
            x = [0] + np.cos(angles).tolist()
            y = [0] + np.sin(angles).tolist()

            xy = np.column_stack([x, y])
            if size is None:
                ax.scatter([xpos], [ypos], marker=xy)
            else:
                ax.scatter([xpos], [ypos], marker=xy, s=size)

        return ax

    def plot_som_class_pies(self, xs, ys, type_dataset='train'):
        labels_map = self._miniSOM.labels_map(xs, ys)
        label_names = np.unique(ys)

        plt.figure(figsize=(10, 8))
        plt.title('Input\'s BMUs activations')
        the_grid = GridSpec(self._n + 1, self._m + 1)
        for position in labels_map.keys():
            label_fracs = [labels_map[position][l] for l in label_names]
            plt.subplot(the_grid[position[0], position[1]], aspect=1)
            patches, texts = plt.pie(label_fracs)
        plt.subplot(the_grid[self._n, :])
        plt.axis('off')
        plt.legend(patches, Constants.label_classes, loc='upper center', bbox_to_anchor=(0.5, 0), ncol=5)
        # plt.savefig('resulting_images/som_iris_pies.png')
        plt.show()

    def plot_quantization_error(self):
        error = self._miniSOM.q_error_pca_init
        iterations = self._miniSOM.iter_x
        plt.plot(iterations, error)
        plt.ylabel('quantization error')
        plt.xlabel('iteration index')
        plt.show()
