import math
import operator
import os

from matplotlib.gridspec import GridSpec
from SelfOrganizingMaps.miniSOM import MiniSom
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
        self._som = SOM(n, m, dim, alpha=learning_rate, sigma=sigma, n_iterations=n_iterations, batch_size=1, data=data)
        self._n = n
        self._m = m
        self._dim = dim
        self._n_iterations = n_iterations
        self._learning_rate = learning_rate
        self._sigma = sigma
        self._neighborhood_function = neighborhood_function

    def set_params(self, learning_rate=None, sigma=None, n_iterations=None):
        if learning_rate is not None:
            self._learning_rate = learning_rate
        if sigma is not None:
            self._sigma = sigma
        if n_iterations is not None:
            self._n_iterations = n_iterations

    def train(self, xs, verbose=True, pca_initialization_weights=True, weights=None):
        """
        Function that train the SOM

        :param xs: training set
        :param verbose: if true print the status of the training at each step
        """
        if pca_initialization_weights and weights is None:
            self._miniSOM.pca_weights_init(xs)
        if weights is not None:
            self._miniSOM.set_weights(weights)
        self._miniSOM.train_random(xs, self._n_iterations, verbose=verbose)
        weights = self._miniSOM.get_weights().transpose(2, 0, 1).reshape(np.shape(xs)[1], -1).transpose()
        self._set_weights(weights)
        self._som.set_weights(weights)

    def _set_weights(self, weights):
        self._weightages = weights

    def get_weights(self):
        return self._miniSOM.get_weights()

    def get_dimensions(self):
        return self._n, self._m

    def labels_map(self, xs, ys):
        return self._miniSOM.labels_map(xs, ys)

    def plot_som(self, xs, ys, type_dataset='train'):
        """
        Function that plot the SOM

        :param xs: list of elements
        :param ys: label of the elements
        :param type_dataset: label 'train' or 'test' used to pathfile
        """
        plt.figure(figsize=FIG_SIZE)
        plt.title('Input\'s BMUs activations', fontsize=20)
        plt.xlim([-1, self._m])
        plt.ylim([-1, self._n])
        plt.gca().set_xticks(np.arange(-1, self._m, 1))
        plt.gca().set_yticks(np.arange(-1, self._n, 1))
        plt.gca().set_xticklabels([])
        plt.gca().set_yticklabels([])
        plt.gca().tick_params(axis=u'both', which=u'both', length=0)
        plt.gca().grid(alpha=0.2, linestyle=':', color='black')

        classes = Constants.label_classes
        colors = sb.color_palette('bright', n_colors=len(classes))
        labels_map = self._miniSOM.labels_map(xs, ys)

        for bmu, value in labels_map.items():
            for class_label, count in value.items():
                size = 60 / 2 + np.log(1 + count ** 2) * 60
                plt.scatter(bmu[1] + .5, bmu[0] + .5, s=size, color=colors[class_label], alpha=0.8,
                            edgecolors=colors[class_label])

        plt.axis([0, self._miniSOM.get_weights().shape[1], 0, self._miniSOM.get_weights().shape[0]])
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

    def plot_u_matrix(self, name='som'):
        """
        Function that plot the Unified Distance Matrix
        """
        umatrix = self._miniSOM.distance_map()
        plt.figure(figsize=FIG_SIZE)
        plt.xlim([-1, self._m])
        plt.ylim([-1, self._n])
        plt.gca().set_xticks(np.arange(-1, self._m, 1))
        plt.gca().set_yticks(np.arange(-1, self._n, 1))
        plt.gca().set_xticklabels([])
        plt.gca().set_yticklabels([])
        plt.gca().tick_params(axis=u'both', which=u'both', length=0)
        img_path = os.path.join(Constants.PLOT_FOLDER, 'temp', 'u-matrix_{}.png'.format(name))
        plt.imshow(umatrix, origin='higher', interpolation='spline36')
        plt.title('Unified Distance Matrix', fontsize=20)
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
        """
        Function that plot a chart of frequencies activation of the BMUs
        :param xs: data used to compute the activation frequencies
        :return:
        """
        plt.figure(figsize=FIG_SIZE)
        plt.title('BMUs Activation Frequencies', fontsize=20)
        plt.xlim([-1, self._m])
        plt.ylim([-1, self._n])
        plt.gca().set_xticks(np.arange(-1, self._m, 1))
        plt.gca().set_yticks(np.arange(-1, self._n, 1))
        plt.gca().set_xticklabels([])
        plt.gca().set_yticklabels([])
        plt.gca().tick_params(axis=u'both', which=u'both', length=0)
        frequencies = self._miniSOM.activation_response(xs)
        plt.pcolor(frequencies, cmap='Greens')
        plt.colorbar()
        img_path = os.path.join(Constants.PLOT_FOLDER, 'temp', 'activation-frequencies.png')
        plt.savefig(img_path)
        plt.show()

    def neuron_locations(self):
        """
        Function that get the location of the neuron on the map

        :return: position of the neurons as list of couple
        """
        positions = []
        for i in range(self._n):
            for j in range(self._m):
                positions.append((i, j))
        return positions

    def plot_som_class_pies(self, xs, ys, type_dataset='train'):
        """
        Function tha plot the SOM so that each neuron is a pie chart
        where each slice is a class mapped on that neuron

        :param xs: data list
        :param ys: label associated to the data list
        :param type_dataset: 'train' or 'test' type of data
        """
        labels_map = self._miniSOM.labels_map(xs, ys)
        label_names = np.unique(ys)

        plt.figure(figsize=(10, 8), constrained_layout=False)
        the_grid = GridSpec(self._n + 1, self._m)
        for position in labels_map.keys():
            label_fracs = [labels_map[position][l] for l in label_names]
            plt.subplot(the_grid[self._n - 1 - position[0], position[1]])
            patches, texts = plt.pie(label_fracs)
        plt.subplot(the_grid[self._n, :])
        plt.axis('off')
        plt.legend(patches, Constants.label_classes, loc='upper center', bbox_to_anchor=(0.5, 0), ncol=5, fontsize=14)
        img_path = os.path.join(Constants.PLOT_FOLDER, 'temp', 'som_mapping_pies_{}.png'.format(type_dataset))
        plt.savefig(img_path)
        plt.show()

    def plot_quantization_error(self):
        error = self._miniSOM.q_error_pca_init
        iterations = self._miniSOM.iter_x
        plt.plot(iterations, error)
        plt.ylabel('quantization error')
        plt.xlabel('iteration index')
        plt.show()

    def plot_confusion_map(self, xs, ys, classes):
        labels_map = self._miniSOM.labels_map(xs, ys)
        n_classes = len(classes)
        confusion = [len(labels_map[neuron]) / n_classes if len(labels_map[neuron]) > 1 else 0 for neuron in
                     self.neuron_locations()]
        confusion = np.array(confusion).reshape(self._n, -1)
        scaler = len(classes) / float(len(classes) - 1)
        plt.title('SOM Confusion Map', fontsize=15)
        plt.imshow(confusion * scaler, cmap='Oranges', origin="lower", clim=(0.0, 1.0))
        plt.xlim([-1, self._m])
        plt.ylim([-1, self._n])
        plt.gca().set_xticks(np.arange(-1, self._m, 1))
        plt.gca().set_yticks(np.arange(-1, self._n, 1))
        plt.gca().set_xticklabels([])
        plt.gca().set_yticklabels([])
        plt.gca().tick_params(axis=u'both', which=u'both', length=0)
        plt.colorbar()
        plt.show()
