# Copyright 2017 Giorgia Fenoglio, Mattia Cerrato
#
# This file is part of NNsTaxonomicResponding.
#
# NNsTaxonomicResponding is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# NNsTaxonomicResponding is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NNsTaxonomicResponding.  If not, see <http://www.gnu.org/licenses/>.

import tensorflow as tf
import numpy as np
import math
import os
import matplotlib
import sys

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 8})
import matplotlib.pyplot as plt
from utils.constants import Constants
from matplotlib import colors
from scipy.stats import f as fisher_f
from scipy.stats import norm
from profilehooks import profile
import time
import bisect
import pandas as pd


class SOM(object):
    """
    2-D Self-Organizing Map with Gaussian Neighbourhood function
    and linearly decreasing learning rate.
    """

    # To check if the SOM has been trained
    _trained = False

    def __init__(self, m, n, dim, n_iterations=50, alpha=None, sigma=None,
                 tau=0.5, threshold=0.6, batch_size=500, num_classes=100,
                 checkpoint_loc=None, data='audio', sigma_decay='time',
                 lr_decay='time', weights=None):
        """
        Initializes all necessary components of the TensorFlow
        Graph.

        m X n are the dimensions of the SOM. 'n_iterations' should
        should be an integer denoting the number of iterations undergone
        while training.
        'dim' is the dimensionality of the training inputs.
        'alpha' is a number denoting the initial time(iteration no)-based
        learning rate. Default value is 0.3
        'sigma' is the the initial neighbourhood value, denoting
        the radius of influence of the BMU while training. By default, its
        taken to be half of max(m, n).
        """

        # Assign required variables first
        self._m = m
        self._n = n
        if alpha is None:
            self.alpha = 0.3
        else:
            self.alpha = float(alpha)

        if sigma is None:
            self.sigma = max(m, n) / 2.0
        else:
            self.sigma = float(sigma)

        self.tau = tau
        self.threshold = threshold

        self.batch_size = batch_size

        self._n_iterations = abs(int(n_iterations))

        self.data = data

        self.logs_path = Constants.DATA_FOLDER + '/tblogs/' + self.get_experiment_name()

        self.num_classes = num_classes

        # helper structure
        self.neuron_loc_list = list([tuple(loc) for loc in self._neuron_locations(self._m, self._n)])

        self.train_bmu_class_dict = None
        self.test_bmu_class_dict = None

        if not os.path.exists(self.logs_path):
            os.makedirs(self.logs_path)

        if checkpoint_loc is None:
            self.checkpoint_loc = Constants.DATA_FOLDER + '/saved_models/' + self.get_experiment_name()
        else:
            self.checkpoint_loc = checkpoint_loc

        ##INITIALIZE GRAPH
        self._graph = tf.Graph()

        ##POPULATE GRAPH WITH NECESSARY COMPONENTS
        with self._graph.as_default():

            ##VARIABLES AND CONSTANT OPS FOR DATA STORAGE

            # Randomly initialized weightage vectors for all neurons,
            # stored together as a matrix Variable of size [m*n, dim]
            self._weightage_vects = tf.Variable(tf.random_normal(
                [m * n, dim], mean=0, stddev=1))

            if weights is not None:
                self._weightage_vects = tf.Variable(tf.cast(weights, tf.float32))
            # self._weightage_vects = tf.Variable(tf.random_uniform(
            #     [m * n, dim], minval=-1, maxval=1))

            # Matrix of size [m*n, 2] for SOM grid locations
            # of neurons
            self._location_vects = tf.constant(np.array(
                list(self._neuron_locations(m, n))))

            ##PLACEHOLDERS FOR TRAINING INPUTS
            # We need to assign them as attributes to self, since they
            # will be fed in during training

            # The training vectors
            self._vect_input = tf.placeholder("float", [None, dim])
            # Class vectors, useful for computing class compactness as we train
            self._class_input = tf.placeholder("int32", [None])
            # Test vectors and test classes
            self._vect_test = tf.placeholder("float", [None, dim])
            self._class_test = tf.placeholder("int32", [None])
            # Iteration number
            self._iter_input = tf.placeholder("float")
            # Summaries placeholder
            self._train_compactness = tf.placeholder("float")
            self._test_compactness = tf.placeholder("float")
            self._train_population_convergence = tf.placeholder("float")
            self._test_population_convergence = tf.placeholder("float")
            self._train_mean_convergence = tf.placeholder("float")
            self._test_mean_convergence = tf.placeholder("float")
            self._train_var_convergence = tf.placeholder("float")
            self._test_var_convergence = tf.placeholder("float")
            self._avg_delta = tf.placeholder("float")
            self._train_quant_error = tf.placeholder("float")
            self._test_quant_error = tf.placeholder("float")
            self._train_confusion = tf.placeholder("float")
            self._test_confusion = tf.placeholder("float")
            self._train_usage_rate = tf.placeholder("float")
            self._test_usage_rate = tf.placeholder("float")
            self._train_worst_confusion = tf.placeholder("float")
            self._test_worst_confusion = tf.placeholder("float")

            ##SUMMARIES
            train_mean, train_std = tf.nn.moments(self._train_compactness, axes=[0])
            test_mean, test_std = tf.nn.moments(self._test_compactness, axes=[0])
            tf.summary.scalar("Train Mean Compactness", train_mean)
            tf.summary.scalar("Test Mean Compactness", test_mean)
            tf.summary.scalar("Train Compactness Variance", train_std)
            tf.summary.scalar("Test Compactness Variance", test_std)
            tf.summary.scalar("Train Population Convergence", self._train_population_convergence)
            tf.summary.scalar("Test Population Convergence", self._test_population_convergence)
            tf.summary.scalar("Train Mean Convergence", self._train_mean_convergence)
            tf.summary.scalar("Test Mean Convergence", self._test_mean_convergence)
            tf.summary.scalar("Train Var Convergence", self._train_var_convergence)
            tf.summary.scalar("Test Var Convergence", self._test_var_convergence)
            tf.summary.scalar("Average Delta", self._avg_delta)
            tf.summary.scalar("Train Quantization Error", self._train_quant_error)
            tf.summary.scalar("Test Quantization Error", self._test_quant_error)
            tf.summary.scalar("Train Confusion", self._train_confusion)
            tf.summary.scalar("Test Confusion", self._test_confusion)
            tf.summary.scalar("Train Worst Confusion", self._train_worst_confusion)
            tf.summary.scalar("Test Worst Confusion", self._test_worst_confusion)
            tf.summary.scalar("Train BMU Usage Rate", self._train_usage_rate)
            tf.summary.scalar("Test BMU Usage Rate", self._test_usage_rate)

            # will be set when computing the class compactness for the first time
            self.train_inter_class_distance = None
            self.test_inter_class_distance = None

            self.summaries = tf.summary.merge_all()

            ##CONSTRUCT TRAINING OP PIECE BY PIECE
            # Only the final, 'root' training op needs to be assigned as
            # an attribute to self, since all the rest will be executed
            # automatically during training

            bmu_indexes = self._get_bmu(self._vect_input)

            # This will extract the location of the BMU based on the BMU's
            # index. This has dimensionality [batch_size, 2] where 2 is (i, j),
            # the location of the BMU in the map
            bmu_loc = tf.gather(self._location_vects, bmu_indexes)

            # To compute the alpha and sigma values based on iteration
            # number
            learning_rate = 1.0 - tf.div(self._iter_input, tf.cast(self._n_iterations, "float"))
            if sigma_decay == 'constant':
                _sigma_op = self.sigma
            else:
                _sigma_op = self.alpha * learning_rate
            if lr_decay == 'constant':
                _alpha_op = self.alpha
            else:
                _alpha_op = self.alpha * learning_rate

            # Construct the op that will generate a vector with learning
            # rates for all neurons, based on iteration number and location
            # wrt BMU.

            # Tensor of shape [batch_size, num_neurons] containing the distances
            # between the BMU and all other neurons, for each batch
            bmu_distance_squares = self._get_bmu_distances(bmu_loc)

            neighbourhood_func = tf.exp(tf.negative(tf.div(tf.cast(
                bmu_distance_squares, "float32"), tf.pow(_sigma_op, 2))))
            learning_rate_op = _alpha_op * neighbourhood_func

            # Finally, the op that will use learning_rate_op to update
            # the weightage vectors of all neurons based on a particular
            # input
            learning_rate_matrix = _alpha_op * neighbourhood_func

            self.weightage_delta = self._get_weight_delta(learning_rate_matrix)

            new_weightages_op = tf.add(self._weightage_vects,
                                       self.weightage_delta)
            self._training_op = tf.assign(self._weightage_vects,
                                          new_weightages_op)

            # INITIALIZE SESSION
            # uncomment this to run on cpu
            config = tf.ConfigProto(
                device_count={'GPU': 0}
            )
            self._sess = tf.Session(config=config)
            # self._sess = tf.Session()

            # INITIALIZE VARIABLES
            init_op = tf.global_variables_initializer()
            self._sess.run(init_op)

            self._random_initial_weights = np.copy(self._weightage_vects.eval(session=self._sess))

    def _get_weight_delta(self, learning_rate_matrix):
        """
        """
        diff_matrix = tf.cast(tf.expand_dims(self._vect_input, 1) - self._weightage_vects, "float32")
        mul = tf.expand_dims(learning_rate_matrix, 2) * diff_matrix
        delta = tf.reduce_mean(mul, 0)
        return delta

    def _get_bmu_distances(self, bmu_loc):
        """
        """
        squared_distances = tf.reduce_sum((self._location_vects - tf.expand_dims(bmu_loc, 1)) ** 2, 2)
        return squared_distances

    def set_weights(self, weights):
        self._weightage_vects = weights
        self._weightages = weights
        self._locations = np.array(list(self._neuron_locations(self._m, self._n)))

    def _get_bmu(self, vects):
        """
        Returns the BMU for each example in vect. The return value's dimensionality
        is therefore vect.shape[0]
        """
        squared_differences = (self._weightage_vects - tf.expand_dims(vects, 1)) ** 2
        squared_distances = tf.reduce_sum(squared_differences, 2)
        bmu_index = tf.argmin(squared_distances, 1)
        return bmu_index

    def _neuron_locations(self, m, n):
        """
        Yields one by one the 2-D locations of the individual neurons
        in the SOM.
        """
        # Nested iterations over both dimensions
        # to generate all 2-D locations in the map
        for i in range(m):
            for j in range(n):
                yield np.array([i, j])

    def train(self, input_vects, input_classes=None, test_vects=None, test_classes=None,
              logging=True, save_every=40):
        """
        Trains the SOM.
        'input_vects' should be an iterable of 1-D NumPy arrays with
        dimensionality as provided during initialization of this SOM.
        Current weightage vectors for all neurons(initially random) are
        taken as starting conditions for training.
        """
        with self._sess:
            saver = tf.train.Saver(max_to_keep=int(np.ceil(self._n_iterations / save_every)))
            summary_writer = tf.summary.FileWriter(self.logs_path)

            # path = os.path.join(Constants.DATA_FOLDER, 'plots', 'temp', '{}_weights.csv'.format(self.data))
            # pd.DataFrame(self._weightage_vects.eval()).to_csv(path)

            old_train_comp = [0]
            old_test_comp = [0]
            for iter_no in range(self._n_iterations):
                if iter_no % 5 == 0:
                    print('Iteration {}'.format(iter_no))
                    # delta sanity check
                    delta = self._sess.run(self.weightage_delta, feed_dict={self._vect_input: input_vects[0:1],
                                                                            self._iter_input: iter_no})
                    assert not np.any(np.isnan(delta))
                count = 0
                avg_delta = []
                num_batches = int(np.ceil(len(input_vects) / self.batch_size))
                for i in range(num_batches):
                    count = count + 1
                    start = self.batch_size * i
                    end = self.batch_size * (i + 1)
                    _, a = self._sess.run([self._training_op, self.weightage_delta],
                                          feed_dict={self._vect_input: input_vects[start:end],
                                                     self._iter_input: iter_no})
                    avg_delta.append(np.mean(a))
                    check_arr = a < 1e-28
                    if np.all(check_arr):
                        print('Warning: training seems to have converged - deltas extremely low.')
                    break
                avg_delta = np.mean(avg_delta)

                # Store a centroid grid for easy retrieval later on
                centroid_grid = [[] for i in range(self._m)]
                self._weightages = list(self._sess.run(self._weightage_vects))
                self._locations = list(self._sess.run(self._location_vects))

                if iter_no % save_every == 0:
                    if logging == True:
                        # Run summaries
                        if input_classes is not None:
                            train_comp = old_train_comp
                            train_comp, train_confusion, train_worst_confusion, train_usage_rate = self.compactness_stats(
                                input_vects, input_classes)
                            # print('Train compactness: {}'.format(np.mean(train_comp)))
                            # print('Train confusion: {}'.format(train_confusion))
                            old_train_comp = train_comp
                            train_mean_conv, train_var_conv, train_conv = self.population_based_convergence(input_vects)
                            train_quant_error = self.quantization_error(input_vects)
                            # train_quant_error = [0]
                            # print(train_conv)
                        else:
                            train_comp = [0]
                            train_conv = [0]
                        if test_classes is not None:
                            test_comp = old_test_comp
                            test_comp, test_confusion, test_worst_confusion, test_usage_rate = self.compactness_stats(
                                test_vects, test_classes, train=False)
                            # print('Test compactness: {}'.format(np.mean(test_comp)))
                            # print('Test confusion: {}'.format(test_confusion))
                            old_test_comp = test_comp
                            test_mean_conv, test_var_conv, test_conv = self.population_based_convergence(test_vects)
                            test_quant_error = self.quantization_error(test_vects)
                            # test_quant_error = [0]
                            # print(test_conv)
                        else:
                            test_comp = [0]
                            test_conv = [0]
                        summary = self._sess.run(self.summaries,
                                                 feed_dict={self._train_compactness: train_comp,
                                                            self._test_compactness: test_comp,
                                                            self._train_population_convergence: train_conv,
                                                            self._test_population_convergence: test_conv,
                                                            self._train_mean_convergence: train_mean_conv,
                                                            self._test_mean_convergence: test_mean_conv,
                                                            self._train_var_convergence: train_var_conv,
                                                            self._test_var_convergence: test_var_conv,
                                                            self._avg_delta: avg_delta,
                                                            self._train_confusion: train_confusion,
                                                            self._test_confusion: test_confusion,
                                                            self._train_quant_error: train_quant_error,
                                                            self._test_quant_error: test_quant_error,
                                                            self._train_usage_rate: train_usage_rate,
                                                            self._test_usage_rate: test_usage_rate,
                                                            self._train_worst_confusion: train_worst_confusion,
                                                            self._test_worst_confusion: test_worst_confusion
                                                            })
                        summary_writer.add_summary(summary, global_step=iter_no)

                # Save model periodically
                if iter_no % save_every == 0:
                    dirpath = self.checkpoint_loc + '_' + str(iter_no) + '_epoch' + os.sep
                    if not os.path.exists(self.checkpoint_loc):
                        os.makedirs(self.checkpoint_loc)
                    path = dirpath + 'model'
                    # print('Saving in {}'.format(path))
                    # saver.save(self._sess, path, global_step=iter_no)
            for i, loc in enumerate(self._locations):
                centroid_grid[loc[0]].append(self._weightages[i])
            self._centroid_grid = centroid_grid

            self._trained = True

            # Save the final model
            dirpath = self.checkpoint_loc + '_final' + os.sep
            if not os.path.exists(self.checkpoint_loc):
                os.makedirs(self.checkpoint_loc)
            path = dirpath + 'model'
            # print('Saving in {}'.format(path))
            # saver.save(self._sess, path)

    def restore_trained(self, path):
        """
        path should be the folder containing the checkpoints.
        """
        ckpt = tf.train.get_checkpoint_state(path)
        if ckpt and ckpt.model_checkpoint_path:
            with self._sess:
                saver = tf.train.Saver()
                saver.restore(self._sess, ckpt.model_checkpoint_path)

                # restore usefull variable
                centroid_grid = [[] for i in range(self._m)]
                self._weightages = list(self._sess.run(self._weightage_vects))
                self._locations = list(self._sess.run(self._location_vects))
                for i, loc in enumerate(self._locations):
                    centroid_grid[loc[0]].append(self._weightages[i])
                self._centroid_grid = centroid_grid

                self._trained = True

                print('RESTORED SOM MODEL')
                return True
        else:
            print('NO CHECKPOINT FOUND')
            return False

    def get_experiment_name(self):
        return str(self.data) + '_' + str(self._m) + 'x' + str(self._n) + '_tau' + str(self.tau) + '_thrsh' \
               + str(self.threshold) + '_sigma' + str(self.sigma) + '_batch' + str(self.batch_size) \
               + '_alpha' + str(self.alpha)

    def get_centroids(self):
        """
        Returns a list of 'm' lists, with each inner list containing
        the 'n' corresponding centroid locations as 1-D NumPy arrays.
        """
        if not self._trained:
            raise ValueError("SOM not trained yet")
        return self._centroid_grid

    def map_vects(self, input_vects):
        """
        Maps each input vector to the relevant neuron in the SOM
        grid.
        'input_vects' should be an iterable of 1-D NumPy arrays with
        dimensionality as provided during initialization of this SOM.
        Returns a list of 1-D NumPy arrays containing (row, column)
        info for each input vector(in the same order), corresponding
        to mapped neuron.
        """

        to_return = []
        for vect in input_vects:
            min_index = min([i for i in range(len(self._weightages))],
                            key=lambda x: np.linalg.norm(vect -
                                                         self._weightages[x]))
            to_return.append(self._locations[min_index])
        return to_return

    # @profile
    def get_BMU(self, input_vect):
        min_index = min([i for i in range(len(self._weightages))],
                        key=lambda x: np.linalg.norm(input_vect -
                                                     self._weightages[x]))
        return [min_index, self._locations[min_index]]

    # @profile
    def get_BMU_mine(self, input_vect):
        diff = np.linalg.norm(self._weightages - input_vect, axis=1)
        min_index = np.argmin(diff)
        return [min_index, self._locations[min_index]]

    # @profile
    def map_vects_parallel(self, input_vects):
        input_vects = input_vects[:, np.newaxis, :]
        diff_tensor = self._weightages - input_vects
        diff_tensor = np.linalg.norm(diff_tensor, axis=2)
        min_indexes = np.argmin(diff_tensor, axis=1).tolist()
        result = []
        for index in min_indexes:
            result.append(self._locations[index])
        return result

    # @profile
    def map_vects_memory_aware(self, input_vects):
        result = []
        for x in input_vects:
            _, bmu_loc = self.get_BMU_mine(x)
            result.append(bmu_loc)
        return result

    # def map_vects_confusion(self, input_vects, ys):
        result = []
        collapse_dict = {neuron_loc: [] for neuron_loc in self.neuron_loc_list}
        for index, x in enumerate(input_vects):
            _, bmu_loc = self.get_BMU_mine(x)
            collapse_dict[tuple(bmu_loc)].append(ys[index])
            result.append(bmu_loc)
        bmu_confusion = 0
        real_bmu_counter = 0
        for bmu_loc, class_list in collapse_dict.items():
            if class_list == []:
                continue
            bmu_confusion += len(set(class_list))
            real_bmu_counter += 1
        bmu_confusion /= real_bmu_counter
        bmu_confusion /= self.num_classes
        return result, bmu_confusion

    # def map_vects_get_confusion_stats(self, input_vects, ys):
        result = []
        collapse_dict = {neuron_loc: [] for neuron_loc in self.neuron_loc_list}
        worst_offenders = [0 for i in range(1)]
        for index, x in enumerate(input_vects):
            _, bmu_loc = self.get_BMU_mine(x)
            collapse_dict[tuple(bmu_loc)].append(ys[index])
            result.append(bmu_loc)
        bmu_confusion = 0
        real_bmu_counter = 0
        for bmu_loc, class_list in collapse_dict.items():
            if class_list == []:
                continue
            num_classes = len(set(class_list))
            bmu_confusion += num_classes
            if num_classes > worst_offenders[0]:
                bisect.insort(worst_offenders, num_classes)
                if len(worst_offenders) >= (self._m * self._n / 20):
                    worst_offenders = worst_offenders[1:]
            real_bmu_counter += 1
        bmu_confusion /= real_bmu_counter
        bmu_confusion /= self.num_classes
        bmu_usage_rate = real_bmu_counter / (self._m * self._n)
        bmu_confusion_worst = np.mean(worst_offenders) / num_classes
        return result, bmu_confusion, bmu_confusion_worst, bmu_usage_rate

    def detect_superpositions(self, l):
        for l_i in l:
            if len(l_i) > 1:
                if all(x == l_i[0] for x in l_i) == False:
                    return True
        return False

    def print_som_evaluation(self, input_vects, ys):
        bmu_positions = self.map_vects_memory_aware(input_vects)
        class_comp = self.compactness_stats(input_vects, ys, train=False)
        _, confusion = self.map_vects_confusion(input_vects, ys)
        print('Average Compactness: {}'.format(np.mean(class_comp)))
        print('Compactness Variance: {}'.format(np.var(class_comp)))
        print('Confusion: {}'.format(np.mean(confusion)))

    def compute_compactness_confusion(self, input_vects, ys):
        """
        Compute compactness and confusion for vector in input
        :param input_vects: data vector used to compute compactness and confusion
        :param ys: labels associated to input_vects
        :return: mean compactness, compactness variance, mean confusion
        """
        class_comp = self.compactness_stats(input_vects, ys, train=False)
        _, confusion = self.map_vects_confusion(input_vects, ys)
        return np.mean(class_comp), np.var(class_comp), np.mean(confusion)

    def neuron_collapse(self, input_vects, bmu_positions=None):
        if bmu_positions == None:
            bmu_positions = self.map_vects_memory_aware(input_vects)
        bmu_positions = [tuple(bmu_pos) for bmu_pos in bmu_positions]
        ratio_examples = len(set(bmu_positions)) / len(input_vects)
        ratio_neurons = len(set(bmu_positions)) / (self._m * self._n)
        print('Detected {} unique BMUs. \n Ratio of {} over the examples; ratio of {} over the number of neurons' \
              .format(len(set(bmu_positions)), ratio_examples, ratio_neurons))
        return ratio_examples, ratio_neurons

    def neuron_collapse_classwise(self, input_vects, ys, bmu_positions=None, ):
        if bmu_positions == None:
            bmu_positions = self.map_vects_memory_aware(input_vects)
        bmu_positions = [tuple(bmu_pos) for bmu_pos in bmu_positions]
        class_belonging_dict = {y: [] for y in list(set(ys))}
        classwise_collapse = [0 for y in list(set(ys))]
        classwise_bmus = [[] for y in list(set(ys))]
        all_neurons = list(self._neuron_locations(self._m, self._n))
        all_neurons = [tuple(bmu_pos) for bmu_pos in all_neurons]
        all_neurons = {neuron_position: [] for neuron_position in all_neurons}
        for i, y in enumerate(ys):
            class_belonging_dict[y].append(i)
        for y in set(ys):
            class_xs = [input_vects[i] for i in class_belonging_dict[y]]
            class_bmu_positions = self.map_vects_memory_aware(class_xs)
            class_bmu_positions = [tuple(class_bmu_pos) for class_bmu_pos in class_bmu_positions]
            for bmu_pos in class_bmu_positions:
                all_neurons[bmu_pos].append(y)
            class_unique_bmus = list(set(class_bmu_positions))
            classwise_bmus[y] = class_unique_bmus
            classwise_collapse[y] = len(class_unique_bmus)
        non_bmu_positions = list(self._neuron_locations(self._m, self._n))
        non_bmu_positions = [tuple(non_bmu_pos) for non_bmu_pos in non_bmu_positions]
        for bmu_position in bmu_positions:
            try:
                non_bmu_positions.remove(bmu_position)
            except ValueError:
                pass
        mindiff = sys.maxsize
        reference_histogram = np.array([1 / self.num_classes for i in list(range(self.num_classes))])
        histograms = []
        worst_histogram = []
        for neuron, class_list in all_neurons.items():
            if class_list == []:
                continue
            class_occurrence_histogram, _ = np.histogram(class_list, bins=list(range(self.num_classes + 1)),
                                                         density=True)
            histograms.append(class_occurrence_histogram)
            diff = np.linalg.norm(class_occurrence_histogram - reference_histogram)
            if diff < mindiff:
                mindiff = diff
                worst_histogram = class_occurrence_histogram
        print('Worst histogram: {}'.format(worst_histogram))
        print('Mean histogram: {}'.format(np.mean(histograms, axis=0)))
        print('Mean overlap: {}'.format(np.mean(histograms)))
        print('Non-BMU neurons: {}, ratio over the neurons {}'.format(len(non_bmu_positions),
                                                                      len(non_bmu_positions) / (self._m * self._n)))
        print('Per-class unique BMUs: {}'.format(classwise_collapse))
        print('If there was no class overlap, the non-BMU neurons would be {}'.format(
            self._m * self._n - sum(classwise_collapse)))
        print('Overlap index: {}'.format(1 - (self._m * self._n - sum(classwise_collapse)) / len(non_bmu_positions)))
        print('Average BMUs for each class: {}'.format(np.mean(classwise_collapse)))
        print('Average ratio over the examples: {}'.format(np.mean(classwise_collapse) / len(input_vects)))
        print('Average ratio over the neurons: {}'.format(np.mean(classwise_collapse) / (self._m * self._n)))
        return classwise_collapse, classwise_bmus

    #    def memorize_examples_by_class(self, X, y):
    #        self.bmu_class_dict = {i : [] for i in range(self._n * self._m)}
    #        for i, (x, yi) in enumerate(zip(X, y)):
    #            activations, _ = self.get_activations(x, normalize=False, mode='exp', threshold=False)
    #            bmu_index = np.argmax(activations)
    #            self.bmu_class_dict[bmu_index].append(yi)
    #        superpositions = self.detect_superpositions(self.bmu_class_dict.values())
    #        print('More than a class mapped to a neuron: '+ str(superpositions))
    #        return superpositions

    def memorize_examples_by_class(self, xs, ys, train=True):
        bmu_class_dict = {i: [] for i in range(self._n * self._m)}
        result = []
        superpositions = False
        for i, (x, yi) in enumerate(zip(xs, ys)):
            bmu_index, bmu_loc = self.get_BMU_mine(x)
            result.append(bmu_index)
            if superpositions == False:
                if len(bmu_class_dict[bmu_index]) > 0 and yi not in bmu_class_dict[bmu_index]:
                    superpositions = True
            bmu_class_dict[bmu_index].append(yi)
        if train:
            self.train_bmu_class_dict = bmu_class_dict
        else:
            self.test_bmu_class_dict = bmu_class_dict
        return superpositions

    def get_activations(self, input_vect, normalize=True, threshold=0.6, mode='exp',
                        tau=0.6):
        # get activations for the word learning

        # Quantization error:
        activations = list()
        pos_activations = list()
        for i in range(len(self._weightages)):
            d = np.array([])
            d = (np.absolute(input_vect - self._weightages[i])).tolist()
            if mode == 'exp':
                activations.append(math.exp(-(np.sum(d) / len(d)) / tau))
            if mode == 'linear':
                activations.append(1 / np.sum(d))
            pos_activations.append(self._locations[i])
        activations = np.array(activations)
        if normalize:
            max_ = max(activations)
            min_ = min(activations)
            activations = (activations - min_) / float(max_ - min_)
        idx = activations < threshold
        activations[idx] = 0
        return [activations, pos_activations]

    def plot_som(self, X, y, plot_name='som-viz.png'):
        image_grid = np.zeros(shape=(self._n, self._m))

        color_names = \
            {0: 'black', 1: 'blue', 2: 'skyblue',
             3: 'aqua', 4: 'darkgray', 5: 'green', 6: 'red',
             7: 'cyan', 8: 'violet', 9: 'yellow'}
        # Map colours to their closest neurons
        mapped = self.map_vects(X)

        # Plot
        plt.imshow(image_grid)
        plt.title('Color SOM')
        for i, m in enumerate(mapped):
            plt.text(m[1], m[0], color_names[y[i]], ha='center', va='center',
                     bbox=dict(facecolor=color_names[y[i]], alpha=0.5, lw=0))
        plt.savefig(os.path.join(Constants.PLOT_FOLDER, plot_name))
        plt.close()

    def compactness_stats(self, xs, ys, train=True, strategy='memory-aware'):
        confusion = [0]
        if strategy == 'memory-aware':
            bmu_positions, confusion, worst_confusion, usage_rate = self.map_vects_get_confusion_stats(xs, ys)
        elif strategy == 'parallel':
            bmu_positions = self.map_vects_parallel(xs)
        else:
            raise ValueError('Unrecognized strategy parameter in class_compactness function.')
            sys.exit(1)
        class_belonging_dict = {y: [] for y in list(set(ys))}
        for i, y in enumerate(ys):
            class_belonging_dict[y].append(i)
        intra_class_distance = [0 for y in list(set(ys))]
        for i, y in enumerate(set(ys)):
            for index, j in enumerate(class_belonging_dict[y]):
                x1 = xs[j]
                for k in class_belonging_dict[y][index + 1:]:
                    # x2 = xs[k]
                    pos_x1 = bmu_positions[j]
                    pos_x2 = bmu_positions[k]
                    intra_class_distance[i] += np.linalg.norm(pos_x1 - pos_x2)
        if train == True:
            inter_class_distance = self.train_inter_class_distance
        else:
            inter_class_distance = self.test_inter_class_distance
        if inter_class_distance is None or inter_class_distance == 0.0:
            inter_class_distance = 0
            for i, x1 in enumerate(xs):
                for j, x2 in enumerate(xs[i + 1:]):
                    pos_x1 = bmu_positions[i]
                    pos_x2 = bmu_positions[j]
                    inter_class_distance += np.linalg.norm(pos_x1 - pos_x2)
            inter_class_distance /= len(xs)
            if train == True:
                self.train_inter_class_distance = inter_class_distance
            else:
                self.test_inter_class_distance = inter_class_distance
        if train == True:
            class_comp = intra_class_distance / self.train_inter_class_distance
            # print("intra {} inter {}".format(intra_class_distance, self.train_inter_class_distance))
        else:
            class_comp = intra_class_distance / self.test_inter_class_distance
        return class_comp, confusion, worst_confusion, usage_rate

    def population_based_convergence(self, xs, alpha=0.05):
        '''
        Population based convergence is a feature-by-feature convergence criterion.
        This implementation is based on "A Convergence Criterion for Self-Organizing
        Maps" by B. Ott, 2012.

        Name mapping from variables to paper:
        data_feature_mean: $x^1$
        neuron_feature_mean: $x^2$
        data_feature_var: $\sigma^2_1$
        neuron_feature_var: $\sigma^2_2$
        num_samples: $n_1$
        num_neurons: $n_2$
        '''
        weights = self._sess.run(self._weightage_vects)
        data_feature_mean = np.mean(xs, axis=0)
        neuron_feature_mean = np.mean(weights, axis=0)
        data_feature_var = np.var(xs, axis=0)
        neuron_feature_var = np.var(weights, axis=0)
        num_samples = len(xs)
        num_neurons = (self._m * self._n)

        z = norm.ppf(q=1 - (alpha / 2))
        lhs = (data_feature_mean - neuron_feature_mean) \
              - z * np.sqrt(data_feature_var / num_samples + neuron_feature_var / num_neurons)  # eq. 17 lhs
        rhs = (data_feature_mean - neuron_feature_mean) \
              + z * np.sqrt(data_feature_var / num_samples + neuron_feature_var / num_neurons)  # eq. 17 rhs

        mean_stat = np.multiply(lhs, rhs)
        mean_pos_converged = np.where(mean_stat[mean_stat < 0])[0]

        # std convergence
        fisher_f_stat = fisher_f.ppf(q=1 - (alpha / 2), dfn=num_samples - 1, dfd=num_neurons - 1)

        lhs = np.divide(data_feature_var, neuron_feature_var) * (1 / fisher_f_stat)
        rhs = np.divide(data_feature_var, neuron_feature_var) * fisher_f_stat

        lhs = (lhs <= 1).astype(int)
        rhs = (rhs >= 1).astype(int)
        var_stat = np.multiply(lhs, rhs)
        var_pos_converged = var_stat[var_stat != 0]

        # print('Mean converged features: {}'.format(len(mean_pos_converged)))
        # print('Current ratio: {}'.format(data_feature_var / neuron_feature_var))
        # print('Average ratio: {}'.format(np.mean(data_feature_var / neuron_feature_var)))
        # print('Variance of ratio: {}'.format(np.var(data_feature_var / neuron_feature_var)))
        # print('Var converged features: {}'.format(len(var_pos_converged)))

        # return normalized values for mean, variance and total convergence
        return len(mean_pos_converged) / len(neuron_feature_mean), \
               len(var_pos_converged) / len(neuron_feature_mean), \
               len(np.intersect1d(mean_pos_converged, var_pos_converged, assume_unique=True)) / len(neuron_feature_mean)

    def quantization_error(self, xs):
        weights = self._sess.run(self._weightage_vects)
        diff = np.sum([np.linalg.norm(weights - x) for x in xs])
        return diff


if __name__ == '__main__':
    pass
