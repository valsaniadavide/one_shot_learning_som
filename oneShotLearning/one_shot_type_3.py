import os
import pandas as pd
import numpy as np

from models.som.HebbianModel import HebbianModel
from models.som.SOM import SOM
from oneShotLearning.utility import *
from oneShotLearning.clean_folders import clean_statistics_folders
from oneShotLearning.input_tests import get_min_max_mean_input_feature
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, AffinityPropagation
from sklearn.metrics import accuracy_score
from utils.constants import Constants
from utils.utils import from_npy_visual_data

if __name__ == '__main__':
    xs, ys, a_xs, a_ys, filenames_visual, filenames_audio = import_data(Constants.visual_data_path,
                                                                        Constants.audio_data_path)
    stats = []
    v_xs, v_ys, _ = from_npy_visual_data(os.path.join(Constants.DATA2_FOLDER, 'visual_10classes_train_as.npy'))
    v_xs = StandardScaler().fit_transform(v_xs)

    a_xs_train, a_ys_train, a_xs_test, a_ys_test = get_random_classes(a_xs, a_ys, Constants.classes, 10, -1)
    v_xs_train, v_ys_train, v_xs_test, v_ys_test = get_random_classes(xs, ys, Constants.classes, 10, -1)

    class_0_v_xs, class_0_v_ys, others_v_xs, others_v_ys = get_examples_of_class(xs, ys, Constants.classes, 1)
    class_0_a_xs, class_0_a_ys, others_a_xs, others_a_ys = get_examples_of_class(a_xs, a_ys, Constants.classes, 1)

    a_dim = len(a_xs[0])
    v_dim = len(v_xs[0])

    name_statistics = ''

    clean_statistics_folders()

    for i in range(0, 1):
        print('shape audio input ', np.shape(others_a_xs))
        print('shape visual input ', np.shape(others_v_xs))
        print('shape audio input labels', np.shape(others_a_ys))
        print('shape visual input labels', np.shape(others_v_ys))

        # random.seed(a=None, version=2)
        # learning_rate = round(random.uniform(0.02, 0.35), 2)
        # sigma = random.randrange(1, 5, 1)
        print('\n ***** TEST N°={} ***** \n'.format(i + 1))
        # som_a = SOM(20, 30, a_dim, alpha=0.3, sigma=15, n_iterations=100, batch_size=1)
        # type_file = 'visual_' + str(i + 1)
        type_file = 'visual'
        som_v = SOM(30, 30, v_dim, alpha=0.3, sigma=15, n_iterations=500, batch_size=1, data=type_file)

        # som_a.train(others_a_xs, input_classes=others_a_ys, test_vects=others_a_xs, test_classes=others_a_ys, save_every=100)
        # print('Trained SOM Audio')
        som_v.train(others_v_xs, input_classes=others_v_ys,test_vects=others_v_xs, test_classes=others_v_ys, save_every=100)
        # print('Trained SOM Visual')

        # som_v.print_som_evaluation(v_xs_test, v_ys_test)
        # som_v.neuron_collapse_classwise(v_xs_test, v_ys_test)
        # print_charts(som_v, v_xs_test, v_ys_test, Constants.label_classes, 'visual', "Visual SOM", 'test_data')
        # print_charts(som_a, a_xs_test, a_ys_test, Constants.label_classes, 'Audio', "Audio SOM", 'test_data')
        print_charts(som_v, others_v_xs, others_v_ys, Constants.label_classes, 'visual', "Visual SOM", 'training_data')
        # print_charts(som_a, others_a_xs, others_a_ys, Constants.label_classes, 'Audio', "Audio SOM", 'training_data')

        v_compact, v_compact_var, v_confus = som_v.compute_compactness_confusion(others_v_xs, others_v_ys)
        # a_compact, a_compact_var, a_confus = som_a.compute_compactness_confusion(others_a_xs, others_a_ys)
        # som_v.plot_som(v_xs_test, v_ys_test, plot_name='som-vis.png')
        # som_a.plot_som(a_xs_test, a_ys_test, plot_name='som-aud.png')
        # v_mean_compactness = np.mean(v_compact)
        # v_mean_variance = np.mean(v_compact_var)
        # a_mean_compactness = np.mean(a_compact)
        # a_mean_variance = np.mean(a_compact_var)
        #
        # df_stats_weights = get_min_max_mean_input_feature(som_v._random_initial_weights)
        # print('Random Initial SOM Weights:\n', df_stats_weights.head(n=10))
        #
        # df_stats_input = get_min_max_mean_input_feature(others_v_xs)
        # print('\nInput Statistics:\n', df_stats_input.head(n=10))
        #
        # hebbian_model = HebbianModel(som_a, som_v, a_dim, v_dim, n_presentations=10)
        # hebbian_model.train(a_xs_train, v_xs_train)
        #
        # accuracy_train = hebbian_model.evaluate(others_a_xs, others_v_xs, others_a_ys, others_v_ys, source='v',
        #                                         prediction_alg='regular')
        # # accuracy_test = hebbian_model.evaluate(a_xs_test, v_xs_test, a_ys_test, v_ys_test, source='v',
        # #                                      prediction_alg='regular')
        #
        # stats.append([som_v._n, som_v._m, 0.3, 15, accuracy_train, 'NaN', v_mean_compactness,
        #               v_mean_variance, a_mean_compactness, a_mean_variance, v_confus, a_confus])
        #
        # print("Accuracy Training set", accuracy_train)
        # # print("Accuracy Test set", accuracy_test)
        # print('test_n={}, n={}, m={}, learning_rate={}, sigma={}, accuracy_train={}, accuracy_test={}, v_compact={}, '
        #       'v_compact_var={}, v_confus={}, a_compact={}, a_compact_var={}, a_confus={} '
        #       .format(i, som_v._n, som_v._m, 0.3, 15, accuracy_train, 'NaN', v_mean_compactness,
        #               v_mean_variance, v_confus, a_mean_compactness, a_mean_variance, a_confus))

    # df = pd.DataFrame(data=stats, columns=Constants.columns_stat)
    # df.head(n=10)
    # df.to_csv(os.path.join(Constants.PLOT_FOLDER, 'temp', 'statistics.csv'))
