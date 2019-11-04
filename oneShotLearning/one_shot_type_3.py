import os
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import scale, StandardScaler

from SelfOrganizingMaps.SelfOrganizingMap import SelfOrganizingMap
from models.som.HebbianModel import HebbianModel
from models.som.SOM import SOM
from oneShotLearning.utility import *
from oneShotLearning.clean_folders import clean_statistics_folders
from utils.constants import Constants
from utils.utils import from_npy_visual_data

if __name__ == '__main__':
    v_xs, v_ys, a_xs, a_ys, filenames_visual, filenames_audio = import_data('visual_10classes_train_cb.npy',
                                                                            Constants.audio_data_path, segmented=True)
    # v_xs, v_ys, _ = from_npy_visual_data(os.path.join(Constants.DATA2_FOLDER, 'visual_10classes_train_as.npy'))
    # v_xs, v_ys, _ = from_npy_visual_data(os.path.join(Constants.DATA2_FOLDER, 'visual_10classes_train_cb.npy'))

    v_dim = np.shape(v_xs)[1]
    a_dim = np.shape(a_xs)[1]

    h_a_xs_train, h_a_ys_train, a_xs_test, a_ys_test = get_random_classes(a_xs, a_ys, Constants.classes, 10, -1)
    h_v_xs_train, h_v_ys_train, v_xs_test, v_ys_test = get_random_classes(v_xs, v_ys, Constants.classes, 10, -1)

    v_ys = np.array(v_ys)
    a_ys = np.array(a_ys)

    alpha, sigma = 0.3, 1
    n_iters = 5000
    n, m = 20, 20
    scaler = 'Standard Scaler'
    splits_v, splits_a = 7, 3

    clean_statistics_folders()

    for i in range(1, 6):
        kf_v = StratifiedKFold(n_splits=splits_v, shuffle=True, random_state=10)
        kf_a = StratifiedKFold(n_splits=splits_a, shuffle=True, random_state=10)

        sigma = i
        print('\n\n ++++++++++++++++++++++++++++++ TEST SIGMA={} ++++++++++++++++++++++++++++++ \n\n'.format(sigma))

        iterations = 1
        accuracy_list, accuracy_list_source_a = [], []
        precisions, precisions_a = [], []
        recalls, recalls_a = [], []
        f1_scores, f1_scores_a = [], []
        for train_index, test_index in kf_v.split(v_xs, v_ys):
            som_v = SelfOrganizingMap(n, m, v_dim, n_iterations=n_iters, sigma=sigma, learning_rate=alpha,
                                      neighborhood_function='gaussian')

            v_xs_train, v_xs_test = v_xs[np.array(train_index)], v_xs[np.array(test_index)]
            v_ys_train, v_ys_test = v_ys[np.array(train_index)], v_ys[np.array(test_index)]
            for a_train_index, a_test_index in kf_a.split(a_xs, a_ys):
                som_a = SelfOrganizingMap(n, m, a_dim, n_iterations=n_iters, sigma=sigma, learning_rate=alpha,
                                          neighborhood_function='gaussian')

                print('\n ***** TEST {}/{} *****\n'.format(iterations, splits_v * splits_a))
                a_xs_train, a_xs_test = a_xs[np.array(a_train_index)], a_xs[np.array(a_test_index)]
                a_ys_train, a_ys_test = a_ys[np.array(a_train_index)], a_ys[np.array(a_test_index)]
                som_v.train(v_xs, pca_initialization_weights=True)
                som_a.train(a_xs, pca_initialization_weights=True)
                hebbian_model = HebbianModel(som_a, som_v, a_dim, v_dim, n_presentations=10)
                hebbian_model.train(h_a_xs_train, h_v_xs_train)

                precision_v, recall_v, f1_score_v = hebbian_model.compute_recall_precision_fscore(a_xs_test, v_xs_test,
                                                                                                  a_ys_test, v_ys_test,
                                                                                                  source='v')
                precision_a, recall_a, f1_score_a = hebbian_model.compute_recall_precision_fscore(a_xs_test, v_xs_test,
                                                                                                  a_ys_test, v_ys_test,
                                                                                                  source='a')
                accuracy_test_v = hebbian_model.evaluate(a_xs_test, v_xs_test, a_ys_test, v_ys_test, source='v',
                                                         prediction_alg='regular')
                accuracy_test_a = hebbian_model.evaluate(a_xs_test, v_xs_test, a_ys_test, v_ys_test, source='a',
                                                         prediction_alg='regular')

                precisions.append(precision_v), recalls.append(recall_v), f1_scores.append(f1_score_v)
                precisions_a.append(precision_a), recalls_a.append(recall_a), f1_scores_a.append(f1_score_a)
                accuracy_list.append(accuracy_test_v), accuracy_list_source_a.append(accuracy_test_a)

                print('--> Accuracy Test Set (Source \'Visual\'): {}'.format(accuracy_test_v))
                print('\tPrecision={}, Recall={}, F1_score={}'.format(precision_v, recall_v, f1_score_v))
                print('--> Accuracy Test Set (Source \'Audio\'): {}'.format(accuracy_test_a))
                print('\tPrecision={}, Recall={}, F1_score={}'.format(precision_a, recall_a, f1_score_a))
                iterations += 1

        mean_accuracy, mean_variance = np.mean(accuracy_list),np.var(accuracy_list)
        mean_accuracy_a, mean_variance_a = np.mean(accuracy_list_source_a), np.var(accuracy_list_source_a)
        mean_precision_v, mean_precision_a = np.mean(precisions), np.mean(precisions_a)
        mean_recall_v, mean_recall_a = np.mean(recalls), np.mean(recalls_a)
        mean_f1_scores_v, mean_f1_scores_a = np.mean(f1_scores), np.mean(f1_scores_a)

        write_md_file(os.path.join(Constants.PLOT_FOLDER, 'temp', 'Tests.md'), mean_accuracy, mean_accuracy_a,
                      mean_variance, mean_variance_a, scaler,
                      '{}x{}'.format(n, m), alpha, sigma, n_iters, splits_v, splits_a, 'Test (Sigma {})'.format(sigma),
                      [mean_precision_v, mean_precision_a], [mean_recall_v, mean_recall_a],
                      [mean_f1_scores_v, mean_f1_scores_a])

        print('\n\nMean Accuracy (source visual)={} (var={}) on v_kf={} and a_kf={} '.format(np.mean(accuracy_list),
                                                                                             np.var(accuracy_list),
                                                                                             kf_v.get_n_splits(),
                                                                                             kf_a.get_n_splits()))
        print('Mean Accuracy (source audio)={} (var={}) on v_kf={} and a_kf={} '.format(np.mean(accuracy_list_source_a),
                                                                                        np.var(accuracy_list_source_a),
                                                                                        kf_v.get_n_splits(),
                                                                                        kf_a.get_n_splits()))
