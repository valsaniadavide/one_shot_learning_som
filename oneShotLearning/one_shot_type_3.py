import os
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import scale, StandardScaler

from SelfOrganizingMaps.SelfOrganizingMap import SelfOrganizingMap
from models.som.HebbianModel import HebbianModel
from models.som.SOM import SOM
from oneShotLearning.utility import *
from oneShotLearning.clean_folders import clean_statistics_folders
from utils.constants import Constants
from utils.utils import from_npy_visual_data
import matplotlib.pyplot as plt
from scipy.stats import norm

if __name__ == '__main__':
    v_xs_all, v_ys_all, a_xs_all, a_ys_all, filenames_visual, filenames_audio = import_data(
        Constants.visual_data_path_segmented, Constants.audio_data_path, segmented=True)

    v_dim, a_dim = np.shape(v_xs_all)[1], np.shape(a_xs_all)[1]
    print(np.shape(a_xs_all))
    alpha, sigma = 0.4, 3
    n_iters = 5500
    n, m = 20, 30
    scaler = 'Standard Scaler'
    splits_v, splits_a = 7, 3

    clean_statistics_folders()

    # for i in Constants.classes:
    kf_v = StratifiedKFold(n_splits=splits_v, shuffle=True, random_state=10)
    kf_a = StratifiedKFold(n_splits=splits_a, shuffle=True, random_state=10)
    # i = 9 - i
    i = 9
    v_xs_all_train, v_xs_all_test, v_ys_all_train, v_ys_all_test = train_test_split(v_xs_all, v_ys_all,
                                                                                    test_size=0.35,
                                                                                    stratify=v_ys_all)
    a_xs_all_train, a_xs_all_test, a_ys_all_train, a_ys_all_test = train_test_split(a_xs_all, a_ys_all,
                                                                                    test_size=0.35, random_state=42,
                                                                                    stratify=a_ys_all)

    # Extract the class i-th from the train and the test sets
    class_ext_v_xs, class_ext_v_ys, v_xs, v_ys = get_examples_of_class(v_xs_all_train, v_ys_all_train,
                                                                       Constants.classes, i)
    class_ext_a_xs, class_ext_a_ys, a_xs, a_ys = get_examples_of_class(a_xs_all_train, a_ys_all_train,
                                                                       Constants.classes, i)
    _, _, v_xs_n_1_classes_test, v_ys_n_1_classes_test = get_examples_of_class(v_xs_all_test, v_ys_all_test,
                                                                               Constants.classes, i)
    _, _, a_xs_n_1_classes_test, a_ys_n_1_classes_test = get_examples_of_class(a_xs_all_test, a_ys_all_test,
                                                                               Constants.classes, i)

    # Get 10 examples from the extracted class data for one-shot learning
    class_ext_v_xs_train, class_ext_v_ys_train, class_ext_v_xs_test, class_ext_v_ys_test = get_random_classes(
        class_ext_v_xs, class_ext_v_ys, [i], 10, -1)
    class_ext_a_xs_train, class_ext_a_ys_train, class_ext_a_xs_test, class_ext_a_ys_test = get_random_classes(
        class_ext_a_xs, class_ext_a_ys, [i], 10, -1)

    # Create a new dataset of 10 random examples per class used for hebbian training
    h_a_xs_train, h_a_ys_train, _, _ = get_random_classes(a_xs_all_train, a_ys_all_train,
                                                          Constants.classes, 10, -1)
    h_v_xs_train, h_v_ys_train, _, _ = get_random_classes(v_xs_all_train, v_ys_all_train,
                                                          Constants.classes, 10, -1)

    h_a_xs_train_n_1, h_a_ys_train_n_1, h_a_xs_test_n_1, h_a_ys_test_n_1 = get_random_classes(a_xs, a_ys,
                                                                                              Constants.classes[:-1],
                                                                                              10, -1)
    h_v_xs_train_n_1, h_v_ys_train_n_1, h_v_xs_test_n_1, h_v_ys_test_n_1 = get_random_classes(v_xs, v_ys,
                                                                                              Constants.classes[:-1],
                                                                                              10, -1)

    print('\n\n ++++++++++++++++++++++++++++++ TEST ONE SHOT ON CLASS = \'{}\' ++++++++++++++++++++++++++++++ \n\n'
          .format(Constants.label_classes[i]))

    iterations = 1
    accuracy_list, accuracy_list_source_a = [], []
    precisions, precisions_a = [], []
    recalls, recalls_a = [], []
    f1_scores, f1_scores_a = [], []

    # **********************************

    som_v = SelfOrganizingMap(n, m, v_dim, n_iterations=n_iters, sigma=sigma, learning_rate=alpha)
    som_a = SelfOrganizingMap(n, m, a_dim, n_iterations=n_iters, sigma=sigma, learning_rate=alpha)

    weights_v, weights_a, som_v, som_a = train_som_and_get_weight(som_v, som_a, v_xs, a_xs)

    hebbian_model = HebbianModel(som_a, som_v, a_dim, v_dim, n_presentations=10, n_classes=len(Constants.classes) - 1)
    hebbian_model.train(h_a_xs_train_n_1, h_v_xs_train_n_1)
    accuracy_test_v = hebbian_model.evaluate(h_a_xs_test_n_1, h_v_xs_test_n_1, h_a_ys_test_n_1, h_v_ys_test_n_1,
                                             source='v',
                                             prediction_alg='regular')
    accuracy_test_a = hebbian_model.evaluate(h_a_xs_test_n_1, h_v_xs_test_n_1, h_a_ys_test_n_1, h_v_ys_test_n_1,
                                             source='a',
                                             prediction_alg='regular')
    print('\n--> Accuracy Test Set (Source \'Visual\') n-1 classes: {}'.format(accuracy_test_v))
    print('--> Accuracy Test Set (Source \'Audio\') n-1 classes: {}\n'.format(accuracy_test_a))

    # som_v.plot_som_class_pies(v_xs_n_1_classes_test, v_ys_n_1_classes_test, type_dataset='trained_n-1_classes')
    # som_a.plot_som_class_pies(a_xs_n_1_classes_test, a_ys_n_1_classes_test,
    #                           type_dataset='audio_trained_n-1_classes')
    som_v.plot_som(v_xs_n_1_classes_test, v_ys_n_1_classes_test, type_dataset='video_trained_n-1_classes')
    som_a.plot_som(a_xs_n_1_classes_test, a_ys_n_1_classes_test,
                   type_dataset='audio_trained_n-1_classes')
    som_v.plot_u_matrix(name='video_trained_n-1_classes')
    som_a.plot_u_matrix(name='audio_trained_n-1_classes')

    som_one_shot_v = SelfOrganizingMap(n, m, v_dim, n_iterations=300, sigma=0.8, learning_rate=0.1)
    som_one_shot_a = SelfOrganizingMap(n, m, a_dim, n_iterations=300, sigma=0.8, learning_rate=0.1)

    print('--> One Shot training')
    # Train both SOMs with one shot dataset assigning precomputed weights
    # som_v.set_params(learning_rate=0.1, sigma=0.8, n_iterations=50)
    # som_a.set_params(learning_rate=0.1, sigma=0.8, n_iterations=50)
    som_one_shot_v.train(class_ext_v_xs_train, weights=weights_v)
    som_one_shot_a.train(class_ext_a_xs_train, weights=weights_a)

    print('Plotting som')
    # som_one_shot_v.plot_som_class_pies(v_xs_all_test, v_ys_all_test, type_dataset='trained_one_shot')
    # som_one_shot_a.plot_som_class_pies(a_xs_all_test, a_ys_all_test, type_dataset='audio_trained_one_shot')
    som_one_shot_v.plot_som(v_xs_all_test, v_ys_all_test, type_dataset='video_trained_one_shot')
    som_one_shot_a.plot_som(a_xs_all_test, a_ys_all_test, type_dataset='audio_trained_one_shot')

    som_one_shot_a.plot_u_matrix(name='video_trained_one_shot')
    som_one_shot_a.plot_u_matrix(name='audio_trained_one_shot')

    print('Training Hebbian Model')
    hebbian_model = HebbianModel(som_one_shot_a, som_one_shot_v, a_dim, v_dim, n_presentations=10)
    hebbian_model.train(h_a_xs_train, h_v_xs_train)

    hebbian_model.plot_class_som_activations(a_xs_all_test, v_xs_all_test, a_ys_all_test, v_ys_all_test, source='v',
                                             som_type='audio', title_extended='visual to audio')
    hebbian_model.plot_class_som_activations(a_xs_all_test, v_xs_all_test, a_ys_all_test, v_ys_all_test, source='a',
                                             som_type='video', title_extended='audio to visual')

    print('Compute precision, recall, f_score')
    precision_v, recall_v, f1_score_v = hebbian_model.compute_recall_precision_fscore(
        a_xs_all_test, v_xs_all_test, a_ys_all_test, v_ys_all_test, source='v')
    precision_a, recall_a, f1_score_a = hebbian_model.compute_recall_precision_fscore(
        a_xs_all_test, v_xs_all_test, a_ys_all_test, v_ys_all_test, source='a')
    print('Compute accuracy')
    accuracy_test_v = hebbian_model.evaluate(a_xs_all_test, v_xs_all_test, a_ys_all_test, v_ys_all_test, source='v',
                                             prediction_alg='regular')
    accuracy_test_a = hebbian_model.evaluate(a_xs_all_test, v_xs_all_test, a_ys_all_test, v_ys_all_test, source='a',
                                             prediction_alg='regular')
    print('Compute accuracy class')
    accuracy_class = hebbian_model.evalute_class(a_xs_all_test, v_xs_all_test, a_ys_all_test, v_ys_all_test,
                                                 source='v', class_to_evaluate=i)
    accuracy_class_a = hebbian_model.evalute_class(a_xs_all_test, v_xs_all_test, a_ys_all_test, v_ys_all_test,
                                                   source='a', class_to_evaluate=i)

    print('Accuracy Class \'{}\' (Visual) = {}'.format(Constants.label_classes[i], accuracy_class))
    print('Accuracy Class \'{}\' (Audio)= {}'.format(Constants.label_classes[i], accuracy_class_a))

    print('--> Accuracy Test Set (Source \'Visual\'): {}'.format(accuracy_test_v))
    print('\tPrecision={}, Recall={}, F1_score={}'.format(precision_v, recall_v, f1_score_v))
    print('--> Accuracy Test Set (Source \'Audio\'): {}'.format(accuracy_test_a))
    print('\tPrecision={}, Recall={}, F1_score={}'.format(precision_a, recall_a, f1_score_a))

    # **********************************

    #
    # for train_index, test_index in kf_v.split(v_xs, v_ys):
    #
    #     som_v = SelfOrganizingMap(n, m, v_dim, n_iterations=n_iters, sigma=sigma, learning_rate=alpha)
    #
    #     # Splits visual data into train and test based on k-folds indexes
    #     v_xs_train, v_xs_test = v_xs[np.array(train_index)], v_xs[np.array(test_index)]
    #     v_ys_train, v_ys_test = v_ys[np.array(train_index)], v_ys[np.array(test_index)]
    #
    #     for a_train_index, a_test_index in kf_a.split(a_xs, a_ys):
    #         print('\n *************** TEST {}/{} ***************\n'.format(iterations, splits_v * splits_a))
    #
    #         som_a = SelfOrganizingMap(n, m, a_dim, n_iterations=n_iters, sigma=sigma, learning_rate=alpha)
    #
    #         # Splits audio data into train and test based on k-folds indexes
    #         a_xs_train, a_xs_test = a_xs[np.array(a_train_index)], a_xs[np.array(a_test_index)]
    #         a_ys_train, a_ys_test = a_ys[np.array(a_train_index)], a_ys[np.array(a_test_index)]
    #
    #         # Training of both som on n-1 classes and saving weights
    #         weights_v, weights_a, som_v, som_a = train_som_and_get_weight(som_v, som_a, v_xs_train, a_xs_train)
    #
    #         som_v.plot_som_class_pies(v_xs_train, v_ys_train, type_dataset='trained_n-1_classes')
    #         som_a.plot_som_class_pies(a_xs_train, a_ys_train, type_dataset='audio_trained_n-1_classes')
    #
    #         som_one_shot_v = SelfOrganizingMap(n, m, v_dim, n_iterations=800, sigma=0.8, learning_rate=0.2)
    #         som_one_shot_a = SelfOrganizingMap(n, m, a_dim, n_iterations=800, sigma=0.8, learning_rate=0.2)
    #
    #         print('--> One Shot training')
    #         # Train both SOMs with one shot dataset assigning precomputed weights
    #         som_one_shot_v.train(class_ext_v_xs_train, weights=weights_v)
    #         som_one_shot_a.train(class_ext_a_xs_train, weights=weights_a)
    #
    #         print('Plotting som')
    #         som_one_shot_v.plot_som_class_pies(h_v_xs_test, h_v_ys_test, type_dataset='trained_one_shot')
    #         som_one_shot_a.plot_som_class_pies(h_a_xs_test, h_a_ys_test, type_dataset='audio_trained_one_shot')
    #
    #         print('Training Hebbian Model')
    #         hebbian_model = HebbianModel(som_one_shot_a, som_one_shot_v, a_dim, v_dim, n_presentations=10)
    #         hebbian_model.train(h_a_xs_train, h_v_xs_train)
    #         print('Compute statistics')
    #
    #         precision_v, recall_v, f1_score_v = hebbian_model.compute_recall_precision_fscore(
    #             h_a_xs_test, h_v_xs_test, h_a_ys_test, h_v_ys_test, source='v')
    #         precision_a, recall_a, f1_score_a = hebbian_model.compute_recall_precision_fscore(
    #             h_a_xs_test, h_v_xs_test, h_a_ys_test, h_v_ys_test, source='a')
    #         accuracy_test_v = hebbian_model.evaluate(h_a_xs_test, h_v_xs_test, h_a_ys_test, h_v_ys_test, source='v',
    #                                                  prediction_alg='regular')
    #         accuracy_test_a = hebbian_model.evaluate(h_a_xs_test, h_v_xs_test, h_a_ys_test, h_v_ys_test, source='a',
    #                                                  prediction_alg='regular')
    #         accuracy_class = hebbian_model.evalute_class(h_a_xs_test, h_v_xs_test, h_a_ys_test, h_v_ys_test,
    #                                                      source='v', class_to_evaluate=i)
    #         accuracy_class_a = hebbian_model.evalute_class(h_a_xs_test, h_v_xs_test, h_a_ys_test, h_v_ys_test,
    #                                                        source='a', class_to_evaluate=i)
    #
    #         print('Accuracy Class \'{}\' (Visual) = {}'.format(Constants.label_classes[i], accuracy_class))
    #         print('Accuracy Class \'{}\' (Audio)= {}'.format(Constants.label_classes[i], accuracy_class_a))
    #
    #         precisions.append(precision_v), recalls.append(recall_v), f1_scores.append(f1_score_v)
    #         precisions_a.append(precision_a), recalls_a.append(recall_a), f1_scores_a.append(f1_score_a)
    #         accuracy_list.append(accuracy_test_v), accuracy_list_source_a.append(accuracy_test_a)
    #
    #         print('--> Accuracy Test Set (Source \'Visual\'): {}'.format(accuracy_test_v))
    #         print('\tPrecision={}, Recall={}, F1_score={}'.format(precision_v, recall_v, f1_score_v))
    #         print('--> Accuracy Test Set (Source \'Audio\'): {}'.format(accuracy_test_a))
    #         print('\tPrecision={}, Recall={}, F1_score={}'.format(precision_a, recall_a, f1_score_a))
    #         iterations += 1
    #
    # mean_accuracy, mean_variance = np.mean(accuracy_list), np.var(accuracy_list)
    # mean_accuracy_a, mean_variance_a = np.mean(accuracy_list_source_a), np.var(accuracy_list_source_a)
    # mean_precision_v, mean_precision_a = np.mean(precisions), np.mean(precisions_a)
    # mean_recall_v, mean_recall_a = np.mean(recalls), np.mean(recalls_a)
    # mean_f1_scores_v, mean_f1_scores_a = np.mean(f1_scores), np.mean(f1_scores_a)
    #
    # write_md_file(os.path.join(Constants.PLOT_FOLDER, 'temp', 'tests_all_dataset.md'), mean_accuracy,
    #               mean_accuracy_a,
    #               mean_variance, mean_variance_a, scaler,
    #               '{}x{}'.format(n, m), alpha, sigma, n_iters, splits_v, splits_a, 'Test (Sigma {})'.format(sigma),
    #               [mean_precision_v, mean_precision_a], [mean_recall_v, mean_recall_a],
    #               [mean_f1_scores_v, mean_f1_scores_a])
    #
    # print('\n\nMean Accuracy (source visual)={} (var={}) on v_kf={} and a_kf={} '.format(np.mean(accuracy_list),
    #                                                                                      np.var(accuracy_list),
    #                                                                                      kf_v.get_n_splits(),
    #                                                                                      kf_a.get_n_splits()))
    # print('Mean Accuracy (source audio)={} (var={}) on v_kf={} and a_kf={} '.format(np.mean(accuracy_list_source_a),
    #                                                                                 np.var(accuracy_list_source_a),
    #                                                                                 kf_v.get_n_splits(),
    #                                                                                 kf_a.get_n_splits()))
