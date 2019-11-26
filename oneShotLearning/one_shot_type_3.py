from mdutils import MdUtils

from sklearn.model_selection import train_test_split

from SelfOrganizingMaps.SelfOrganizingMap import SelfOrganizingMap
from models.som.HebbianModel import HebbianModel
from oneShotLearning.utility import *
from oneShotLearning.clean_folders import clean_statistics_folders
from utils.constants import Constants

if __name__ == '__main__':
    v_xs_all, v_ys_all, a_xs_all, a_ys_all, filenames_visual, filenames_audio = import_data(
        Constants.visual_data_path_segmented, Constants.audio_data_path, segmented=True)

    v_dim, a_dim = np.shape(v_xs_all)[1], np.shape(a_xs_all)[1]

    alpha, sigma = 0.3, 3
    n_iters = 5500
    n, m = 20, 30
    scaler = 'Standard Scaler'
    splits_v, splits_a = 7, 3
    iterations_one_shot = 300
    sigma_one_shot = 0.8
    alpha_one_shot = 0.1
    hebbian_threshold = .85
    shots = 5
    clean_statistics_folders()

    # Split dataset into training set and test set (balanced examples per class)
    v_xs_all_train, v_xs_all_test, v_ys_all_train, v_ys_all_test = train_test_split(v_xs_all, v_ys_all,
                                                                                    test_size=0.35, random_state=42,
                                                                                    stratify=v_ys_all)
    a_xs_all_train, a_xs_all_test, a_ys_all_train, a_ys_all_test = train_test_split(a_xs_all, a_ys_all,
                                                                                    test_size=0.35, random_state=42,
                                                                                    stratify=a_ys_all)

    # Create a new dataset of 10 random examples per class used for hebbian training
    h_a_xs_train, h_a_ys_train, _, _ = get_random_classes(a_xs_all_train, a_ys_all_train,
                                                          Constants.classes, 10, -1)
    h_v_xs_train, h_v_ys_train, _, _ = get_random_classes(v_xs_all_train, v_ys_all_train,
                                                          Constants.classes, 10, -1)
    for hebbian_threshold in range(60, 90, 5):

        test_root_folder = os.path.join(Constants.PLOT_FOLDER, 'temp',
                                        'one_shot_results_threshold_{}'.format(int(hebbian_threshold)))
        safe_create_folder(test_root_folder)
        hebbian_threshold = hebbian_threshold / 100

        for i in Constants.classes:
            label_classes = Constants.label_classes.copy()
            label_classes.pop(i)
            classes = Constants.classes.copy()
            classes.pop(i)
            # Extract the class i-th from the train and the test sets
            class_ext_v_xs, class_ext_v_ys, v_xs, v_ys = get_examples_of_class(v_xs_all_train, v_ys_all_train,
                                                                               Constants.classes, i)
            class_ext_a_xs, class_ext_a_ys, a_xs, a_ys = get_examples_of_class(a_xs_all_train, a_ys_all_train,
                                                                               Constants.classes, i)
            _, _, v_xs_n_1_classes_test, v_ys_n_1_classes_test = get_examples_of_class(v_xs_all_test, v_ys_all_test,
                                                                                       Constants.classes, i)
            _, _, a_xs_n_1_classes_test, a_ys_n_1_classes_test = get_examples_of_class(a_xs_all_test, a_ys_all_test,
                                                                                       Constants.classes, i)

            # Get n examples from the extracted class data for one-shot learning
            class_ext_v_xs_train, class_ext_v_ys_train, class_ext_v_xs_test, class_ext_v_ys_test = get_random_classes(
                class_ext_v_xs, class_ext_v_ys, [i], shots, -1)
            class_ext_a_xs_train, class_ext_a_ys_train, class_ext_a_xs_test, class_ext_a_ys_test = get_random_classes(
                class_ext_a_xs, class_ext_a_ys, [i], shots, -1)

            # Get 10 examples from the n-1 classes, used to train the hebbian model for SOMs pre one-shot
            h_a_xs_train_n_1, h_a_ys_train_n_1, h_a_xs_test_n_1, h_a_ys_test_n_1 = get_random_classes(a_xs, a_ys,
                                                                                                      Constants.classes,
                                                                                                      10, -1,
                                                                                                      class_to_exclude=i)
            h_v_xs_train_n_1, h_v_ys_train_n_1, h_v_xs_test_n_1, h_v_ys_test_n_1 = get_random_classes(v_xs, v_ys,
                                                                                                      Constants.classes,
                                                                                                      10, -1,
                                                                                                      class_to_exclude=i)

            statistics = ['Data', 'Value']
            statistics.extend(['Size', '{}x{}'.format(n, m)])
            statistics.extend(['Hebbian Threshold', '{}'.format(hebbian_threshold)])
            statistics.extend(['Num. Shots', '{}'.format(shots)])
            statistics.extend(['+++', '+++'])
            statistics.extend(['Learning Rate SOMs n-1 classes', '{}'.format(alpha)])
            statistics.extend(['Sigma SOMs n-1 classes', '{}'.format(sigma)])
            statistics.extend(['Iterations SOMs n-1 classes', '{}'.format(n_iters)])

            print(
                '\n\n ++++++++++++++++++++++++++++ {}) TEST ONE SHOT ON CLASS = \'{}\' ++++++++++++++++++++++++++++ \n\n'
                    .format(i + 1, Constants.label_classes[i]))

            iterations = 1
            accuracy_list, accuracy_list_source_a = [], []
            precisions, precisions_a = [], []
            recalls, recalls_a = [], []
            f1_scores, f1_scores_a = [], []

            # Creating folders to save test results

            root_folder = '{}_one_shot_class_{}'.format(i + 1, Constants.label_classes[i])
            path = os.path.join(test_root_folder, root_folder)
            test_path = os.path.join(path, 'test_set')
            training_path = os.path.join(path, 'training_set')
            safe_create_folder(path)
            safe_create_folder(os.path.join(path, 'training_set', 'plots_classes_pre_one_shot'))
            safe_create_folder(os.path.join(path, 'training_set', 'plots_one_shots_results'))
            safe_create_folder(os.path.join(path, 'training_set', 'plots_activations_per_class'))
            safe_create_folder(os.path.join(path, 'test_set', 'plots_classes_pre_one_shot'))
            safe_create_folder(os.path.join(path, 'test_set', 'plots_one_shots_results'))
            safe_create_folder(os.path.join(path, 'test_set', 'plots_activations_per_class'))

            som_v = SelfOrganizingMap(n, m, v_dim, n_iterations=n_iters, sigma=sigma, learning_rate=alpha)
            som_a = SelfOrganizingMap(n, m, a_dim, n_iterations=n_iters, sigma=sigma, learning_rate=alpha)

            weights_v, weights_a, som_v, som_a = train_som_and_get_weight(som_v, som_a, v_xs, a_xs)

            # Training Hebbian Model for SOMs with n-1 classes
            hebbian_model = HebbianModel(som_a, som_v, a_dim, v_dim, n_presentations=10,
                                         n_classes=len(Constants.classes) - 1, threshold=hebbian_threshold)
            hebbian_model.train(h_a_xs_train_n_1, h_v_xs_train_n_1)

            # Evaluating Hebbian Model for SOMs with n-1 classes
            accuracy_test_v = hebbian_model.evaluate(h_a_xs_test_n_1, h_v_xs_test_n_1, h_a_ys_test_n_1, h_v_ys_test_n_1,
                                                     source='v',
                                                     prediction_alg='regular')
            accuracy_test_a = hebbian_model.evaluate(h_a_xs_test_n_1, h_v_xs_test_n_1, h_a_ys_test_n_1, h_v_ys_test_n_1,
                                                     source='a',
                                                     prediction_alg='regular')
            accuracy_train_v = hebbian_model.evaluate(h_a_xs_train_n_1, h_v_xs_train_n_1, h_a_ys_train_n_1,
                                                      h_v_ys_train_n_1,
                                                      source='v',
                                                      prediction_alg='regular')
            accuracy_train_a = hebbian_model.evaluate(h_a_xs_train_n_1, h_v_xs_train_n_1, h_a_ys_train_n_1,
                                                      h_v_ys_train_n_1,
                                                      source='a',
                                                      prediction_alg='regular')

            statistics.extend(
                ['Accuracy Test Set (Source \'Visual\') n-1 classes', '{}'.format(round(accuracy_test_v, 2))])
            statistics.extend(
                ['Accuracy Test Set (Source \'Audio\') n-1 classes', '{}'.format(round(accuracy_test_a, 2))])
            statistics.extend(
                ['Accuracy Training Set (Source \'Visual\') n-1 classes', '{}'.format(round(accuracy_train_v, 2))])
            statistics.extend(
                ['Accuracy Training Set (Source \'Audio\') n-1 classes', '{}'.format(round(accuracy_train_a, 2))])

            print('\n--> Accuracy Test Set (Source \'Visual\') n-1 classes: {}'.format(accuracy_test_v))
            print('--> Accuracy Test Set (Source \'Audio\') n-1 classes: {}\n'.format(accuracy_test_a))

            som_v.plot_som(v_xs_n_1_classes_test, v_ys_n_1_classes_test,
                           os.path.join(test_path, 'plots_classes_pre_one_shot'),
                           type_dataset='video_trained_n-1_classes',
                           label_classes=label_classes, class_to_exclude=i)
            som_a.plot_som(a_xs_n_1_classes_test, a_ys_n_1_classes_test,
                           os.path.join(test_path, 'plots_classes_pre_one_shot'),
                           type_dataset='audio_trained_n-1_classes',
                           label_classes=label_classes, class_to_exclude=i)
            som_v.plot_som(v_xs, v_ys, os.path.join(training_path, 'plots_classes_pre_one_shot'),
                           type_dataset='video_trained_n-1_classes',
                           label_classes=label_classes, class_to_exclude=i)
            som_a.plot_som(a_xs, a_ys, os.path.join(training_path, 'plots_classes_pre_one_shot'),
                           type_dataset='audio_trained_n-1_classes',
                           label_classes=label_classes, class_to_exclude=i)

            som_v.plot_u_matrix(os.path.join(training_path, 'plots_classes_pre_one_shot'),
                                name='video_trained_n-1_classes')
            som_a.plot_u_matrix(os.path.join(training_path, 'plots_classes_pre_one_shot'),
                                name='audio_trained_n-1_classes')

            som_one_shot_v = SelfOrganizingMap(n, m, v_dim, n_iterations=iterations_one_shot, sigma=sigma_one_shot,
                                               learning_rate=alpha_one_shot)
            som_one_shot_a = SelfOrganizingMap(n, m, a_dim, n_iterations=iterations_one_shot, sigma=sigma_one_shot,
                                               learning_rate=alpha_one_shot)

            print('--> One Shot training...')
            # Train both SOMs with one shot dataset assigning precomputed weights
            som_one_shot_v.train(class_ext_v_xs_train, weights=weights_v)
            som_one_shot_a.train(class_ext_a_xs_train, weights=weights_a)

            statistics.extend(['***', '***'])
            statistics.extend(['Learning Rate SOMs One-shot', '{}'.format(alpha_one_shot)])
            statistics.extend(['Sigma SOMs n-1 One-shot', '{}'.format(sigma_one_shot)])
            statistics.extend(['Iterations SOMs One-shot', '{}'.format(iterations_one_shot)])

            print('--> Plotting som...')
            som_one_shot_v.plot_som(v_xs_all_test, v_ys_all_test, os.path.join(test_path, 'plots_one_shots_results'),
                                    type_dataset='video_trained_one_shot')
            som_one_shot_a.plot_som(a_xs_all_test, a_ys_all_test, os.path.join(test_path, 'plots_one_shots_results'),
                                    type_dataset='audio_trained_one_shot')
            som_one_shot_v.plot_som(v_xs_all_train, v_ys_all_train,
                                    os.path.join(training_path, 'plots_one_shots_results'),
                                    type_dataset='video_trained_one_shot')
            som_one_shot_a.plot_som(a_xs_all_train, a_ys_all_train,
                                    os.path.join(training_path, 'plots_one_shots_results'),
                                    type_dataset='audio_trained_one_shot')

            som_one_shot_v.plot_u_matrix(os.path.join(training_path, 'plots_one_shots_results'),
                                         name='video_trained_one_shot')
            som_one_shot_a.plot_u_matrix(os.path.join(training_path, 'plots_one_shots_results'),
                                         name='audio_trained_one_shot')

            print('--> Training Hebbian Model One-Shot SOMs...')

            # Training Hebbian Model for one-shot SOMs
            hebbian_model = HebbianModel(som_one_shot_a, som_one_shot_v, a_dim, v_dim, n_presentations=10,
                                         threshold=hebbian_threshold)
            hebbian_model.train(h_a_xs_train, h_v_xs_train)

            # print('Compute precision, recall, f_score')
            # precision_v, recall_v, f1_score_v = hebbian_model.compute_recall_precision_fscore(
            #     a_xs_all_test, v_xs_all_test, a_ys_all_test, v_ys_all_test, source='v')
            # precision_a, recall_a, f1_score_a = hebbian_model.compute_recall_precision_fscore(
            #     a_xs_all_test, v_xs_all_test, a_ys_all_test, v_ys_all_test, source='a')
            print('--> Compute accuracy...')
            accuracy_test_v = hebbian_model.evaluate(a_xs_all_test, v_xs_all_test, a_ys_all_test, v_ys_all_test,
                                                     source='v',
                                                     prediction_alg='regular')
            accuracy_test_a = hebbian_model.evaluate(a_xs_all_test, v_xs_all_test, a_ys_all_test, v_ys_all_test,
                                                     source='a',
                                                     prediction_alg='regular')
            accuracy_train_v = hebbian_model.evaluate(a_xs_all_train, v_xs_all_train, a_ys_all_train, v_ys_all_train,
                                                      source='v', prediction_alg='regular')
            accuracy_train_a = hebbian_model.evaluate(a_xs_all_train, v_xs_all_train, a_ys_all_train, v_ys_all_train,
                                                      source='a', prediction_alg='regular')
            # print('Compute accuracy class')
            # accuracy_class = hebbian_model.evalute_class(a_xs_all_test, v_xs_all_test, a_ys_all_test, v_ys_all_test,
            #                                              source='v', class_to_evaluate=i)
            # accuracy_class_a = hebbian_model.evalute_class(a_xs_all_test, v_xs_all_test, a_ys_all_test, v_ys_all_test,
            #                                                source='a', class_to_evaluate=i)

            # print('Accuracy Class \'{}\' (Visual) = {}'.format(Constants.label_classes[i], accuracy_class))
            # print('Accuracy Class \'{}\' (Audio)= {}'.format(Constants.label_classes[i], accuracy_class_a))

            print('--> Accuracy Test Set (Source \'Visual\'): {}'.format(accuracy_test_v))
            # print('\tPrecision={}, Recall={}, F1_score={}'.format(precision_v, recall_v, f1_score_v))
            print('--> Accuracy Test Set (Source \'Audio\'): {}'.format(accuracy_test_a))
            # print('\tPrecision={}, Recall={}, F1_score={}'.format(precision_a, recall_a, f1_score_a))

            # Plot for each class both SOM activations
            for index_class in Constants.classes:
                print('--> Evaluating Activation for class \'{}\''.format(Constants.label_classes[index_class]))
                class_path = '{}) activations class "{}"'.format(index_class, Constants.label_classes[index_class])
                path_training_actv = os.path.join(training_path, 'plots_activations_per_class', class_path)
                path_test_actv = os.path.join(test_path, 'plots_activations_per_class', class_path)
                safe_create_folder(path_training_actv)
                safe_create_folder(path_test_actv)
                hebbian_model.plot_class_som_activations(a_xs_all_test, v_xs_all_test, a_ys_all_test, v_ys_all_test,
                                                         path_test_actv,
                                                         source='v',
                                                         som_type='audio', title_extended='visual to audio',
                                                         class_to_extract=index_class)
                hebbian_model.plot_class_som_activations(a_xs_all_test, v_xs_all_test, a_ys_all_test, v_ys_all_test,
                                                         path_test_actv,
                                                         source='a',
                                                         som_type='video', title_extended='audio to visual',
                                                         class_to_extract=index_class)
                hebbian_model.plot_class_som_activations(a_xs_all_train, v_xs_all_train, a_ys_all_train, v_ys_all_train,
                                                         path_training_actv,
                                                         source='v',
                                                         som_type='audio', title_extended='visual to audio',
                                                         class_to_extract=index_class)
                hebbian_model.plot_class_som_activations(a_xs_all_train, v_xs_all_train, a_ys_all_train, v_ys_all_train,
                                                         path_training_actv,
                                                         source='a',
                                                         som_type='video', title_extended='audio to visual',
                                                         class_to_extract=index_class)

            statistics.extend(
                ['Accuracy Test Set (Source \'Visual\') One-Shot', '{}'.format(round(accuracy_test_v, 2))])
            statistics.extend(['Accuracy Test Set (Source \'Audio\') One-Shot', '{}'.format(round(accuracy_test_a, 2))])
            statistics.extend(
                ['Accuracy Training Set (Source \'Visual\') One-Shot', '{}'.format(round(accuracy_train_v, 2))])
            statistics.extend(
                ['Accuracy Training Set (Source \'Audio\') One-Shot', '{}'.format(round(accuracy_train_a, 2))])

            # Creating and saving markdown file of results about the class analyzed
            mdFile = MdUtils(file_name=os.path.join(path, 'results_class_{}'.format(Constants.label_classes[i])),
                             title='Statistics about one-shot on class \'{}\''.format(Constants.label_classes[i]))
            mdFile.new_line()
            mdFile.new_table(columns=2, rows=20, text=statistics,
                             text_align='center')
            mdFile.create_md_file()
