import os
import pandas as pd

from models.som.HebbianModel import HebbianModel
from models.som.SOM import SOM
from oneShotLearning.utility import *

from utils.constants import Constants

audio_data_path = os.path.join(Constants.DATA_FOLDER,
                               '10classes',
                               'audio_data_25t.csv')
visual_data_path = os.path.join(Constants.DATA_FOLDER,
                                '10classes',
                                'VisualInputTrainingSet.csv')
classes = list(range(0, 10))
columns_stat = ['n', 'm', 'alpha', 'sigma', 'accuracy_train', 'accuracy_test', 'v_compact', 'v_compact_var', 'v_confus',
                'a_compact', 'a_compact_var', 'a_confus']
label_classes = ['table', 'mug', 'tree', 'dog', 'house', 'book', 'guitar', 'fish', 'cat', 'bird']

if __name__ == '__main__':
    v_xs, v_ys, a_xs, a_ys = import_data(visual_data_path, audio_data_path)

    stats = []
    for i in range(0, 20):
        a_xs_train, a_ys_train, a_xs_test, a_ys_test = get_random_classes(a_xs, a_ys, classes, 10, -1)
        v_xs_train, v_ys_train, v_xs_test, v_ys_test = get_random_classes(v_xs, v_ys, classes, 10, -1)

        # print('shape audio input train', np.shape(a_xs_train))
        # print('shape audio labels train', np.shape(a_ys_train))
        # print('shape visual input train', np.shape(v_xs_train))
        # print('shape visual labels train', np.shape(v_ys_train))

        a_dim = len(a_xs[0])
        v_dim = len(v_xs[0])
        random.seed(a=None, version=2)
        learning_rate = round(random.uniform(0.2, 0.35), 2)
        sigma = random.randrange(75, 100, 3)

        som_a = SOM(20, 30, a_dim, alpha=learning_rate, sigma=sigma, n_iterations=1000, batch_size=1)
        # type_file = 'visual_' + str(i + 1)
        type_file = 'visual'
        som_v = SOM(20, 30, v_dim, alpha=learning_rate, sigma=sigma, n_iterations=1000, batch_size=1, data=type_file)

        som_a.train(a_xs_train, input_classes=a_ys_train, test_vects=a_xs_test, test_classes=a_ys_test, save_every=100)
        som_v.train(v_xs_train, input_classes=v_ys_train, test_vects=v_xs_test, test_classes=v_ys_test, save_every=100)

        # som_v.print_som_evaluation(v_xs_test, v_ys_test)
        # som_v.neuron_collapse_classwise(v_xs_test, v_ys_test)
        print_charts(som_v, v_xs_test, v_ys_test, label_classes, 'visual', "Visual SOM")
        print_charts(som_a, a_xs_test, a_ys_test, label_classes, 'Audio', "Audio SOM")

        v_compact, v_compact_var, v_confus = som_v.compute_compactness_confusion(v_xs_test, v_ys_test)
        a_compact, a_compact_var, a_confus = som_a.compute_compactness_confusion(a_xs_test, a_ys_test)
        # som_v.plot_som(v_xs_test, v_ys_test, plot_name='som-vis.png')
        # som_a.plot_som(a_xs_test, a_ys_test, plot_name='som-aud.png')

        hebbian_model = HebbianModel(som_a, som_v, a_dim, v_dim, n_presentations=10)
        hebbian_model.train(a_xs_train, v_xs_train)

        accuracy_train = hebbian_model.evaluate(a_xs_train, v_xs_train, a_ys_train, v_ys_train, source='v',
                                                prediction_alg='regular')
        accuracy_test = hebbian_model.evaluate(a_xs_test, v_xs_test, a_ys_test, v_ys_test, source='v',
                                               prediction_alg='regular')

        stats.append([som_v._n, som_v._m, learning_rate, sigma, accuracy_train, accuracy_test, v_compact, v_compact_var,
                      v_confus, a_compact, a_compact_var, a_confus])

        print("Accuracy Training set", accuracy_train)
        print("Accuracy Test set", accuracy_test)
        print('test_n={}, n={}, m={}, learning_rate={}, sigma={}, accuracy_train={}, accuracy_test={}, v_compact={}, '
              'v_compact_var={}, v_confus={}, a_compact={}, a_compact_var={}, a_confus={} '
              .format(i, som_v._n, som_v._m, learning_rate, sigma, accuracy_train, accuracy_test, v_compact,
                      v_compact_var, v_confus, a_compact, a_compact_var, a_confus))

    df = pd.DataFrame(data=stats, columns=columns_stat)
    df.head(n=10)
    df.to_csv(os.path.join(Constants.PLOT_FOLDER, 'temp', 'statistics.csv'))
