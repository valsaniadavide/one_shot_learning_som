import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from models.som.HebbianModel import HebbianModel
from models.som.SOM import SOM
from models.som.SOMTest import show_som, show_confusion
from oneShotLearning.utility import *

from utils.constants import Constants
from utils.utils import from_csv_with_filenames, from_csv_visual_10classes

audio_data_path = os.path.join(Constants.DATA_FOLDER,
                               '10classes',
                               'audio_data_25t.csv')
visual_data_path = os.path.join(Constants.DATA_FOLDER,
                                '10classes',
                                'VisualInputTrainingSet.csv')
classes = list(range(0, 10))

if __name__ == '__main__':
    print('classes', classes)
    a_xs, a_ys, _ = from_csv_with_filenames(audio_data_path)
    v_xs, v_ys = from_csv_visual_10classes(visual_data_path)
    a_ys = [int(y) - 1000 for y in a_ys]
    v_ys = [int(y) - 1000 for y in v_ys]
    # scale data to 0-1 range
    a_xs = StandardScaler().fit_transform(a_xs)
    v_xs = StandardScaler().fit_transform(v_xs)
    for i in range(0, 1):
        a_xs_train, a_ys_train, a_xs_test, a_ys_test = get_random_classes(a_xs, a_ys, classes, 10, -1)
        v_xs_train, v_ys_train, v_xs_test, v_ys_test = get_random_classes(v_xs, v_ys, classes, 10, -1)
        # print('a_xs_train', np.shape(a_xs_train))
        # print('a_ys_train', a_ys_train)
        # print('a_xs_test', np.shape(a_xs_test))
        # print('a_ys_test', a_ys_test)
        # print('shape audio input train', np.shape(a_xs_train))
        # print('shape audio labels train', np.shape(a_ys_train))
        # print('shape visual input train', np.shape(v_xs_train))
        # print('shape visual labels train', np.shape(v_ys_train))
        a_dim = len(a_xs[0])
        v_dim = len(v_xs[0])
        som_a = SOM(20, 30, a_dim, alpha=2, sigma=10, n_iterations=100,
                    tau=0.1, threshold=0.6, batch_size=1)
        # type_file = 'visual_' + str(i + 1)
        type_file = 'visual'
        som_v = SOM(20, 30, v_dim, alpha=0.8, sigma=20, n_iterations=2500, threshold=0.6, batch_size=1,
                    data=type_file)
        # som_a.train(a_xs_train, input_classes=a_ys_train, test_vects=a_xs_test, test_classes=a_ys_test, save_every=100)
        som_v.train(v_xs_train, input_classes=v_ys_train, test_vects=v_xs_test, test_classes=v_ys_test, save_every=100)
        show_som(som_v, v_xs, v_ys, ['table', 'mug', 'tree', 'dog', 'house', 'book', 'guitar', 'fish', 'cat', 'bird'],
                 'Video SOM (bounding boxes)', show=False, dark=False, scatter=True,
                 legend=True, point_size=120, suffix='_segm_trsf_minmax')
        show_confusion(som_v, v_xs, v_ys, title="Video SOM confusion")
        # hebbian_model = HebbianModel(som_a, som_v, a_dim, v_dim, n_presentations=10)
        # hebbian_model.train(a_xs_train, v_xs_train)
        # accuracy = hebbian_model.evaluate(a_xs_train, v_xs_train, a_ys_train, v_ys_train, source='v',
        #                                   prediction_alg='regular')
        # print("Accuracy Training set", accuracy)
        # accuracy_test = hebbian_model.evaluate(a_xs_test, v_xs_test, a_ys_test, v_ys_test, source='v',
        #                                        prediction_alg='regular')
        # print("Accuracy Test set", accuracy_test)
