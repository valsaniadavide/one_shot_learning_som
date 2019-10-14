import os

from RepresentationExperiments.distance_experiments import get_prototypes
from models.som.SOM import SOM
from oneShotLearning.utility import import_data, get_random_classes
from utils.constants import Constants
from models.som.wordLearningTest import showActivationsS, getActivationsOnce
from utils.utils import to_csv
from models.som.wordLearningTest import getAllInputClass, getAllInputClassAudio

audio_data_path = os.path.join(Constants.DATA_FOLDER,
                               '10classes',
                               'audio_data_25t.csv')
visual_data_path = os.path.join(Constants.DATA_FOLDER,
                                '10classes',
                                'VisualInputTrainingSet.csv')

classes = list(range(0, 10))

label_classes = ['table', 'mug', 'tree', 'dog', 'house', 'book', 'guitar', 'fish', 'cat', 'bird']

if __name__ == '__main__':
    v_xs, v_ys, a_xs, a_ys, filenames_visual, filenames_audio = import_data(visual_data_path, audio_data_path)
    print('filenames_audio={}'.format(filenames_audio))
    print('filenames_visual={}'.format(filenames_visual))
    a_xs_train, a_ys_train, a_xs_test, a_ys_test = get_random_classes(a_xs, a_ys, classes, 10, -1)
    v_xs_train, v_ys_train, v_xs_test, v_ys_test = get_random_classes(v_xs, v_ys, classes, 10, -1)
    v_dim = len(v_xs[0])
    a_dim = len(a_xs[0])
    som_v = SOM(20, 30, v_dim, alpha=0.3, sigma=12, n_iterations=100, batch_size=1, data='visual')
    som_v.train(v_xs_train, input_classes=v_ys_train, test_vects=v_xs_test, test_classes=v_ys_test, save_every=100)
    som_a = SOM(20, 30, a_dim, alpha=0.3, sigma=12, n_iterations=100, batch_size=1)
    som_a.train(a_xs_train, input_classes=a_ys_train, test_vects=a_xs_test, test_classes=a_ys_test, save_every=100)
    INPUTV = dict()
    INPUTU = dict()
    tinputU = dict()
    tinputV = dict()

    for c in classes:
        INPUTV[c] = getAllInputClass(c, os.path.join(Constants.DATA_FOLDER, '10classes', 'VisualInputTrainingSet.csv'))
        INPUTU[c] = getAllInputClassAudio(c, os.path.join(Constants.DATA_FOLDER, '10classes', 'audio_data_25t.csv'))

    print('getActivationsOnce')
    activations = getActivationsOnce(som_v, som_a, INPUTV, INPUTU)
    # activations = getActivationsOnce(som_v, som_a, v_xs_test, a_xs_test)
    print('Done compute activations')
    showActivationsS('table', activations, 1)
