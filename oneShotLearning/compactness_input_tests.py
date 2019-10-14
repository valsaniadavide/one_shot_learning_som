from oneShotLearning.utility import import_data, get_random_classes, get_min_max_mean_input_feature, inputs_compactness
from utils.constants import Constants
import numpy as np


if __name__ == '__main__':
    v_xs, v_ys, a_xs, a_ys, filenames_visual, filenames_audio = import_data(Constants.visual_data_path,
                                                                            Constants.audio_data_path)

    a_xs_train, a_ys_train, a_xs_test, a_ys_test = get_random_classes(a_xs, a_ys, Constants.classes, 10, -1)
    v_xs_train, v_ys_train, v_xs_test, v_ys_test = get_random_classes(v_xs, v_ys, Constants.classes, 10, -1)
    compactness = inputs_compactness(v_xs, v_ys)
    print('compactness', compactness)
    print('shape', np.shape(compactness))
