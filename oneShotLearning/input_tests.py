import os
from oneShotLearning.utility import import_data, get_random_classes
from utils.constants import Constants
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, Normalizer


from utils.utils import from_npy_visual_data


def get_min_max_mean_input_feature(xs):
    df = pd.DataFrame(xs)
    # df = df.transpose()
    data_stat = pd.DataFrame()
    data_stat['min_value'] = df.min(axis=0)
    data_stat['max_value'] = df.max(axis=0)
    data_stat['mean_value'] = df.mean(axis=0)
    return data_stat


if __name__ == '__main__':
    v_xs, v_ys, a_xs, a_ys, filenames_visual, filenames_audio = import_data(Constants.visual_data_path,
                                                                            Constants.audio_data_path)

    # a_xs_train, a_ys_train, a_xs_test, a_ys_test = get_random_classes(a_xs, a_ys, Constants.classes, 10, -1)
    # v_xs_train, v_ys_train, v_xs_test, v_ys_test = get_random_classes(v_xs, v_ys, Constants.classes, 10, -1)
    # v_xs, v_ys, _ = from_npy_visual_data(os.path.join(Constants.DATA2_FOLDER, 'visual-10classes-bbox.npy'))
    v_xs = MaxAbsScaler().fit_transform(v_xs)
    print(np.shape(v_xs))
    df_stats_input = get_min_max_mean_input_feature(v_xs)
    print(np.shape(df_stats_input))
    print(df_stats_input)
    # print(tf.__version__)
    # weights = tf.random_normal([30 * 20, 2048], mean=0, stddev=1)

