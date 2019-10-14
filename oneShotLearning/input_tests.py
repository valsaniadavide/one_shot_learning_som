import os
from oneShotLearning.utility import import_data, get_random_classes, get_min_max_mean_input_feature
from utils.constants import Constants
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, Normalizer
from sklearn.cluster import KMeans

from utils.utils import from_npy_visual_data

if __name__ == '__main__':
    v_xs, v_ys, a_xs, a_ys, filenames_visual, filenames_audio = import_data(Constants.visual_data_path,
                                                                            Constants.audio_data_path)

    a_xs_train, a_ys_train, a_xs_test, a_ys_test = get_random_classes(a_xs, a_ys, Constants.classes, 10, -1)
    v_xs_train, v_ys_train, v_xs_test, v_ys_test = get_random_classes(v_xs, v_ys, Constants.classes, 10, -1)
    # v_xs, v_ys, _ = from_npy_visual_data(os.path.join(Constants.DATA2_FOLDER, 'visual-10classes-bbox.npy'))
    # v_xs = MaxAbsScaler().fit_transform(v_xs)
    # print(np.shape(v_xs))
    # print(v_xs_train)
    print(np.shape(v_xs_test))
    df_stats_input = get_min_max_mean_input_feature(v_xs_test)
    print(np.shape(df_stats_input))
    # print(np.shape(df_stats_input))
    print(df_stats_input)

    kmeans = KMeans(n_clusters=10).fit(v_xs_train)
    predicted = kmeans.predict(v_xs_test)
    print('predicted: {}'.format(np.array(predicted)))
    print('real:  {}'.format(v_ys_test))

    correct = 0
    for yi, yj in zip(predicted, v_ys_test):
        if yi == yj:
            correct += 1
    print('Accuracy K-Means: {}'.format(correct/len(predicted)))
    # matrix = tf.Variable(tf.random_normal([8 * 8, 2048], mean=0, stddev=1))
    # sess = tf.Session()
    #
    # # INITIALIZE VARIABLES
    # init_op = tf.global_variables_initializer()
    # sess.run(init_op)
    #
    # matrix_np = matrix.eval(session=sess)
    # print(matrix_np)
    # print(tf.__version__)
    # weights = tf.random_normal([30 * 20, 2048], mean=0, stddev=1)
