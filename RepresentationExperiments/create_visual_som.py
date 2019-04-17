from models.som.SOM import SOM
from models.som.SOMTest import showSom
import numpy as np
from utils.constants import Constants
from utils.utils import from_csv_visual_10classes
from sklearn.preprocessing import MinMaxScaler
import os
import logging

visual_data_path = os.path.join(Constants.DATA_FOLDER,
                                '10classes',
                                'VisualInputTrainingSet.csv')
N = 1000
lenExample = 2048

if __name__ == '__main__':
    v_xs, v_ys = from_csv_visual_10classes(visual_data_path)
    v_xs = MinMaxScaler().fit_transform(v_xs)

    som = SOM(20, 30, lenExample,
          checkpoint_dir=os.path.join(Constants.DATA_FOLDER, 'visual_model_mine', ''),
          n_iterations=100, sigma=4.0)

    som.restore_trained()

    showSom(som, v_xs, v_ys, 1, 'Visual map')
