from models.som.SOM import SOM
from models.som.wordLearningTest import iterativeTraining
from utils.constants import Constants
from utils.utils import from_csv_with_filenames
from utils.utils import to_csv
from sklearn.preprocessing import MinMaxScaler
from RepresentationExperiments.distance_experiments import get_prototypes
import os


"""
Train an auditive som, test it alongside the visual one
"""

somv_path = os.path.join(Constants.DATA_FOLDER,
                         '10classes',
                         'visual_model')

somu_path = os.path.join(Constants.DATA_FOLDER,
                         '10classes',
                         'audio_model')

audio_data_path = os.path.join(Constants.DATA_FOLDER,
                               '10classes',
                               'audio_data_25t.csv')

if __name__ == '__main__':
    xs, ys, filenames = from_csv_with_filenames(audio_data_path)
    vect_size = len(xs[0])
    xs = MinMaxScaler().fit_transform(xs)
    audio_som = SOM(20, 30, vect_size, alpha=0.3, sigma=15, n_iterations=100, batch_size=1)
    audio_som.train(xs, input_classes=ys, test_vects=xs, test_classes=ys, save_every=100)
    # audio_som.train(xs)
    proto = get_prototypes(xs, [int(y) - 1000 for y in ys])
    to_csv(proto.T, ys, os.path.join(Constants.DATA_FOLDER,
                                     '10classes',
                                     'audio_prototypes.csv'))
    iterativeTraining(somv_path, somu_path)
