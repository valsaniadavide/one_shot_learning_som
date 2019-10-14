import os

from oneShotLearning.utility import clean_folders
from utils.constants import Constants


def clean_statistics_folders():
    clean_folders(os.path.join(Constants.PLOT_FOLDER, 'temp', 'test_data', 'som_confusion'))
    clean_folders(os.path.join(Constants.PLOT_FOLDER, 'temp', 'training_data', 'som_confusion'))
    clean_folders(os.path.join(Constants.PLOT_FOLDER, 'temp', 'test_data', 'som_mapping'))
    clean_folders(os.path.join(Constants.PLOT_FOLDER, 'temp', 'training_data', 'som_mapping'))
    clean_folders(os.path.join(Constants.SAVED_MODELS))
    clean_folders(os.path.join(Constants.TBLOGS))
