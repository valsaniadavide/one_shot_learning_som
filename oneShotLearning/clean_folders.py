import os, shutil
from utils.constants import Constants


def clean_folders(path):
    for the_file in os.listdir(path):
        file_path = os.path.join(path, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)


def clean_statistics_folders():
    clean_folders(os.path.join(Constants.PLOT_FOLDER, 'temp', 'som_confusion'))
    clean_folders(os.path.join(Constants.PLOT_FOLDER, 'temp', 'som_mapping'))
    clean_folders(os.path.join(Constants.SAVED_MODELS))
    clean_folders(os.path.join(Constants.TBLOGS))
