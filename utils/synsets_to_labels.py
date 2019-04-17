import json
import os
from constants import Constants

SYNSETS_FILE = os.path.join(Constants.DATA_FOLDER, 'synsets-labels.txt')
LABELS_FILE = os.path.join(Constants.DATA_FOLDER, 'imagenet-labels.json')
SYNSETS_IN_USE = os.path.join(Constants.DATA_FOLDER,
                              'labels100classes.txt')


def create_synset_dict():
    synsets = []
    with open(SYNSETS_FILE, 'r') as f:
        lines = f.readlines()
        synsets = [line.split(' ')[0] for line in lines]
    with open(LABELS_FILE, 'r') as f:
        labels_dict = json.load(f)
    new_dict = {synset: i for i, synset in enumerate(synsets)}
    return new_dict


def create_reduced_synset_dict():
    new_dict = create_synset_dict()
    with open(SYNSETS_IN_USE, 'r') as f:
        lines = f.readlines()
        synsets_100 = [line.strip('\n') for line in lines]
    reduced_dict = {k: v for k, v in new_dict.items() if k in synsets_100}
    return reduced_dict


if __name__ == '__main__':
    print(create_reduced_synset_dict())
