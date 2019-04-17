import csv
import json
import os
from utils.constants import Constants

def filter_synset_file():
    all_synsets_path = Constants.DATA_FOLDER + '/imagenet_synset_to_human_label_map.txt'
    used_synsets = Constants.DATA_FOLDER + '/thesis_synsets.txt'
    with open(all_synsets_path, 'r') as all_synsets_file:
        with open(used_synsets, 'r') as used_synsets_file:
            all_synsets = all_synsets_file.readlines()
            all_synsets = [s.strip('\n') for s in all_synsets]
            used_synsets = used_synsets_file.readlines()
            used_synsets = [s.strip('\n') for s in used_synsets]
    all_synsets_dict = {}
    for s in all_synsets:
        s_split = s.split('\t')
        all_synsets_dict[s_split[0]] = s_split[1]
    string = ""
    for s in used_synsets:
        descr = all_synsets_dict[s]
        string += s + ' ' + descr + '\n'
    with open(Constants.DATA_FOLDER + '/thesis_synsets_descr.txt', 'w') as text_file:
         text_file.write(string)

def synset_labels_dict():
    file_path = os.path.join(Constants.DATA_FOLDER, 'synsets-labels.txt')
    with open(file_path, 'r') as f:
        l = f.readlines()
    d = {}
    for i, li in enumerate(l):
        li = li.strip('\n')
        li = li.split(' ')[0]
        d[li] = i
    with open(os.path.join(Constants.DATA_FOLDER, 'synset_labels_dict.json'), 'w') as f:
        json.dump(d, f, indent=2)

def labels_synset_dict():
    file_path = os.path.join(Constants.DATA_FOLDER, 'synsets-labels.txt')
    with open(file_path, 'r') as f:
        l = f.readlines()
    d = {}
    for i, li in enumerate(l):
        li = li.strip('\n')
        li = li.split(' ')[0]
        d[i] = li
    with open(os.path.join(Constants.DATA_FOLDER, 'labels_synset_dict.json'), 'w') as f:
        json.dump(d, f, indent=2)

if __name__ == '__main__':
    synset_labels_dict()
    labels_synset_dict()
    filter_synset_file()
