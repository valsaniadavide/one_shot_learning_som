# Copyright 2017 Giorgia Fenoglio
#
# This file is part of NNsTaxonomicResponding.
#
# NNsTaxonomicResponding is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# NNsTaxonomicResponding is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NNsTaxonomicResponding.  If not, see <http://www.gnu.org/licenses/>.
from matplotlib import pyplot as plt
import numpy as np
from .SOM import SOM
import os
import matplotlib.patches as m_patches
import seaborn as sb
from utils.constants import Constants
import matplotlib

# matplotlib.use('TkAgg')

fInput = 'input10classes/VisualInputTrainingSet.csv'
N = 1000
lenExample = 2048
NumXClass = 10


def material_palette():
    cols = [
        (229, 57, 53),  # red
        (171, 71, 188),  # purple
        (63, 81, 181),  # indigo
        (3, 155, 229),  # blue
        (0, 137, 123),  # teal
        (124, 179, 66),  # lighgreen
        (240, 98, 146),  # pink
        (255, 152, 0),  # orange
        (109, 76, 65),  # brown
        (96, 125, 139)  # grey
    ]
    return [tuple([v / float(255) for v in c]) for c in cols]


def create_color_dict(ys, colors):
    unique_y = len(set(ys))
    d = {}
    assigned_colors = []
    assigned_labels = []
    i = 0
    j = 0
    while len(assigned_colors) < len(colors):
        if colors[j] not in assigned_colors and ys[i] not in assigned_labels:
            d[ys[i]] = colors[j]
            assigned_colors.append(colors[j])
            assigned_labels.append(ys[i])
            j += 1
            i += 1
    print(d)
    return d


def printToFileCSV(prototipi, file):
    """
      print of the prototypes in file.csv
      prototipi: dictionary of the prototypes to print
    """
    f = open(file, 'w')
    # stampa su file
    for k in prototipi.keys():
        st = k + ','
        for v in prototipi[k]:
            st += str(v) + ','
            st = st[0:-1]
            f.write(st + '\n')
    f.close()


def show_som(som, xs, ys, labels, title, files=None, show=False, dark=True, scatter=True, point_size=48, legend=True,
             suffix='', subpath='test_data'):
    """
    Generates a plot displaying the SOM with its active BMUs and the relative examples
    associated with them. Each class is associated with a different color.
    :param SOM som:     SOM instance initialized with the wanted parameters (possibly already trained)
    :param arr xs:  np array (n, dims) containing data to be displayed on the map
    :param arr labels:  np array (n,) containing a label for each example
    :param str title:   title for the plot
    :param arr files:   actually no clue
    :param bool show:   whether to call show() or save to file
    """
    # matplotlib.use('TkAgg')  # in order to print something
    matplotlib.rcParams.update({'font.size': 16})
    # print('Building graph "{}"...'.format(title))
    classes = np.unique(ys)
    mapped = np.array(som.map_vects(xs))

    bmu_list = []
    for c in classes:
        class_bmu = mapped[np.where(ys == c)]
        bmu_list.append(np.unique(class_bmu, axis=0, return_counts=True))
    # print('Done mapping inputs, preparing canvas...')
    # for 80 classes readability
    palette = 'cubehelix' if len(classes) > 10 else "bright"

    fig = plt.figure(figsize=(11, 6.5))
    if dark:
        plt.gca().set_facecolor((0.15, 0.15, 0.15))

    plt.xlim([-1, som._n])
    plt.ylim([-1, som._m])
    plt.gca().set_xticks(np.arange(-1, som._n, 1))
    plt.gca().set_yticks(np.arange(-1, som._m, 1))
    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])
    plt.gca().tick_params(axis=u'both', which=u'both', length=0)
    plt.gca().grid(alpha=0.2, linestyle=':', color='white' if dark else 'black')
    # plt.title(title)

    # generate colors based on the # of classes
    np.random.seed(42)
    # colors = ['white', 'red', 'blue', 'cyan', 'yellow', 'green', 'gray', 'brown', 'orange', 'magenta']
    colors = sb.color_palette(palette, n_colors=len(classes))
    color_dict = {label: col for label, col in zip(classes, colors)}

    # print('Adding labels for each mapped input...', end='')
    if files == None:
        if scatter:
            for i, (bmu, counts) in enumerate(bmu_list):
                xx = [m[1] for m in bmu]
                yy = [m[0] for m in bmu]
                size = point_size / 2 + np.log(1 + counts ** 2) * point_size
                # size = counts * point_size
                plt.scatter(xx, yy, s=size, color=colors[i], alpha=0.6 if dark else 0.9)
        else:
            for i, m in enumerate(mapped):
                plt.text(m[1], m[0], str('__'), ha='center', va='center', color=color_dict[ys[i]], alpha=0.5,
                         bbox=dict(facecolor=color_dict[ys[i]], alpha=0.8, lw=0, boxstyle='round4'))
    else:
        for i, m in enumerate(mapped):
            plt.text(m[1], m[0], str('_{:03d}_'.format(i)), ha='center', va='center', color=color_dict[ys[i]],
                     alpha=0.5,
                     bbox=dict(facecolor=color_dict[ys[i]], alpha=0.8, lw=0, boxstyle='round4'))
            print('{}: {}'.format(i, files[i]))
    # print('done.')

    # print('Drawing legend...')
    patch_list = []
    for i in range(len(classes)):
        patch = m_patches.Patch(color=colors[i], label=labels[i])
        patch_list.append(patch)

    if legend & len(classes) <= 10:
        plt.legend(handles=patch_list, loc='center left', bbox_to_anchor=(1, 0.5))
    img_name = 'som_{}x{}_s{}_a{}_{}.png'.format(som._m, som._n, som.sigma, som.alpha, suffix)
    img_path = os.path.join(Constants.PLOT_FOLDER, 'temp', subpath, 'som_mapping', img_name)
    # print('Saving file: {} ...'.format(img_path))
    fig.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(img_path)
    plt.close()


def show_confusion(som, xs, ys, title="SOM confusion", palette="viridis", suffix='visual', subpath='test_data'):
    # matplotlib.use('TkAgg')  # in order to print something
    matplotlib.rcParams.update({'font.size': 16})
    # print('Building graph "{}"...'.format(title))
    classes = np.unique(ys)
    mapped = np.array(som.map_vects(xs))
    confusion_map = np.zeros((som._m, som._n))
    scaler = len(classes) / float(len(classes) - 1)  # max confusion when n out of n are different, so reverse

    # create a grid of lists, one for each node (in order to count the class freqs for each node)
    distributions_map = [[[] for j in range(som._n)] for i in range(som._m)]

    for y, bmu in zip(ys, mapped):
        distributions_map[bmu[0]][bmu[1]].append(y)

    for x in range(som._m):
        for y in range(som._n):
            vals = distributions_map[x][y]
            if len(vals) == 0:
                continue  # continue if the node is empty (is not a bmu)
            classes, counts = np.unique(vals, return_counts=True)
            if len(classes) <= 1:
                confusion_map[x, y] = 0
            else:
                counts = np.sort(-counts)
                confusion_map[x, y] = np.sum(counts[1:]) / np.sum(counts)

    # print('Done mapping inputs, preparing canvas...')
    img_name = 'confusion_map_som_{}x{}_s{}_a{}_{}.png'.format(som._m, som._n, som.sigma, som.alpha, suffix)
    img_path = os.path.join(Constants.PLOT_FOLDER, 'temp', subpath, 'som_confusion', img_name)
    plt.imshow(confusion_map * scaler, cmap=palette, origin="lower", clim=(0.0, 1.0))
    plt.axis('off')
    plt.colorbar()
    plt.savefig(img_path)
    # plt.show()
    plt.close()
    # print('Done, saved figure')


def classPrototype(inputs, nameInputs):
    # build the prototypes of the different classes
    protClass = dict()
    nameS = list(set(nameInputs))
    temp = np.array(inputs)

    i = 0
    for name in nameS:
        protClass[name] = np.mean(temp[i:i + NumXClass][:], axis=0)
        i = i + NumXClass

    # printToFileCSV(protClass,'prototipi.csv')
    return protClass


if __name__ == '__main__':
    # read the inputs from the file fInput and show the SOM with the BMUs of each input

    inputs = np.zeros(shape=(N, lenExample))
    nameInputs = list()

    # read the inputs
    with open(fInput, 'r') as inp:
        i = 0
        for line in inp:
            if len(line) > 2:
                inputs[i] = (np.array(line.split(',')[1:])).astype(np.float)
                nameInputs.append((line.split(',')[0]).split('/')[6])
                print(nameInputs)
                i = i + 1

    prototipi = classPrototype(inputs, nameInputs)

    # get the 20x30 SOM or train a new one (if the folder does not contain the model)
    som = SOM(20, 30, lenExample, checkpoint_dir='./AudioModel10classes/', n_iterations=20, sigma=4.0)

    loaded = som.restore_trained()
    if not loaded:
        som.train(inputs)

    for k in range(len(nameInputs)):
        nameInputs[k] = nameInputs[k].split('_')[0]
