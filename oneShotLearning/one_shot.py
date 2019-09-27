import os

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets.samples_generator import make_blobs

from utils.constants import Constants


def print_pca(xs, y, type='video'):
    pca = sklearnPCA(n_components=2)  # 2-dimensional PCA
    transformed = pd.DataFrame(pca.fit_transform(xs))
    transformed.plot(x=0, y=1)
    # plt.scatter(transformed[y == 1][0], transformed[y == 1][1], label='Class 1', c='red')
    # plt.scatter(transformed[y == 2][0], transformed[y == 2][1], label='Class 2', c='blue')
    # plt.scatter(transformed[y == 3][0], transformed[y == 3][1], label='Class 3', c='lightgreen')
    # plt.scatter(transformed[y == 1][0], transformed[y == 1][1], label='Class 1', c='yellow')
    # plt.scatter(transformed[y == 2][0], transformed[y == 2][1], label='Class 2', c='green')
    # plt.scatter(transformed[y == 3][0], transformed[y == 3][1], label='Class 3', c='grey')
    # plt.scatter(transformed[y == 1][0], transformed[y == 1][1], label='Class 1', c='black')
    # plt.scatter(transformed[y == 2][0], transformed[y == 2][1], label='Class 2', c='orange')
    # plt.scatter(transformed[y == 3][0], transformed[y == 3][1], label='Class 3', c='magenta')
    # plt.scatter(transformed[y == 3][0], transformed[y == 3][1], label='Class 3', c='lightblue')
    plt.legend()
    plt.savefig(os.path.join(Constants.PLOT_FOLDER, type))

