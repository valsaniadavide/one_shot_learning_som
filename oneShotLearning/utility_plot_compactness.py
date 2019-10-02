import os
from utils.constants import Constants
import numpy as np
from oneShotLearning.one_shot_type_1 import label_classes
import matplotlib.pyplot as plt
import pandas as pd


def plot_data(row):
    alpha = row['alpha']
    sigma = row['sigma']
    x = np.arange(len(label_classes))  # the label locations
    width = 0.35  # the width of the bars
    print(type(row['v_compact']))
    print(row['v_compact'])
    fig, ax = plt.subplots()
    v_compact = row['v_compact'][1:-1]
    a_compact = row['a_compact'][1:-1]
    v_compact = v_compact.split()
    a_compact = a_compact.split()
    v_compact = list(map(float, v_compact))
    a_compact = list(map(float, a_compact))
    print(v_compact)
    rects1 = ax.bar(x - width / 2, v_compact, width, label='Visual')
    rects2 = ax.bar(x + width / 2, a_compact, width, label='Audio')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Mean Compactness')
    ax.set_title('Mean Compactness (a={}, s={})'.format(alpha, sigma))
    ax.set_xticks(x)
    ax.set_xticklabels(label_classes)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(round(height,2)),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    fig.tight_layout()
    image_path = os.path.join(Constants.ROOT_FOLDER, 'oneShotLearning', 'statistics', 'Test a.0.01-0.30_s.5-20',
                              'compactness', 'compactness_som_a={}_s={}.png'.format(alpha, sigma))
    plt.savefig(image_path)
    plt.close()


if __name__ == '__main__':
    path = os.path.join(Constants.ROOT_FOLDER, 'oneShotLearning', 'statistics', 'Test a.0.01-0.30_s.5-20',
                        'statistics.csv')
    df = pd.read_csv(path)
    print(df.head(n=10))
    for index, row in df.iterrows():
        plot_data(row)
