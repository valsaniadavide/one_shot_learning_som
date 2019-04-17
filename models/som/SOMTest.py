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
from colour import Color
from .SOM import SOM
import os
import math
import random
import matplotlib.patches as mpatches
from utils.constants import Constants

fInput = 'input10classes/VisualInputTrainingSet.csv'
N = 1000
lenExample = 2048
NumXClass = 10

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

def printToFileCSV(prototipi,file):
  """
    print of the prototypes in file.csv
    prototipi: dictionary of the prototypes to print
  """

  f = open(file,'w')

  # stampa su file
  for k in prototipi.keys():
    st = k+','
    for v in prototipi[k]:
      st += str(v)+','
    st = st[0:-1]
    f.write(st+'\n')

  f.close()

def showSom(som,inputs,nameInputs,count,title, filenames=None):
  """
    build of the map with the color associated to the different classes
  """
  print('costruzione mappa '+title)
  mapped = som.map_vects(inputs)
  image_grid = np.zeros(shape=(20,30,3))
  plt.figure(count)
  plt.imshow(image_grid)
  plt.title(title)
  inputClass = nameInputs[0]

  # color generation
  classColor = list()
  ## for 100 classes
  # for i in range(100):
  #   print(i)
  #   c = Color(rgb=(random.random(), random.random(), random.random()))
  #   classColor.append(str(c))
  ## for 10 classes:
  classColor = ['white','red','blue','cyan','yellow','green','gray','brown','orange','magenta']
  color_dict = create_color_dict(nameInputs, classColor)

  if filenames == None:
    for i, m in enumerate(mapped):
      plt.text(m[1], m[0], str('____'), ha='center', va='center', color=color_dict[nameInputs[i]], alpha=0.5,
          bbox=dict(facecolor=color_dict[nameInputs[i]], alpha=0.6, lw=0, boxstyle='round4'))
  else:
    for i, m in enumerate(mapped):
      plt.text(m[1], m[0], str('_{:03d}_'.format(i)), ha='center', va='center', color=color_dict[nameInputs[i]], alpha=0.5,
          bbox=dict(facecolor=color_dict[nameInputs[i]], alpha=0.6, lw=0, boxstyle='round4'))
      print('{}: {}'.format(i, filenames[i]))


  ## draw of the prototypes on the map
  # for k in prototipi.keys():
  #     [BMUi, BMUpos] = som.get_BMU(prototipi[k])
  #     plt.text(BMUpos[1], BMUpos[0], str(k), ha='center', va='center',
  #             bbox=dict(facecolor='white', alpha=0.9, lw=0))
  plt.draw()


  # draw a legend
  plt.figure()
  reverse_color_dict = {v: k for k, v in color_dict.items()}
  patch_list = []
  for i in range(len(classColor)):
      patch = mpatches.Patch(color=classColor[i], label=reverse_color_dict[classColor[i]])
      patch_list.append(patch)
  plt.legend(handles=patch_list)

  plt.savefig(os.path.join(Constants.PLOT_FOLDER, 'viz_som.png'))
  return plt


def classPrototype(inputs,nameInputs):
  #build the prototypes of the different classes
  protClass = dict()
  nameS = list(set(nameInputs))
  temp = np.array(inputs)

  i = 0
  for name in nameS:
    protClass[name] = np.mean(temp[i:i+NumXClass][:],axis=0)
    i = i + NumXClass

  #printToFileCSV(protClass,'prototipi.csv')
  return protClass



if __name__ == '__main__':
  #read the inputs from the file fInput and show the SOM with the BMUs of each input

  inputs = np.zeros(shape=(N,lenExample))
  nameInputs = list()

  # read the inputs
  with open(fInput, 'r') as inp:
      i = 0
      for line in inp:
        if len(line)>2:
          inputs[i] = (np.array(line.split(',')[1:])).astype(np.float)
          nameInputs.append((line.split(',')[0]).split('/')[6])
          print(nameInputs)
          i = i+1

  prototipi = classPrototype(inputs,nameInputs)

  #get the 20x30 SOM or train a new one (if the folder does not contain the model)
  som = SOM(20, 30, lenExample, checkpoint_dir= './AudioModel10classes/', n_iterations=20,sigma=4.0)

  loaded = som.restore_trained()
  if not loaded:
    som.train(inputs)

  for k in range(len(nameInputs)):
    nameInputs[k] = nameInputs[k].split('_')[0]

  #shows the SOM
  showSom(som,inputs,nameInputs,1,'Visual map')
