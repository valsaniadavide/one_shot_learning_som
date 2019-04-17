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

import matplotlib
#matplotlib.use('Agg')
#from matplotlib import pyplot as plt
import numpy as np
from colour import Color
from .SOM import SOM
import os
import math
import random
from numpy.linalg import norm
from utils.constants import Constants


dimN = 20
dimM = 30
numIterations = 400
N = 1000
NxClass = 100

def restoreSOM(checkpoint_dir,lenExamples):
    """
        restore the som which model is in the checkpoint_dir
    """
    som = SOM(dimN, dimM, lenExamples, checkpoint_dir= checkpoint_dir, n_iterations=numIterations)

    loaded = som.restore_trained()
    if not loaded:
        raise ValueError("SOM in "+checkpoint_dir+" not trained yet")

    return som

def getInputClass(className,fileInput):
    """
        Read the input file and extract the names of the classes used
    """
    f = open(fileInput,'r')
    inputC = None
    for l in f:
        lSplit = l.split(',')
        if className in lSplit[0]:
            print(lSplit[0])
            inputC = np.array(lSplit[1:]).astype(float)
            break
    return inputC


def getRandomInputClass(className,fileInput):
    """
        Return a random input from the class className
    """
    f = open(fileInput,'r')
    inputC = None
    for l in f:
        v = np.random.randint(4)
        if v == 3:
            lSplit = l.split(',')
            if className in lSplit[0]:
                print (lSplit[0])
                inputC = np.array(lSplit[1:]).astype(float)
                break
    f.close()
    return inputC

def getAllInputClass(className,fileInput):
    """
        Return all the input of the class className
    """
    f = open(fileInput,'r')
    inputC = []
    for l in f:
        lSplit = l.split(',')
        if str(className) in lSplit[0]:
            inputC.append(np.array(lSplit[1:]).astype(float))
    f.close()
    return inputC

def getAllInputClassAudio(className, file_path):
    f = open(file_path,'r')
    inputC = None
    for l in f:
        lSplit = l.split(',')
        if str(className) in lSplit[-1]:
            inputC = np.array(lSplit[1:-1]).astype(float)
    return inputC

def showSomActivations(activations,posActivations,count,title):
    """
        Shows the SOM with its activations
    """
    image_grid = np.zeros(shape=(dimN,dimM,3))
    plt.figure(count)
    plt.title(title)
    plt.imshow(image_grid)

      #normalization of the values of the activations
    maxA = max(activations)
    print(maxA)

    minA = min(activations)
    print(minA)

    a = 0
    b = 1

    for i in range(len(activations)):
        t = ((b-a)*(activations[i]-minA))/(maxA-minA) + a
        ## suppression:
        #if t < 0.6:
        #    t = 0.0
        print(i)
        plt.text(posActivations[i][1], posActivations[i][0], '..', ha='center', va='center',
                bbox=dict(facecolor=str(t), alpha=1, lw=0))

    plt.draw()
    plt.show()
    return plt

def getAllInputs(fInput,lenExample):
    """
        Read all the inputs from a file
    """
    inputs = np.zeros(shape=(N,lenExample))
    nameInputs = list()

    with open(fInput, 'r') as inp:
      i = 0
      for line in inp:
        if len(line)>2:
          inputs[i] = (np.array(line.split(',')[1:])).astype(np.float)
          nameInputs.append((line.split(',')[0]).split('/')[5])
          i = i+1
    #nameInputs.sort()
    return [inputs,nameInputs]

def showSom(som,inputs,nameInputs,count,title):
    """
        Shows the SOM highlighting the BMU of each input
    """
    print('costruzione mappa '+title)
    mapped = som.map_vects(inputs)
    image_grid = np.zeros(shape=(20,30,3))
    plt.figure(count)
    plt.imshow(image_grid)
    plt.title(title)
    inputClass = nameInputs[0]

    #color generation
    # classColor = list()
    # for i in range(10):
    #     c = Color(rgb=(random.random(), random.random(), random.random()))
    #     classColor.append(str(c))
    classColor = ['white','red','blue','cyan','yellow','green','gray','brown','orange','magenta']

    iColor = 0

    lenExample = len(inputs[0])
    print(lenExample)
    prototipi = classPrototype(inputs,nameInputs)

    print(inputClass+' -- '+classColor[iColor])

    for i, m in enumerate(mapped):
      if nameInputs[i] != inputClass:
        inputClass = nameInputs[i]
        iColor = iColor + 1
        print(inputClass+' -- '+classColor[iColor])


      plt.text(m[1], m[0], '....', ha='center', va='center',
            bbox=dict(facecolor=classColor[iColor], alpha=0.5, lw=0))

    # for k in prototipi.keys():
    #     [BMUi, BMUpos] = som.get_BMU(prototipi[k])
    #     plt.text(BMUpos[1], BMUpos[0], str(k), ha='center', va='center',
    #             bbox=dict(facecolor='white', alpha=0.9, lw=0))
    plt.draw()

    return plt

def printToFileCSV(prototipi,file):
  """
        print the prototypes in a csv file
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

def classPrototype(inputs,nameInputs):
  """
        extract for each class the prototype staring from the inputs of that class
  """
  protClass = dict()
  nameS = list(set(nameInputs))
  nameS = np.sort(nameS)
  print(nameS)

  temp = np.array(inputs)

  i = 0
  for name in nameS:
    protClass[name] = np.mean(temp[i:i+NxClass][:],axis=0)
    i = i + NxClass

  #printToFileCSV(protClass,'./prototipiVisivi.csv')
  return protClass

def updatesynapses(S,classes,SOMU,SOMV,INPUTV,INPUTU,ite,maxIter):
    """
        update all the synpases between the SOMU (auditory) and the SOMV (visual)
        based on the activation produced by the inputs INPUTV (visual) and INPUTU (auditory)
    """
    print('updating synapses')
    # initializations of the synapses
    # S: matrix of size numberOfAuditoryNeurons X numberOfVisualNeurons

    if S == None:
        m = 1/math.sqrt(dimN*dimM*dimN*dimM)
        sd = 1/(1000*math.sqrt(dimN*dimM*dimN*dimM))
        S = np.random.normal(m,sd,(dimN*dimM,dimN*dimM))

    ATTIVAZIONIV = dict()
    ATTIVAZIONIU = dict()
    posAttivazioniU = dict()
    posAttivazioniV = dict()

    lambdaP = 5.0

    #normalization of the values

    a = 0.0
    b = 10.0

    count = 0

    for c in classes:
        print('updating for class '+str(c))

        print('generating visual activation')
        [ATTIVAZIONIV[c],posAttivazioniV[c]] = SOMV.get_activations(INPUTV[c])


        print('suppresion low values')
        maxA = np.amax(np.amax(ATTIVAZIONIV[c]))
        minA = np.amin(np.amin(ATTIVAZIONIV[c]))

        for i in range(len(ATTIVAZIONIV[c])):
            #print(type(ATTIVAZIONIV[c][i]))
            #float(10.0 * (ATTIVAZIONIV[c][i]-minA))
            #float(maxA-minA)
            ATTIVAZIONIV[c][i] = (float(10.0 * (ATTIVAZIONIV[c][i]-minA))/float(maxA-minA))

            #soppressione valori bassi
            #print(type(ATTIVAZIONIV[c][i]))

            if ATTIVAZIONIV[c][i] < 6.0:
                ATTIVAZIONIV[c][i] = 0.0

        #showSomActivations(ATTIVAZIONIV[c],posAttivazioniV[c],count,'visive per classe'+c)

        count += 1
        #uditiva
        print('generating auditory activation')
        [ATTIVAZIONIU[c],posAttivazioniU[c]] = SOMU.get_activations(INPUTU[c])

        print('suppresion low values')
        maxA = np.amax(np.amax(ATTIVAZIONIU[c]))
        minA = np.amin(np.amin(ATTIVAZIONIU[c]))
        for i in range(len(ATTIVAZIONIU[c])):
            ATTIVAZIONIU[c][i] = (float(10.0 * (ATTIVAZIONIU[c][i]-minA))/float(maxA-minA))
            #soppressione valori bassi
            if ATTIVAZIONIU[c][i] < 6.0:
                ATTIVAZIONIU[c][i] = 0.0
        #showSomActivations(ATTIVAZIONIU[c],posAttivazioniU[c],count,'uditive per classe'+c)

        count += 1

        print('updating synapses')
        for i in range(len(ATTIVAZIONIU[c])):
            #for j in range(len(ATTIVAZIONIV[c])):
            #    S[i][j] = S[i][j] + 1 - math.exp(-lambdaP * ATTIVAZIONIU[c][i]*ATTIVAZIONIV[c][j])
            a = np.asarray(ATTIVAZIONIV[c], dtype=np.float32)
            S[i] = S[i] + np.ones(dimN*dimM) - np.exp(- lambdaP * ATTIVAZIONIU[c][i]*a)


        count = count + 3

    print('maxS ---->>> '+str(np.amax(np.amax(S))))
    print('minS ---->>> '+str(np.amin(np.amin(S))))

    if (ite == (maxIter-1)):
        tot = np.sum(np.sum(S))
        print('normalization')
        for i in range(len(S)):
            for j in range(len(S[i])):
                S[i][j] = S[i][j]/tot

    print('maxS ---->>> '+str(np.amax(np.amax(S))))
    print('minS ---->>> '+str(np.amin(np.amin(S))))
    plt.show()
    return S


def updatesynapsesPreLoad(S,classes,SOMU,SOMV,INPUTV,INPUTU,ite,maxIter):
    """
        update all the synpases between the SOMU (auditory) and the SOMV (visual)
        based on the activation produced by the inputs INPUTV (visual) and INPUTU (auditory)
        The activations are already calculated
    """
    print('updating synapses')
    # initializations of the synapses
    # S: matrix of size numberOfAuditoryNeurons X numberOfVisualNeurons

    if np.all(S == 0):
        m = 1/math.sqrt(dimN*dimM*dimN*dimM)
        sd = 1/(1000*math.sqrt(dimN*dimM*dimN*dimM))
        S = np.random.normal(m,sd,(dimN*dimM,dimN*dimM))

    lambdaP = 5.0

    # normalization

    a = 0.0
    b = 10.0

    count = 0

    for c in classes:

        # updating synapses
        for i in range(len(INPUTU[c])):
            #for j in range(len(INPUTV[c])):
            #    S[i][j] = S[i][j] + 1 - math.exp(-lambdaP * ATTIVAZIONIU[c][i]*INPUTV[c][j])
            a = np.asarray(INPUTV[c], dtype=np.float32)
            S[i] = S[i] + np.ones(dimN*dimM) - np.exp(- lambdaP * INPUTU[c][i]*a)


    #print('maxS ---->>> '+str(np.amax(np.amax(S))))
    #print('minS ---->>> '+str(np.amin(np.amin(S))))

    tot = np.sum(np.sum(S))
    #print('normalization')
    if ite == maxIter:
        S = S/tot

    #print('maxS ---->>> '+str(np.amax(np.amax(S))))
    #print('minS ---->>> '+str(np.amin(np.amin(S))))

    return S

def savesynapses(S,outputFile):
    """
        save the synapses on the outputFile
    """
    output = open(outputFile,'w')

    for i in range(len(S)):
        for j in range(len(S[i])):
            output.write(str(S[i][j])+',')
        output.write(';\n')

    output.close()

def restoresynapses():
    """
        restore the synapses from a specific file
    """
    outputFile = './sinapses.csv'
    try:
        output = open(outputFile,'r')
    except:
        return None

    m = 1/math.sqrt(dimN*dimM*dimN*dimM)
    sd = 1/(1000*math.sqrt(dimN*dimM*dimN*dimM))
    S = np.random.normal(m,sd,(dimN*dimM,dimN*dimM))
    i = 0
    for l in output:
        j = 0
        lSplit = l.split(',')
        for v in lSplit:
            if (len(v)>0) and (not ';' in v):
                S[i][j] = float(v)
            j = j+1
        i = i+1

    return S

def propagateActivations(UV,bmu1,S):
    """
        propagate the activations of the bmus from a SOM to the other
    """
    # locAct = list()
    # for i in range(0,dimN-1):
    #     for j in range(0,dimM-1):
    #         locAct.append([i,j])

    act = list()
    locAct = list()
    m = 0
    for i in range(0,dimN-1):
        for j in range(0,dimM-1):
            if UV == 'U':
                act.append(S[bmu1][dimM*i+j])
                locAct.append([i,j])
                if S[bmu1][dimM*i+j] > S[bmu1][m]:
                    m = dimM*i+j
            else:
                act.append(S[dimM*i+j][bmu1])
                locAct.append([i,j])
                if S[dimM*i+j][bmu1] > S[m][bmu1]:
                    m = dimM*i+j

    #showSomActivations(act,locAct,100,'attivazione propagata')
    return m

def propagateActivationsAll(UV,act2,S):
    """
        propagate the activations of all the neurons from a SOM to the other
    """
    # locAct = list()
    # for i in range(dimN):
    #     for j in range(dimM):
    #         locAct.append([i,j])

    act = list()
    #locAct = list()
    #m = 0

    #act = sum(S[:][:] * act2)
    #print(len(t))

    act = np.dot(act2,S)

    # for i in range(dimM*dimN):
    #      t = sum(S[i][:] * act2)
    #      act.append(t)


    #    locAct.append([i,j])

    # for i in range(0,dimN-1):
    #     for j in range(0,dimM-1):
    #         if UV == 'U':
    #             t = 0
    #             for i2 in range(0,dimN-1):
    #                 for j2 in range(0,dimM-1):
    #                     t += S[dimM*i2+j2][dimM*i+j]*act2[dimM*i2+j2]
    #             act.append(t)
    #             locAct.append([i,j])

    #         else:
    #             ## medium old
    #             # t = 0
    #             # for k in range(dimN*dimM):
    #             #     t += S[dimM*i+j][k] * act2[k]

    #             ## new
    #             t = sum(S[dimM*i+j][:] * act2)

    #             ## old
    #             # for i2 in range(0,dimN-1):
    #             #     for j2 in range(0,dimM-1):
    #             #         t += S[dimM*i+j][dimM*i2+j2]*act2[dimM*i2+j2]

    #             act.append(t)
    #             locAct.append([i,j])

    m = np.argmax(act)
    #m = act.index(ml)
    #showSomActivations(act,locAct,100,'attivazione propagata')

    return m

def showActivationsS(c,S,count):
    """
        propagate the activation and show the consequent som
    """
    locAct = list()
    for i in range(0,dimN-1):
        for j in range(0,dimM-1):
            locAct.append([i,j])

    #print(locAct)

    act = list()
    for i in range(0,dimN-1):
        for j in range(0,dimM-1):
            act.append(S[dimM*i+j])
    print(act)

    showSomActivations(act,locAct,count,'activations propagated with class '+c)


def distanceIntraClass(SOM, inputs, nameInputs):
    """
        calculate the intra-cluster and inter-cluster distances based on a specific som and a set of inputs;
        the clusters are the set of bmus that belong to the same class of objects
    """
    print('- extraction bmus -')
    mapped = SOM.map_vects(inputs)

    positions = dict()
    for i, m in enumerate(mapped):
        if nameInputs[i] in positions:
            positions[nameInputs[i]].append([m[1],m[0]])
        else:
            positions[nameInputs[i]] = [[m[1],m[0]]]


    distancesIntra = dict()
    print('- intra-cluster distance - ')
    posKey = positions.keys()
    posKey.sort()
    for c in posKey:
        d = 0
        count = 0
        for i in positions[c]:
            i1 = positions[c].index(i)
            for j in positions[c][i1+1:]:
                d += math.sqrt(( (j[0]-i[0])**2 ) + ((j[1]-i[1])**2 ))
                count += 1

        distancesIntra[c] = d / count

        print('--- '+c+' -> '+str(distancesIntra[c]))


    allPositions = []
    print('- inter-cluster distance :')
    for c in posKey:
        for p in positions[c]:
            allPositions.append(p)
    d = 0
    count = 0
    for i in allPositions:
        i1 = allPositions.index(i)
        for j in allPositions[i1+1:]:
            d += math.sqrt(( (j[0]-i[0])**2 ) + ((j[1]-i[1])**2 ))
            count += 1

    distancesExtra = d / count


    print('- ratio between the intra and inter cluster distances')
    for c in posKey:
        print(str(c) + ';' + str(distancesIntra[c]/distancesExtra))





def getBMUUPositions(som_path):
    """
        get the positions of the BMU in a som starting from a set of inputs
    """
    classes = list(range(0,10))

    inputs = []
    for c in classes:
        inputs.append(getRandomInputClass(c,os.path.join(Constants.DATA_FOLDER, 'input10classes', 'auditoryInput.csv')))

    SOMU = restoreSOM(som_path,10)

    mapped = SOMU.map_vects(inputs)

    out = np.zeros(shape=(10,2))

    for i, m in enumerate(mapped):
        out[i] = [m[0],m[1]]

    return out

def getActivationsOnce(SOMV,SOMU,inputsV,inputsU):
    """
        calculate the activations on the two soms starting from two sets of inputs
    """
    activations = dict()
    activations['U'] = dict()
    activations['V'] = dict()
    for c in inputsU.keys():
        j = 0
        activations['U'][c] = dict()
        for u in inputsU[c]:
            activations['U'][c][j] = SOMU.get_activations(u)
            maxA = np.amax(np.amax(activations['U'][c][j][0]))
            minA = np.amin(np.amin(activations['U'][c][j][0]))
            for i in range(len(activations['U'][c][j][0])):
                activations['U'][c][j][0][i] = (float(10.0 * (activations['U'][c][j][0][i]-minA))/float(maxA-minA))
                if activations['U'][c][j][0][i] < 6.0:
                    activations['U'][c][j][0][i] = 0.0
            j += 1
    for c in inputsV.keys():
        j = 0
        activations['V'][c] = dict()
        for v in inputsV[c]:
            activations['V'][c][j] = SOMV.get_activations(v)
            maxA = np.amax(np.amax(activations['V'][c][j][0]))
            minA = np.amin(np.amin(activations['V'][c][j][0]))
            for i in range(len(activations['V'][c][j][0])):
                activations['V'][c][j][0][i] = (float(10.0 * (activations['V'][c][j][0][i]-minA))/float(maxA-minA))
                if activations['V'][c][j][0][i] < 6.0:
                    activations['V'][c][j][0][i] = 0.0
            j += 1

    return activations

def getBMUonce(SOMV,inputsV):
    """
        calculate the bmus on the a SOM starting from a set of inputs
    """
    print('recupero bmus')
    bmus = dict()
    for c in inputsV.keys():
        print('classe '+str(c))
        bmus[c] = dict()
        for i in range(len(inputsV[c])):
            bmus[c][i] = SOMV.get_BMU(inputsV[c][i])[0]

    return bmus


def iterativeTraining(img_som_path, audio_som_path):
    """
        calculate the taxonomic factor increasing the number of couples
        used for the training of the hebbian connections
    """
    classes = list(range(0,10))

    SOMV = restoreSOM(img_som_path,2048)
    SOMU = restoreSOM(audio_som_path,1000)

    INPUTV = dict()
    INPUTU = dict()
    tinputU = dict()
    tinputV = dict()

    for c in classes:
        INPUTV[c] = getAllInputClass(c, os.path.join(Constants.DATA_FOLDER, '10classes', 'VisualInputTestSet.csv'))
        INPUTU[c] = getAllInputClassAudio(c, os.path.join(Constants.DATA_FOLDER, '10classes', 'audio_prototypes.csv'))

    print('getActivationsOnce')
    activations = getActivationsOnce(SOMV,SOMU,INPUTV,INPUTU)

    print('getBMUonce')
    bmus = getBMUonce(SOMV,INPUTV)

    print('getBMUUPositions')
    # commenting this out because it does not seem to be used further in this function
    #posClassesU = getBMUUPositions()

    niterRes = []
    for ni in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]:
        corrects = []
        niter = ni

        for k in range(1000):
            #use 1000 random training set for each number of couples
            print('iteration -> '+str(k))
            S = np.zeros((10, 60))
            for i in range(niter):
                i0 = np.random.randint(100)
                tinputV[classes[0]] = activations['V'][classes[0]][i0][0]
                i1 = np.random.randint(100)
                tinputV[classes[1]] = activations['V'][classes[1]][i1][0]
                i2 = np.random.randint(100)
                tinputV[classes[2]] = activations['V'][classes[2]][i2][0]
                i3 = np.random.randint(100)
                tinputV[classes[3]] = activations['V'][classes[3]][i3][0]
                i4 = np.random.randint(100)
                tinputV[classes[4]] = activations['V'][classes[4]][i4][0]
                i5 = np.random.randint(100)
                tinputV[classes[5]] = activations['V'][classes[5]][i5][0]
                i6 = np.random.randint(100)
                tinputV[classes[6]] = activations['V'][classes[6]][i6][0]
                i7 = np.random.randint(100)
                tinputV[classes[7]] = activations['V'][classes[7]][i7][0]
                i8 = np.random.randint(100)
                tinputV[classes[8]] = activations['V'][classes[8]][i8][0]
                i9 = np.random.randint(100)
                tinputV[classes[9]] = activations['V'][classes[9]][i9][0]
                for c in classes:
                    random_sample = np.random.randint(len(list(INPUTU)))
                    tinputU[c] = activations['U'][c][random_sample][0]

                S = updatesynapsesPreLoad(S,classes,SOMU,SOMV,tinputV,tinputU,i,niter-1)

                #print(str(i0)+'--'+str(i1)+'--'+str(i2)+'--'+str(i3)+'--'+str(i4)+'--'+
                #    str(i5)+'--'+str(i6)+'--'+str(i7)+'--'+str(i8)+'--'+str(i9)+'--')

            S = np.matrix.transpose(S)
            correct = 0
            #given the synapses built, I evaluate the performance of the model propagating the activation
            # of each visual input to the auditory SOM, finding the BMU and the nearer auditory prototype BMU.
            print('test ...')
            for cl in classes:
                #print('test with class '+cl)
                for j in range(100):
                    #bmuV = bmus[cl][j]

                    #bmuU = propagateActivations('V',bmuV,S)

                    act2 = activations['V'][cl][j]
                    #print(act2)

                    bmuU = propagateActivationsAll('V',act2[0],S)

                    # correct if: the nearer auditory prototype is associate to the input

                    # r = bmuU/dimM
                    # c = bmuU%dimM

                    # dist = np.array([])
                    # for i in range(10):
                    #     dist = np.append(dist,math.hypot(posClassesU[i][0] - r, posClassesU[i][1] - c ))

                    # #print(dist)

                    # if cl == classes[np.argmin(dist)]:
                    #     correct += 1

                    #correct if: the bmu of the propagated activation fall into an area associated with the class of the input

                    actTU = np.array([])
                    for cu in classes:
                        actTU = np.append(actTU,activations['U'][cu][0][0][bmuU])

                    if cl == classes[np.argmax(actTU)]:
                        correct += 1


            corrects.append(correct)
            #print(corrects)

        print(corrects)
        print(sum(corrects) / float(len(corrects)))
        niterRes.append(sum(corrects) / float(len(corrects)))
        print('--------')
        print(niterRes)





def testWordLearning():
    """
        Test the taxonomic factor of the model increasing the number of couples used for each class.
    """
    classesIn = open('./utility/labels10classes.txt','r')
    classes = []
    for c in classesIn:
        classes.append(c[:-1])

    classes.sort()

    modelDirSomV = './VisualModel10classes/'
    modelDirSomU = './AuditoryModel10classes/'

    SOMV = restoreSOM(modelDirSomV,2048)
    SOMU = restoreSOM(modelDirSomU,10)



    S = restoresynapses()
    maxIter = 10
    if S == None:
        for i in range(0,maxIter):
            INPUTV = dict()
            INPUTU = dict()
            for c in classes:
                INPUTV[c] = getRandomInputClass(c,'./input10classes/VisualInputTestSet.csv')
                INPUTU[c] = getRandomInputClass(c,'./input10classes/auditoryInput.csv')

            S = updatesynapses(S,classes,SOMU,SOMV,INPUTV,INPUTU,i,maxIter)
            print('------- '+str(i)+'--------'+str(maxIter))

        savesynapses(S,'./sinapses.csv')

    ###################################
    #        TEST
    ###################################

    # print('Production test')
    # inputTestU = getRandomInputClass('fishes','./input10classes/auditoryInput.csv')

    # [actU,locActU] = SOMU.get_activations(inputTestU)

    # [bmuU,locBmuU] = SOMU.get_BMU(inputTestU)

    # bmuV = propagateActivations('U',bmuU,S)

    # #bmuV = np.where(S[bmuU,:] == max(S[bmuU,:]))
    # #bmuV = bmuV[0][0]

    # for r in range(dimN):
    #     Fbreak = False
    #     for c in range(dimM):
    #         if dimM*r + c == bmuV:
    #             Fbreak = True
    #             break
    #     if Fbreak == True:
    #         break

    # print(r,c)

    # # visualization
    # [iVall,iVallName] = getAllInputs('input10classes/VisualInputTestSet',2048)
    # [iUall,iUallName] = getAllInputs('input10classes/auditoryInput.csv',N/NxClass)
    # pltU = showSom(SOMU,iUall,iUallName,20,'Uditiva')

    # pltU.text(locBmuU[1], locBmuU[0], 'UUU', ha='center', va='center',
 #            bbox=dict(facecolor='white', alpha=1, lw=0))

    # pluV = showSom(SOMV,iVall,iVallName,21,'Visiva')

    # pluV.text(c, r, 'VVV', ha='center', va='center',
 #            bbox=dict(facecolor='white', alpha=1, lw=0))


    # pluV.show()
    # pluV.show()
    # plt.show()

    ###################################
    #        TEST
    ###################################

    print('Comprehension test of the class bird')
    inputTestV = getRandomInputClass('birds','./input10classes/VisualInputTestSet.csv')

    [bmuV,locBmuV] = SOMV.get_BMU(inputTestV)

    bmuU = propagateActivations('V',bmuV,S)

    for r in range(dimN):
        Fbreak = False
        for c in range(dimM):
            if dimM*r + c == bmuU:
                Fbreak = True
                break
        if Fbreak == True:
            break

    print(r,c)

    # visualization
    print('get visive inputs')
    [iVall,iVallName] = getAllInputs('./input10classes/VisualInputTestSet.csv',2048)
    print('get auditory inputs')
    [iUall,iUallName] = getAllInputs('./input10classes/auditoryInput.csv',10)
    pltU = showSom(SOMU,iUall,iUallName,20,'Uditiva')

    pltU.text(c, r, 'UUU', ha='center', va='center',
            bbox=dict(facecolor='white', alpha=1, lw=0))

    pluV = showSom(SOMV,iVall,iVallName,21,'Visiva')

    pluV.text(locBmuV[1], locBmuV[0], 'VVV', ha='center', va='center',
            bbox=dict(facecolor='white', alpha=1, lw=0))

    pltU.show()
    pluV.show()

    plt.show()



if __name__ == '__main__':

    iterativeTraining()
