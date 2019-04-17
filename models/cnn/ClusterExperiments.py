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

from scipy.cluster.vq import kmeans2
from scipy import stats
import numpy as np

import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MiniBatchKMeans
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA

N = 1000
fIn = 'input10classs/visualInput.csv'
lenEx = 1000
classi = 10
ninput = 100

def getAllInputs(fInput,lenExample):
	inputs = np.zeros(shape=(N,lenExample))
	nameInputs = list()

	with open(fInput, 'r') as inp:
	  i = 0
	  count = 0
	  for line in inp:
	    if len(line)>2:
	    	if (count < ninput):
				inputs[i] = (np.array(line.split(',')[1:])).astype(np.float)
				nameInputs.append((line.split(',')[0]).split('/')[6])
				i = i+1
	    	count = count + 1
	    	if count==100:
		    	count = 0

	#nameInputs.sort()
	return [inputs,nameInputs]


def Kmeanscluster():
	print('lettura inputs da '+fIn)
	[inputs,nameInputs] = getAllInputs(fIn,lenEx)
	print('calcolo kmeans')
	[centroid,labels] = kmeans2(inputs,classi,iter=200,minit='random')

	for i in range(classi):
		print('-------------------------------------------')
		print('classe '+nameInputs[ninput*i])
		cLabels = labels[ninput*i:ninput*(i+1)]

		print(cLabels)

		mode = int(stats.mode(cLabels)[0][0])
		print('main class:'+str(mode))
		errors = len(cLabels)-(cLabels.tolist()).count(mode)
		print('errori: '+str(errors)+'/'+str(len(cLabels)))

def printPredictions(prediction,nameInputs):
	mode = list()
	errors = list()
	for i in range(classi):
		print('-------------------------------------------')
		print('classe '+nameInputs[ninput*i])
		cLabels = prediction[ninput*i:ninput*(i+1)]

		print(cLabels)

		mode.append(int(stats.mode(cLabels)[0][0]))
		print('\tmain class:'+str(mode[-1]))

		errors.append(len(cLabels)-(cLabels.tolist()).count(mode[-1]))
		print('errori: '+str(errors[-1])+'/'+str(len(cLabels)))

	print('classi riconosciute:')
	print(mode)
	print('errori per classe (su '+str(len(cLabels))+'):')
	print(errors)
	print(errors)
	print('errore medio:'+str(sum(errors) / float(len(errors))))

def printPredictionsK(prediction,nameInputs):
	mode = list()
	errors = list()
	for i in range(classi):
		print('-------------------------------------------')
		print('classe '+nameInputs[ninput*i])
		cLabels = prediction[ninput*i:ninput*(i+1)]

		print(cLabels)
		l = list(set(cLabels))
		l.sort()
		print('different classes: '+str(len(l)))
		for j in range(len(l)):
			print(str(l[j])+' contains '+str(cLabels.tolist().count(l[j]))+' values')

		#for all algorithms
		mode.append(int(stats.mode(cLabels)[0][0]))
		#for k-nearest
		#mode.append(stats.mode(cLabels)[0][0])

		print('\tmain class:'+str(mode[-1]))

		errors.append(len(cLabels)-(cLabels.tolist()).count(mode[-1]))
		print('errori: '+str(errors[-1])+'/'+str(len(cLabels)))

	print('classi riconosciute:')
	print(mode)
	print('errori per classe (su '+str(len(cLabels))+'):')
	print(errors)
	print(errors)
	print('errore medio:'+str(sum(errors) / float(len(errors))))


def findClassesInClusters(prediction,nameInputs):
	clusters = list(set(prediction))
	for c in clusters:
		indexes = [i for i,x in enumerate(prediction) if x==c]
		inpts = list()
		for i in indexes:
			inpts.append(nameInputs[i])
		mode = stats.mode(inpts)[0][0]
		print('main objects in cluster '+str(c)+' = '+mode)

def distanceIntraCluster(prediction,inputs,nameInputs):
	clusters = list(set(prediction))

	# intra-cluster
	distancesI = dict()
	for c in clusters:
		indexes = [i for i,x in enumerate(prediction) if x==c]
		inpts = list()
		nameInpts = list()
		for i in indexes:
			inpts.append(inputs[i])
			nameInpts.append(nameInputs[i])
		#print(nameInpts)
		sni = list(set(nameInpts))
		sni.sort()
		for si in sni:
			print(si+' -- '+str(nameInpts.count(si)))

		#calcolo distanze euclidee
		D = euclidean_distances(inpts)

		#calcolo distanza intra-cluster
		count = 0
		d = 0
		for i in range(0,len(D)):
			for j in range(i+1,len(D)):
				d += D[i][j]
				count += 1
		try:
			distancesI[c] = d/count
		except:
			distancesI[c] = d
		print(str(c)+' --> '+str(distancesI[c]))

	#extra-cluster
	D = euclidean_distances(inputs)
	d = 0
	count = 0
	for i in range(0,len(D)):
		for j in range(i+1,len(D)):
			d += D[i][j]
			count += 1

	distancesE = d/count
	print(distancesE)
	for c in clusters:
		print(str(c)+' --> '+str(distancesI[c]/distancesE))


def viewClusters(data,nameInputs):
	reduced_data = PCA(n_components=2).fit_transform(data)
	kmeans = KMeans(init='k-means++', n_clusters=classi, n_init=10)
	kmeans.fit(reduced_data)

	# Step size of the mesh. Decrease to increase the quality of the VQ.
	h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

	# Plot the decision boundary. For that, we will assign a color to each
	x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
	y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

	# Obtain labels for each point in mesh. Use last trained model.
	Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])


	# Put the result into a color plot
	Z = Z.reshape(xx.shape)
	plt.figure(1)
	plt.clf()
	plt.imshow(Z, interpolation='nearest',
	           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
	           cmap=plt.cm.Paired,
	           aspect='auto', origin='lower')

	plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
	# Plot the centroids as a white X
	centroids = kmeans.cluster_centers_
	plt.scatter(centroids[:, 0], centroids[:, 1],
	            marker='x', s=169, linewidths=3,
	            color='w', zorder=10)
	plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
	          'Centroids are marked with white cross')
	plt.xlim(x_min, x_max)
	plt.ylim(y_min, y_max)
	plt.xticks(())
	plt.yticks(())
	plt.show()


def KmeansSKLearn():
	print('lettura inputs da '+fIn)
	[inputs,nameInputs] = getAllInputs(fIn,lenEx)
	print('calcolo kmeans')
	km = KMeans(n_clusters = classi,max_iter=100,n_init=50,algorithm='elkan',verbose=1)
	km.fit(inputs)

	#find which cluster each customer is in
	prediction = km.predict(inputs)
	findClassesInClusters(prediction,nameInputs)
	printPredictionsK(prediction,nameInputs)
	distanceIntraCluster(prediction,inputs,nameInputs)

def MiniBatchSKLearn():
	print('lettura inputs da '+fIn)
	[inputs,nameInputs] = getAllInputs(fIn,lenEx)
	print('calcolo minibatch')
	db = MiniBatchKMeans(n_clusters=10, init='k-means++', max_iter=100, batch_size=20, verbose=1)

	#find which cluster each customer is in
	prediction = db.fit_predict(inputs)

	findClassesInClusters(prediction,nameInputs)
	printPredictionsK(prediction,nameInputs)
	distanceIntraCluster(prediction,inputs,nameInputs)

def AffinityPropagationSKLearn():
	print('lettura inputs da '+fIn)
	[inputs,nameInputs] = getAllInputs(fIn,lenEx)
	print('calcolo affinityPropagation')
	km = AffinityPropagation(damping=0.5,verbose=1)
	#km.fit(inputs)

	#find which cluster each customer is in
	prediction = km.fit_predict(inputs)
	findClassesInClusters(prediction,nameInputs)
	#printPredictionsK(prediction,nameInputs)
	distanceIntraCluster(prediction,inputs,nameInputs)


def AgglomerativeSKLearn():
	print('lettura inputs da '+fIn)
	[inputs,nameInputs] = getAllInputs(fIn,lenEx)
	print('calcolo agglomerative clustering')
	ac = AgglomerativeClustering(n_clusters = classi,compute_full_tree=True)
	ac.fit(inputs)

	prediction = ac.fit_predict(inputs)
	findClassesInClusters(prediction,nameInputs)
	printPredictionsK(prediction,nameInputs)
	distanceIntraCluster(prediction,inputs,nameInputs)



if __name__ == '__main__':
	AgglomerativeSKLearn()
