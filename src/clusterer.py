from time import time
import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

def ReduceColumns(data):
	# remove one-single valued columns from the csv file
	drop_ids = []
	index = 0
	for column in data:
		if(1 == len(data[column].unique())):
			print(column)
			drop_ids.append(index)
		index += 1

	print(len(data.columns))
	data = data.drop(data.columns[drop_ids], axis=1)
	print(len(data.columns))
	return data

def StringToNumLabels(data, l):
	for col in l:
		v = data[col].unique()
		if(isinstance(v[0], basestring)):
			dicty = {}
			for i in range(len(v)):
				dicty[v[i]] = i
			#print(dicty)
			data[col] = data[col].replace(dicty)
			data[col] = data[col].convert_objects(convert_numeric=True)
	return data

def IdColsToNumLabels(data, l):
	for col in l:
		if 'ID' in col:
			v = data[col].unique()
			v.sort()
			dicty = {}
			for i in range(len(v)):
				dicty[v[i]] = i
			#print(dicty)
			data[col] = data[col].replace(dicty)
			data[col] = data[col].convert_objects(convert_numeric=True)
	return data

def BenchKMeans(estimator, name, data):
	t0 = time()
	estimator.fit(data)
	sample_size = 300
	metric_score = metrics.silhouette_score(data, estimator.labels_,
							  metric='euclidean',
							  sample_size=sample_size)
	print('%9s %.2fs %.3f' % 
	(name, (time() - t0), metric_score))
	return metric_score
	
def Cluster(data_csv, l, k_min, k_max):

	np.random.seed(42)

	# load Household csv file
	data = pd.read_csv(data_csv)

	# remove 'delete' rows
	print(len(data.index))
	data = data[data['HOUSEHOLD_INACTIVE_REASON'] != 'DELETED']
	print(len(data.index))

	data = ReduceColumns(data)

	data.to_csv('../data/CA_WHS_HOUSEHOLD_SIMPLIFIED.csv')

	# select only the l columns
	data = data[l]

	# here remove if null or nan
	data = data.dropna()

	# here string to num labels
	data = StringToNumLabels(data, l)

	# data.to_csv('../data/CA_test.csv')
	# here ids to num labels
	data = IdColsToNumLabels(data, l)
	#sys.exit()

	data = data[l].values

	print(data.shape)

	# normalize the features
	data_normed = data*1.0 / data.max(axis=0)
	#print(data_normed[1:100][:])

	n_samples, n_features = data_normed.shape
	n_clusters = range(k_min, k_max)
	
	metric_scores = []
	for i in n_clusters:
		m_score = BenchKMeans(KMeans(init='k-means++', n_clusters=i, n_init=10),
					  name="k-means++", data=data_normed)
		metric_scores.append((m_score, i))
	metric_scores = sorted(metric_scores)
	max_metric_score_km = metric_scores[0]

	metric_scores = []
	for i in n_clusters:
		m_score = BenchKMeans(KMeans(init='random', n_clusters=i, n_init=10),
					  name="random", data=data_normed)
		metric_scores.append((m_score, i))
	metric_scores = sorted(metric_scores)
	max_metric_score_rn = metric_scores[0]
	
	print(max_metric_score_km)
	print(max_metric_score_rn)

	pca = PCA(n_components=13).fit(data_normed)
	BenchKMeans(KMeans(init=pca.components_, n_clusters=13, n_init=1),
				  name="PCA-based",
				  data=data_normed)

	####################################################################
	# Visualize the results on PCA-reduced data

	reduced_data = PCA(n_components=2).fit_transform(data_normed)
	#kmeans = KMeans(init='k-means++', n_clusters=13, n_init=10)

	kmeans = KMeans(init='random',n_clusters=max_metric_score_rn[1], n_init=10)
	if(max_metric_score_km[0] < max_metric_score_rn[0]):
		kmeans = KMeans(init='k-means++', n_clusters=max_metric_score_km[1], 
			n_init=10)
	kmeans.fit(reduced_data)

	# Step size of the mesh. Decrease to increase the quality of the VQ.
	h = .02     # point in the mesh [x_min, m_max]x[y_min, y_max].

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
	plt.title('K-means clustering on the household columns (PCA-reduced data)\n'
			  'Centroids are marked with white cross')
	plt.xlim(x_min, x_max)
	plt.ylim(y_min, y_max)
	plt.xticks(())
	plt.yticks(())
	plt.show()

data_csv = '../data/CA_WHS_HOUSEHOLD.csv'
# pick columns as field features
l = ['HOUSEHOLD_CLAIMED_FAMILY_SIZE', 'LOCATION_ID', 'ADMIN_AREA_ID', 
	'HOUSEHOLD_REFUGEE_STATUS']
Cluster(data_csv, l, 2, 7)