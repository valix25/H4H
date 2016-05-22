from time import time
import numpy as np
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

np.random.seed(42)

# load Household csv file
data = pd.read_csv('../data/CA_WHS_HOUSEHOLD.csv')

# remove 'delete' rows
print(len(data.index))
data = data[data['HOUSEHOLD_INACTIVE_REASON'] != 'DELETED']
print(len(data.index))

data = ReduceColumns(data)

data.to_csv('../data/CA_WHS_HOUSEHOLD_SIMPLIFIED.csv')

# pick columns as field features
l = ['HOUSEHOLD_CLAIMED_FAMILY_SIZE', 'LOCATION_ID', 'ADMIN_AREA_ID']

vals = []
for i in l:
	vals.append(data[i].count())
print(vals)

data = data[l].values

print(data.shape)

# normalize the features
data_normed = data*1.0 / data.max(axis=0)
#print(data_normed[1:100][:])

n_samples, n_features = data_normed.shape

def bench_k_means(estimator, name, data):
	t0 = time()
	estimator.fit(data)
	sample_size = 300
	print('%9s %.2fs %.3f' % 
	(name, (time() - t0), metrics.silhouette_score(data, estimator.labels_,
							  metric='euclidean',
							  sample_size=sample_size)))

n_clusters = range(2,14)
for i in n_clusters:
	bench_k_means(KMeans(init='k-means++', n_clusters=i, n_init=10),
	              name="k-means++", data=data_normed)

bench_k_means(KMeans(init='random', n_clusters=20, n_init=10),
              name="random", data=data_normed)

pca = PCA(n_components=13).fit(data_normed)
bench_k_means(KMeans(init=pca.components_, n_clusters=13, n_init=1),
              name="PCA-based",
              data=data_normed)

####################################################################
# Visualize the results on PCA-reduced data

reduced_data = PCA(n_components=2).fit_transform(data_normed)
#kmeans = KMeans(init='k-means++', n_clusters=13, n_init=10)

kmeans = KMeans(init='random', n_clusters=20, n_init=10)
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