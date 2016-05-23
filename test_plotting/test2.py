from bokeh.plotting import figure, output_file, show
from bokeh.charts import Scatter
from bokeh.models import HoverTool, PanTool, BoxSelectTool, BoxZoomTool, ResetTool

import sys
import random
import numpy as np
sys.path.insert(0, '../src/')

from clusterer import Cluster

def htmlcolor(r, g, b):
	def _chkarg(a):
		if isinstance(a, int): # clamp to range 0--255
			if a < 0:
				a = 0
			elif a > 255:
				a = 255
		elif isinstance(a, float): # clamp to range 0.0--1.0 and convert to integer 0--255
			if a < 0.0:
				a = 0
			elif a > 1.0:
				a = 255
			else:
				a = int(round(a*255))
		else:
			raise ValueError('Arguments must be integers or floats.')
		return a
	r = _chkarg(r)
	g = _chkarg(g)
	b = _chkarg(b)
	return '#{:02x}{:02x}{:02x}'.format(r,g,b)

def htmlrandomcolor():
	r = random.randrange(0,256)
	g = random.randrange(0,256)
	b = random.randrange(0,256)
	return htmlcolor(r, g, b)

def getcolors(clusters):
	colors = []
	for i in range(clusters):
		colors.append(htmlrandomcolor())
	return colors

data_csv = '../data/CA_WHS_HOUSEHOLD.csv'
# pick columns as field features
l = ['HOUSEHOLD_CLAIMED_FAMILY_SIZE', 
	'HOUSEHOLD_REFUGEE_STATUS']
(kmeans, store_clusters, reduced_data, clusters) = Cluster(data_csv, l, 2, 7)

colors = getcolors(clusters)
print(colors)

# clustering
output_file("clustering.html", title="Clustering Bokeh example")

#initialize our bokeh plot

x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1

tools = [PanTool(), BoxSelectTool(), BoxZoomTool(), ResetTool(), 
HoverTool(tooltips=[("(x,y)", "($x, $y)")])]

plot = figure(width=500, height=500, title='Clusters', y_range=(y_min, y_max),
	x_range=(x_min, x_max), tools=tools)

#plot centroid / cluster center / group mean for each group

clus_xs = []

clus_ys = []

#we get the  cluster x / y values from the k-means algorithm

for entry in kmeans.cluster_centers_:

   clus_xs.append(entry[0])

   clus_ys.append(entry[1])



#the cluster center is marked by a circle, with a cross in it

plot.circle_cross(x=clus_xs, y=clus_ys, size=40, fill_alpha=0, line_width=2, color=colors)

plot.text(text = [''], x=clus_xs, y=clus_ys, text_font_size=['30pt'])

#text = ['setosa', 'versicolor', 'virginica']

i = 0 #counter

#begin plotting each petal length / width

#We get our x / y values from the original plot data.

#The k-means algorithm tells us which 'color' each plot point is,

#and therefore which group it is a member of.

for k,v in store_clusters.iteritems():
	v = np.asarray(v)
	plot.scatter(v[:, 0], v[:, 1], color=colors[k], size=5)
	#plot.select(dict(type=HoverTool)).tooltips = {"x":"$x", "y":"$y"}

show(plot)
#show(plot)