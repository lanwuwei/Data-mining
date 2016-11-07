#%%
import csv
from sklearn.cluster import KMeans
from sklearn.metrics import *
import matplotlib.pyplot as plt
import math
import numpy as np
np.set_printoptions(precision=2)
#%%
def readData(filename):
    data=[]
    label=[]
    with open(filename) as f:
        contents=csv.reader(f, delimiter=',')
        next(contents, None)
        for row in contents:
            data.append([float(item) for item in row[0:-1]])
            label.append(row[-1])
    return np.array(data),np.array(label)
#%%
def computeSSE(data, labels, centroids):
    k = len(centroids)
    sse = 0
    for c in range(k):
        clusterPoints = [data[i] for i in range(len(data)) if labels[i] == c]
        for point in clusterPoints:
            sse += euclideanDistance(point, centroids[c])**2
    return sse
    
def computeSSB(data, labels, centroids):
    k = len(centroids)
    dataMean = np.mean(data, axis = 0)
#    for i in range(len(data[0])):
#        dataMean.append(sum([x[i] for x in data])/len(data))
    ssb = 0
    for c in range(k):
        clusterPoints = [data[i] for i in range(len(data)) if labels[i] == c]
        ssb += len(clusterPoints)*(euclideanDistance(dataMean, centroids[c])**2)
    return ssb

def euclideanDistance(x1,x2):
    return math.sqrt(sum([(a-b)**2 for a,b in zip(x1,x2)]))
#%%
def convertLabelsToNumeric(labels):
    classes = getClasses(labels)
    newLabels = []
    for l in labels :
        for i in range(len(classes)):
            if classes[i] == l:
                newLabels = newLabels + [i]
    return np.array(newLabels)
    
def getClasses(labels):
    classes = [labels[0]]
    for l in labels:
        exists = False
        for c in classes:
            if(c == l):
                exists = True
        if(exists == False):
            classes = classes + [l]
    return classes

def generateStatistics(data, labels):
    classes = getClasses(labels)
    k = len(classes)
    dataMean = np.mean(data, axis = 0)
    ssb = 0
    print('Cluster#\t#NumOfPoints\tSSE\t\tCentroid')
    for i in range(k):
        clusterPoints = np.array([data[j] for j in range(len(data)) if labels[j] == i])
        centroid = np.mean(clusterPoints, axis = 0)
        ssb = ssb +len(clusterPoints)*(euclideanDistance(dataMean, centroid)**2)
        sse = 0
        for point in clusterPoints:
            sse = sse + euclideanDistance(point, centroid)**2
        print(repr(i) + '\t\t' + repr(len(clusterPoints)) + '\t' + repr('%.2f' % sse ) + '\t' + repr(centroid))
    SLHW = silhouette_score(data, labels, metric='euclidean')
    print('Silhoutte Width:' + repr(SLHW))
    print('SSB:' + repr(ssb))
#%%
data, label = readData('datasets/wine_cleaned.csv')
label = convertLabelsToNumeric(label)
#The data has already been pre-processed in weka
#%%
k = 2
kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, tol=0.0001,
                precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=4, algorithm='auto')
kmeans.fit(data)

#%%
#SSE = computeSSE(data, kmeans.labels_, kmeans.cluster_centers_)
SSB = computeSSB(data, kmeans.labels_, kmeans.cluster_centers_)
#SLHW = silhouette_score(data, kmeans.labels_, metric='euclidean')
generateStatistics(data, kmeans.labels_)
generateStatistics(data, label)

#%%
mapping = {0:1, 1:0}
label = [mapping[l] for l in label]
conMat = confusion_matrix(label, kmeans.labels_)

#%%
data, label = readData('datasets/wine_cleaned_qualityAsClass.csv')
label = convertLabelsToNumeric(label)
k = 6
kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, tol=0.0001,
                precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=4, algorithm='auto')
kmeans.fit(data)
print('Statistics for Kmeans clusters')
generateStatistics(data, kmeans.labels_)
print('Statistics for original cluster')
generateStatistics(data, label)
