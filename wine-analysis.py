#%%
import csv
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import math
#%%
def readData(filename):
    data=[]
    label=[]
    with open(filename) as f:
        contents=csv.reader(f, delimiter=',')
        next(contents, None)
        for row in contents:
            data.append([float(item) for item in row[1:-1]])
            label.append(row[-1])
    return data,label

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
    ssb = 0
    dataMean = []
    for i in range(len(data[0])):
        dataMean.append(sum([x[i] for x in data])/len(data))
    for c in range(k):
        clusterPoints = [data[i] for i in range(len(data)) if labels[i] == c]
        ssb += len(clusterPoints)*(euclideanDistance(dataMean, centroids[c])**2)
    return ssb

def euclideanDistance(x1,x2):
    return math.sqrt(sum([(a-b)**2 for a,b in zip(x1,x2)]))
#%%
data, label = readData('datasets/wine_cleaned.csv')
#The data has already been pre-processed in weka
#%%
k = 2
kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, tol=0.0001,
                precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=4, algorithm='auto')
kmeans.fit(data)

#%%
SSE = computeSSE(data, kmeans.labels_, kmeans.cluster_centers_)
SSB = computeSSB(data, kmeans.labels_, kmeans.cluster_centers_)