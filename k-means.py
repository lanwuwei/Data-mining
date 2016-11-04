from __future__ import division
import csv
from random import random
import math
from collections import Counter
import sys
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def readData(filename):
    firstLine=True
    data=[]
    label=[]
    with open(filename) as f:
        contents=csv.reader(f)
        for row in contents:
            if firstLine:
                title.append(row[0])
                title.append(row[3])
                firstLine=False
                continue
            else:
                if filename=='wine.csv':
                    data.append([float(item) for item in row[1:-2]])
                    label.append(row[-2])
                else:
                    data.append([float(item) for item in row[1:-1]])
                    label.append(row[-1])
    return data,label

def euclideanDistance(x1,x2):
    return math.sqrt(sum([(a-b)**2 for a,b in zip(x1,x2)]))

def SilhouetteCoefficient(dataInstance,dataInstanceLabel):
    count=0
    a=0
    b=float("inf")
    s=0
    for i,item in enumerate(data):
        if predictedlabel[i]==dataInstanceLabel:
            a+=euclideanDistance(dataInstance,item)
            count+=1
        else:
            b=min(b,euclideanDistance(dataInstance,item))
    if count:
        a=a/count
    s=(b-a)/max(a,b)
    return s

def dataAnalysis():
    #compute SSE, SSB
    actualCentroids=[]
    countCentroids=[]
    countCentroids_predicted=[]
    SSE_predict=[]
    SSE_actual=[]
    SSB_actual=0
    SSB_predicted=0
    Silhouette=[]
    M=[]
    for i in range(k):
        actualCentroids.append([0 for item in data[i]])
        countCentroids.append(0)
        countCentroids_predicted.append(0)
        SSE_actual.append(0)
        SSE_predict.append(0)
        Silhouette.append(0)
        M.append(0)
    for i,dataInstance in enumerate(data):
        countCentroids_predicted[predictedlabel[i]]+=1
        centroidIndex=int(actuallabel[i])-1
        countCentroids[centroidIndex] += 1
        actualCentroids[centroidIndex] = [sum(x) for x in zip(actualCentroids[centroidIndex], dataInstance)]
        M=[sum(x) for x in zip(M,dataInstance)]
        Silhouette[predictedlabel[i]]+=SilhouetteCoefficient(dataInstance,predictedlabel[i])
    M=[item/len(data) for item in M]
    for i in range(k):
        if countCentroids[i]:
            actualCentroids[i] = [item / countCentroids[i] for item in actualCentroids[i]]
        if countCentroids_predicted[i]:
            Silhouette[i]=Silhouette[i]/countCentroids_predicted[i]
    for i,dataInstance in enumerate(data):
        SSE_predict[predictedlabel[i]]+=(euclideanDistance(dataInstance,centroids[predictedlabel[i]]))**2
        SSE_actual[int(actuallabel[i])-1]+=(euclideanDistance(dataInstance,actualCentroids[int(actuallabel[i])-1]))**2
    for i in range(k):
        SSB_actual+=countCentroids[i]*(euclideanDistance(actualCentroids[i],M))**2
        SSB_predicted+=countCentroids_predicted[i]*(euclideanDistance(centroids[i],M))**2
    print 'SSB actual and SSB predicted'
    print SSB_actual,SSB_predicted
    print 'SSE actual and SSE predicted'
    print SSE_actual,SSE_predict
    print sum(SSE_actual),sum(SSE_predict)
    print 'Cluster Silhouette and Overall Silhouette'
    print Silhouette,sum(Silhouette)

    c1=[]
    c2=[]
    c3=[]
    c4=[]
    for i in range(k):
        c1.append(0)
        c2.append(0)
        c3.append(0)
        c4.append(0)
    for i,item in enumerate(data):
        if actuallabel[i]=='1':
            c1[predictedlabel[i]]+=1
        elif actuallabel[i]=='2':
            c2[predictedlabel[i]] +=1
        elif actuallabel[i]=='3':
            c3[predictedlabel[i]] += 1
        elif actuallabel[i]=='4':
            c4[predictedlabel[i]]+=1
        print actuallabel[i], predictedlabel[i]
    print Counter(actuallabel).keys()
    print Counter(actuallabel).values()
    print c1,c2,c3,c4
    #sys.exit()
    X=[]
    Y=[]
    X1=[]
    Y1=[]
    X2=[]
    Y2=[]
    X3=[]
    Y3=[]
    for i,label in enumerate(predictedlabel):
        if label==0:
            X.append(data[i][0])
            Y.append(data[i][1])
        elif label==1:
            X1.append(data[i][0])
            Y1.append(data[i][1])
        elif label==2:
            X2.append(data[i][0])
            Y2.append(data[i][1])
        elif label==3:
            X3.append(data[i][0])
            Y3.append(data[i][1])
    plt.scatter(X,Y, marker='o', c='b',label='predicted class 1')
    plt.scatter(X1,Y1, marker='x', c='r',label='predicted class 2')
    if inputFile == 'TwoDimHard.csv':
        plt.scatter(X2,Y2, marker='o', c='g',label='predicted class 3')
        plt.scatter(X3,Y3, marker='x', c='y',label='predicted class 4')
        plt.legend(loc='lower right', ncol=2, fontsize=10)
    else:
        plt.legend(loc='upper right',ncol=2,fontsize=12)
    X_true=[]
    Y_true=[]
    X1_true=[]
    Y1_true=[]
    X2_true=[]
    Y2_true=[]
    X3_true=[]
    Y3_true=[]
    for i,label in enumerate(actuallabel):
        if label=='1':
            X_true.append(data[i][0])
            Y_true.append(data[i][1])
        elif label=='2':
            X1_true.append(data[i][0])
            Y1_true.append(data[i][1])
        elif label=='3':
            X2_true.append(data[i][0])
            Y2_true.append(data[i][1])
        elif label=='4':
            X3_true.append(data[i][0])
            Y3_true.append(data[i][1])
    #plt.scatter(X_true,Y_true, marker='D', c='b',label='actual class 1')
    #plt.scatter(X1_true,Y1_true, marker='+', c='r',label='actual class 2')
    if inputFile=='TwoDimHard.csv':
        #plt.scatter(X2_true,Y2_true, marker='D', c='g',label='actual class 3')
        #plt.scatter(X3_true,Y3_true, marker='+', c='y',label='actual class 4')
        plt.legend(loc='lower right',ncol=2,fontsize=10)
    else:
        plt.legend(loc='upper right', ncol=2, fontsize=12)
    plt.show()

def kMeans(k):
    newcentroids=[]
    for i in range(k):
        newcentroids.append([item*random() for item in data[i+100]])
    if inputFile=='TwoDimEasy.csv':
        if newcentroids[0][0]<newcentroids[1][0]:
            newcentroids[0],newcentroids[1]=newcentroids[1],newcentroids[0]
    elif inputFile=='TwoDimHard.csv':
        x_value=[newcentroids[0][0],newcentroids[1][0],newcentroids[2][0],newcentroids[3][0]]
        y_value=[newcentroids[0][1],newcentroids[1][1],newcentroids[2][1],newcentroids[3][1]]
        y_value.sort()
        index=sorted(range(len(y_value)), key=lambda k: y_value[k])
        temp=[x_value[index[0]],x_value[index[1]],x_value[index[2]],x_value[index[3]]]
        x_value=temp
        if x_value[-1]<x_value[-2]:
            newcentroids[0]=[x_value[-1],y_value[-1]]
            newcentroids[1]=[x_value[-2],y_value[-2]]
        else:
            newcentroids[0] = [x_value[-2], y_value[-2]]
            newcentroids[1] = [x_value[-1], y_value[-1]]
        if x_value[-3]<x_value[0]:
            newcentroids[2]=[x_value[-3],y_value[-3]]
            newcentroids[3]=[x_value[0],y_value[0]]
        else:
            newcentroids[3]=[x_value[-3],y_value[-3]]
            newcentroids[2]=[x_value[0],y_value[0]]
    print newcentroids
    newcentroids=[[0,1],[1,1],[0,0],[1,0]]
    while 1:
        predictedlabel = []
        lastCentroids = newcentroids[:]
        countcentroids=[0 for item in newcentroids]
        #print countcentroids
        for i,dataInstance in enumerate(data):
            distance=[]
            for j in range(k):
                distance.append(euclideanDistance(dataInstance,lastCentroids[j]))
            centroidIndex=distance.index(min(distance))
            predictedlabel.append(centroidIndex)
            countcentroids[centroidIndex]+=1
            newcentroids[centroidIndex]=[sum(x) for x in zip(newcentroids[centroidIndex],dataInstance)]
        #print countcentroids
        #print len(predictedlabel)
        for i in range(k):
            if countcentroids[i]:
                newcentroids[i]=[item/countcentroids[i] for item in newcentroids[i]]
        print lastCentroids
        print  newcentroids
        print '------------------------'
        if newcentroids[0]==lastCentroids[0]:
            break
    return newcentroids,predictedlabel

if __name__=='__main__':
    #inputFile='TwoDimEasy.csv'
    inputFile='TwoDimHard.csv'
    #inputFile='wine.csv'
    title=[]
    data,actuallabel=readData(inputFile)
    k=4
    centroids,predictedlabel=kMeans(k)
    '''
    with open('output.csv','a+') as f:
        pointer=csv.writer(f)
        pointer.writerow(title)
        for i,label in enumerate(predictedlabel):
            pointer.writerow([i+1,label])
    '''
    print centroids
    #kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
    #print kmeans.cluster_centers_
    #predictedlabel=kmeans.labels_
    dataAnalysis()