import numpy as np
import pandas as pd
from numpy import genfromtxt
from sklearn.cluster import kmeans_plusplus
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.cluster import spectral_clustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.feature_selection import SelectKBest
from kmeans import Kmeans
from sklearn.feature_selection import chi2

from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform
import numpy as np
from math import isnan

standedColorList = ['aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'black', 'blanchedalmond', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'gray', 'green', 'greenyellow', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgreen', 'lightgray', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive', 'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'red', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue', 'slategray', 'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'white', 'whitesmoke', 'yellow', 'yellowgreen', ]



 
def hopkins(X):
    X = pd.DataFrame(X)
    d = X.shape[1]
    #d = len(vars) # columns
    n = len(X) # rows
    m = int(0.1 * n) # heuristic from article [1]
    nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)
 
    rand_X = sample(range(0, n, 1), m)
 
    ujd = []
    wjd = []
    for j in range(0, m):
        u_dist, _ = nbrs.kneighbors(uniform(np.amin(X,axis=0),np.amax(X,axis=0),d).reshape(1, -1), 2, return_distance=True)
        ujd.append(u_dist[0][1])
        w_dist, _ = nbrs.kneighbors(X.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)
        wjd.append(w_dist[0][1])
 
    H = sum(ujd) / (sum(ujd) + sum(wjd))
    if isnan(H):
        print(ujd, wjd)
        H = 0
 
    return H


def generatePCAMap(labels,embeddings,embeddingFileName):
    colorList = []
    for item in labels:
        colorList.append(standedColorList[5*int(item)%len(standedColorList)])
        # if(int(item) == 0):
        #     colorList.append('m')
        # else:
        #     colorList.append('k')
            
    pca = PCA(n_components=5)
    reduced = pca.fit_transform(embeddings)

    t = reduced.transpose()
    plt.scatter(t[0], t[1],s=1,c=colorList,linewidths=0)

    plt.title(embeddingFileName)

    plt.savefig(embeddingFileName + '.jpg', format='jpg', dpi=200)
    print('saved '+ embeddingFileName + '.jpg')

def sigmoid(z):
    return 10 / (1.0 + np.exp(-z))

def loaddata(filename):   
    X = pd.read_csv('CCGENERAL.csv')
    idLabels = X['CUST_ID'].values
    X = X.drop(labels='CUST_ID', axis=1).values
    return X, idLabels

def datapreprocess(X):
    X[np.isnan(X)]=0
    print(X.shape)
    
    X = sigmoid(X) 
    
    X = StandardScaler().fit_transform(X)
    return X
    
def main():
    # X = pd.read_csv('CCGENERAL.csv')
    # idLabels = X['CUST_ID'].values
    # X = X.drop(labels='CUST_ID', axis=1).values
    # X[np.isnan(X)]=0
    # print(X.shape)
    
    # X = sigmoid(X) 
    
    # X = StandardScaler().fit_transform(X)
    
    
    X, idLabels = loaddata('CCGENERAL.csv')
    X = datapreprocess(X)
    print(hopkins(X))
    
    y_pred = KMeans(n_clusters=4, random_state=42).fit_predict(X)
    generatePCAMap(y_pred,X,"KMeans")    
    
    clustering = AffinityPropagation(random_state=4).fit(X)
    generatePCAMap(clustering.labels_, X, "AffinityPropagation")
    
    clustering = DBSCAN(eps=0.3, min_samples=4).fit(X)
    generatePCAMap(clustering.labels_, X, "DBSCAN")
    
    clustering = AgglomerativeClustering(n_clusters=4, linkage="ward").fit(X)
    generatePCAMap(clustering.labels_, X, "Agglomerative")
    
    clf = Kmeans(k=4)
    clustering = clf.predict(X)
    generatePCAMap(clustering, X, "kmeans2")
    
    bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=100)
    clustering = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    clustering.fit(X)
    generatePCAMap(clustering.labels_, X, "MeanShift")
    
    # 因为没有标签，所以一般通过评估类的分离情况来决定聚类质量。类内越紧密，类间距离越小则质量越高。我用到过的有sklearn中的Silhouette Coefficient和Calinski-Harabaz Index，sklearn里面解释的很清楚，直接把数据和聚类结果作为输入就可以了。
    Silhouette_Coefficient = metrics.silhouette_score(X, clustering.labels_, metric='euclidean')
    CalinskiHarabasz_index = metrics.calinski_harabasz_score(X, clustering.labels_)
    print(Silhouette_Coefficient,CalinskiHarabasz_index)


if __name__ == "__main__":
    main()
 