import numpy as np
import pandas as pd
from numpy import genfromtxt
# from sklearn.cluster import kmeans_plusplus
# from sklearn.cluster import KMeans
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

# standedColorList = ['aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'black', 'blanchedalmond', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'gray', 'green', 'greenyellow', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgreen', 'lightgray', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive', 'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'red', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue', 'slategray', 'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'white', 'whitesmoke', 'yellow', 'yellowgreen', ]

standedColorList = ['red','blue','orangered','green','darkmagenta', 'purple','pink']

#  应用霍普金斯统计量（Hopkins Statistic） 可以判断数据在空间上的随机性，从而判断数据是否可以聚类
#  如果样本接近随机分布，H的值接近于0.5；
# 如果聚类趋势明显，则随机生成的样本点距离应该远大于实际样本点的距离H的值接近于1.
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
        colorList.append(standedColorList[int(item)%len(standedColorList)])
        # if(int(item) == 0):
        #     colorList.append('m')
        # else:
        #     colorList.append('k')
            
    pca = PCA(n_components=5)
    reduced = pca.fit_transform(embeddings)

    t = reduced.transpose()
    plt.scatter(t[0], t[1],s=1,c=colorList,linewidths=0)

    plt.title(embeddingFileName)

    plt.savefig(embeddingFileName + '.jpg', format='jpg', dpi=300)
    print('saved '+ embeddingFileName + '.jpg')

def sigmoid(z):
    return 1/ (1.0 + np.exp(-z))

def loaddata(filename):   
    X = pd.read_csv('CCGENERAL.csv')
    idLabels = X['CUST_ID'].values
    X = X.drop(labels='CUST_ID', axis=1).values
    return X, idLabels

def datapreprocess(X):
    # X[np.isnan(X)]=0    
    # 使用均值替换nan值
    col_mean = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1])
    
    # print(X.shape)    
    X = sigmoid(X)     
    X = StandardScaler().fit_transform(X)
    return X
    
    
def evaluate(X, clustering, modelName):
    # 轮廓系数（Silhouette Coefficient）结合了聚类的凝聚度（Cohesion）和分离度（Separation），用于评估聚类的效果。该值处于-1~1之间，值越大，表示聚类效果越好。
    # Silhouette Coefficient or silhouette score is a metric used to calculate the goodness of a clustering technique.
    # Its value ranges from -1 to 1.
    # 1: Means clusters are well apart from each other and clearly distinguished.
    # 0: Means clusters are indifferent, or we can say that the distance between clusters is not significant.
    # -1: Means clusters are assigned in the wrong way.
    # Calinski-Harabasz指标通过计算类中各点与类中心的距离平方和来度量类内的紧密度，通过计算各类中心点与数据集中心点距离平方和来度量数据集的分离度，CH指标由分离度与紧密度的比值得到。从而，CH越大代表着类自身越紧密，类与类之间越分散，即更优的聚类结果。
    Silhouette_Coefficient = metrics.silhouette_score(X, clustering, metric='euclidean')
    CalinskiHarabasz_index = metrics.calinski_harabasz_score(X, clustering)
    # print(Silhouette_Coefficient,CalinskiHarabasz_index)
    print("{} Silhouette_Coefficient is {},CalinskiHarabasz_index is {}".format(modelName,Silhouette_Coefficient,CalinskiHarabasz_index))
    with open("log.txt",'a') as f:
        f.writelines("| {} | {:.3f} |{:.3f} |\n".format(modelName,Silhouette_Coefficient,CalinskiHarabasz_index))
    
    
    
def main():
    X, idLabels = loaddata('CCGENERAL.csv')
    X = datapreprocess(X)
    print("Hopkins Statistic is: {}".format(hopkins(X)))
    
    # with open("log.txt",'a') as f:
    #     f.writelines("\n\n### Table Clustering Evaluation\n")
    #     f.writelines("| ModelName | Silhouette_Coefficient | CalinskiHarabasz_index |\n")
        
    # clustering = AffinityPropagation(random_state=4).fit(X)
    # generatePCAMap(clustering.labels_, X, "AffinityPropagation")
    # evaluate(X, clustering.labels_, "AffinityPropagation")
    
    # clustering = DBSCAN(eps=0.3, min_samples=4).fit(X)
    # generatePCAMap(clustering.labels_, X, "DBSCAN")
    # evaluate(X, clustering.labels_, "DBSCAN")
    
    # clustering = AgglomerativeClustering(n_clusters=4, linkage="ward").fit(X)
    # generatePCAMap(clustering.labels_, X, "Agglomerative")
    # evaluate(X, clustering.labels_, "Agglomerative")
    
    # bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=100)
    # clustering = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    # clustering.fit(X)
    # generatePCAMap(clustering.labels_, X, "MeanShift")
    # evaluate(X, clustering.labels_, "MeanShift")
    
    clf = Kmeans(k=3)
    clustering = clf.predict(X)
    generatePCAMap(clustering, X, "kmeans")
    evaluate(X, clustering, "kmeans")

if __name__ == "__main__":
    main()
 