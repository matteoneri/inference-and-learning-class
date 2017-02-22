import numpy as np
import collections
import itertools
from functools import reduce
from matplotlib import pyplot as plt

def generate_cluster_data(N, M, sigma, center = None):
    """
    Generate cluster data in N dimension. The size of the cluster is a sphere
    of radius ~ sigma and the number of points is M.
    """
    if not center:
        center = np.random.randn(N)
    return [np.random.randn(N)*sigma+center for i in range(M)]


def d(x,y):
    return np.linalg.norm(x-y)

def responsability(N, means, point):
    z = list(zip([d(x,point) for x in means],range(len(means))))
    k = min(z)[1]
    r = np.zeros(len(means))
    r[k] = 1.
    return k

def k_means(k, N, data, ax=None, t=2., cluster_dims=None, means=None):
    # inizialization
    responsability_dict = None
    old_means = [np.zeros(N) for i in range(k)]
    if means==None:
        means = [np.random.randn(N) for i in range(k)]
    if ax and N<=2:
        plot_k_means(data,means,ax,t,cluster_dims)

    while np.any([np.any(old_means[i] != means[i]) for i in range(k)]):
        responsability_dict = collections.defaultdict(list)
        for x in data:
            responsability_dict[responsability(N,means,x)].append(x)
            
        # assignment step
        old_means, means = means, []
        for i in range(k):
            if len(responsability_dict[i]):
                means.append(np.divide(reduce(np.add, responsability_dict[i], 0),\
                        len(responsability_dict[i])))
            else:
                means.append(old_means[i])
        if ax and N<=2:
            plot_k_means(data,means,ax,t,cluster_dims,responsability_dict)
    
    return means, responsability_dict

def soft_responsability(k, beta, means, x, norm):
    a = [np.exp(-beta*norm(k-x)) for k in means]
    Z = np.sum(a)
    r = np.array([np.divide(p,Z) for p in a])
    return r

def soft_k_means(k, beta, N, data, ax=None, t=2., cluster_dims=None,
        means=None, norm=np.linalg.norm):
    # INITIALIZATION
    responsabilities_matrix=None
    old_means = np.array([np.zeros(N) for i in range(k)])
    if means==None:
        means = np.array([np.random.randn(N) for i in range(k)])
        #print(means.shape)

    stop = np.array([np.mean(i) for i in means-old_means])
    
    if ax and N<=2:
        plot_k_means(data,means,ax,t,cluster_dims, beta=beta)
    if ax:
        plot_k_means(data[:,:2],means[:2],ax,t,cluster_dims, beta=beta)
    while np.any(stop>1e-4):
        # responsability matrix (k,M) with M the # of points in data
        responsabilities_matrix = [soft_responsability(k, beta, means, x, norm)\
                for x in data]
        responsabilities_matrix = np.array([*zip(*responsabilities_matrix)])
        # ASSIGNMENT STEP
        old_means = means
        # (k,N)
        R_k = [np.sum(r) for r in responsabilities_matrix]
        means = np.dot(responsabilities_matrix,data)
        means = np.array([np.divide(means[i],R_k[i]) for i in range(k)])
        
        stop = np.array([np.mean(i) for i in means-old_means])
        #print(stop)

        if ax and N<=2:
            plot_k_means(data,means,ax,t,cluster_dims, beta=beta)

        
    return means, responsabilities_matrix

    

def plot_k_means(data,means,ax,t,cluster_dims, k_dict=None, beta=None):
    plt.sca(ax)
    colors  = ["b","g","k","y","c"]
    markers = ["o","v","^","D","<",">"]
    if cluster_dims != None:
        pos = np.concatenate([[0],np.cumsum(cluster_dims)])
        if k_dict == None:
            for j in range(len(cluster_dims)):
                [ax.plot(*i, colors[j%5]+"o") for i in data[pos[j]:pos[j+1]]]
        else:
            cluster_dict = {}
            for y in data:
                z = list(zip([d(x,y) for x in means],range(len(means))))
                k = min(z)[1]
                cluster_dict[np.linalg.norm(y)] = k
            for j in range(len(cluster_dims)):
                [ax.plot(*i,
                    colors[j%5]+markers[cluster_dict[np.linalg.norm(i)]%6])\
                            for i in data[pos[j]:pos[j+1]]]
    else:
        [ax.plot(*i, "bo") for i in data]
    [ax.plot(*means[i], "r"+markers[i%6]) for i in range(len(means))]
    if beta:
        circles = [plt.Circle(i, np.sqrt(1/beta),edgecolor="r",fill=False) for i in means]
        [ax.add_artist(c) for c in circles]
    plt.pause(t)
    plt.cla()
    
    
    
def plot_k_means_notebook(data, means, ax, cluster_dims, k_dict=None, beta=None):
    plt.sca(ax)
    colors  = ["b","g","k","y","c"]
    markers = ["o","v","^","D","<",">"]
    if cluster_dims != None:
        pos = np.concatenate([[0],np.cumsum(cluster_dims)])
        if k_dict == None:
            for j in range(len(cluster_dims)):
                [ax.plot(*i, colors[j%5]+"o") for i in data[pos[j]:pos[j+1]]]
        else:
            cluster_dict = {}
            for y in data:
                z = list(zip([d(x,y) for x in means],range(len(means))))
                k = min(z)[1]
                cluster_dict[np.linalg.norm(y)] = k
            for j in range(len(cluster_dims)):
                [ax.plot(*i,
                    colors[j%5]+markers[cluster_dict[np.linalg.norm(i)]%6])\
                            for i in data[pos[j]:pos[j+1]]]
    else:
        [ax.plot(*i, "bo") for i in data]
    [ax.plot(*means[i], "r"+markers[i%6]) for i in range(len(means))]
    if beta:
        circles = [plt.Circle(i, np.sqrt(1/beta),edgecolor="r",fill=False) for i in means]
        [ax.add_artist(c) for c in circles]
