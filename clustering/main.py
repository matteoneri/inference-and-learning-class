import lib
import numpy as np
from matplotlib import pyplot as plt

k     = 6
sigma = 0.1
beta = 10.0
N = 2

np.random.seed(1224567)

cluster_dimensions = np.random.randint(50,500, k)
cluster_dimensions = np.array([50,200,175,300,20,5])

# Creations of k clusters:
data = np.concatenate([lib.generate_cluster_data(N,cluster_dimensions[i], sigma)\
    for i in range(k)])

np.random.seed()

# k-means
plt.ion()

fig, ax = plt.subplots(1,1,figsize = (10,10))
ax.set_aspect("equal","datalim")

means, resp_dict = lib.k_means(k, N, data, ax, 1.0, cluster_dimensions)
#means, resp = lib.soft_k_means(k, beta, N, data, ax, 0.1, cluster_dimensions)

print("DONE")
lib.plot_k_means(data,means,ax,3.0,cluster_dimensions,beta=beta )
lib.plot_k_means(data,means,ax,10.0,cluster_dimensions,beta=beta)
