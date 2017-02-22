from matplotlib import pyplot as plt
import numpy as np
import itertools
import lib

print("handling MNIST data")
from mnist import MNIST
mnstdata = MNIST("./python-mnist/data/")
mnstdata.load_training()
mnstdata.load_testing()

dtype  = [("imgs", "int"), ("lbls", "int")]
values = list(enumerate(mnstdata.train_labels))

b = np.array(values, dtype=dtype)
b = np.sort(b, order="lbls")


b_imgs, b_lbls = [*(zip(*b))]

pos = np.searchsorted(b_lbls, range(10))
pos = np.append(pos, len(b_lbls))

train_imgs = [np.divide(x,np.linalg.norm(x)) for x in mnstdata.train_images]
train_imgs = np.array(train_imgs)
imgs = [train_imgs[list(b_imgs[pos[i]:pos[i+1]])] for i in range(10)]

del mnstdata

means_fro = []
means_euc = []
for i in range(10):
    norms = [np.linalg.norm(x.reshape(28,28),ord="fro") for x in imgs[i]]
    means_fro.append((np.mean(norms), np.sqrt(np.var(norms))))

    norms = [np.linalg.norm(x) for x in imgs[i]]
    means_euc.append((np.mean(norms), np.sqrt(np.var(norms))))

means = [np.mean(np.array([*zip(*imgs[i])]),1) for i in range(10)]
means = np.array(means)

fig, ax = plt.subplots(1,1,figsize = (10,10))
ax.set_aspect("equal","datalim")

print(means)
print(means.shape)

print("soft_k_means")
means, resp_matrixresp_matrix  = lib.soft_k_means(k=10, beta=5., N=784,
        data=train_imgs)
        #means=means)
print(means)

print("k_means")
means, resp_dict = lib.k_means(k=10, N=784, data=train_imgs, means=means)
print(means)

del train_imgs

