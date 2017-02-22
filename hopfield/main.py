import numpy as np
import lib
from matplotlib import pyplot as plt

M = 10
N = 1000
beta = 0.1

aa = lib.create_actractors(M, N)
w = lib.create_weights(aa)
s = lib.random_state(N)

x = lib.evolve(aa, w, s, 100, beta, keeptrack=True)
plt.plot([i for i in x])
plt.show()

