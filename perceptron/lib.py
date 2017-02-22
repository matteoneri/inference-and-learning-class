import numpy as np
from functools import reduce

class teacher:
    def __init__(self, N):
        temp = np.random.uniform(-1,1,N)
        temp /= np.sqrt(np.dot(temp,temp))
        self._w = np.r_[np.random.rand(),temp]
        self._N = N

    def generate_sample(self, M):
        l = lambda x: np.sign(np.dot(np.r_[1,x],self._w))
        sample = np.array([np.random.randn(self._N) for i in range(M)])
        sample_labels = np.array([l(x) for x in sample])
        return sample, sample_labels


class student:
    def __init__(self,N):
        self._w = np.random.uniform(-1,1,N+1)
        self._w /= np.sqrt(np.dot(self._w,self._w))
        
#    def _h(self, x, y, p=1):
#        h = (np.dot(x,self._w)*y) 
#        return h
#
#    def E(X,Y):
#        assert len(X)==len(Y)
#        M = len(X)
#        def l(u,v):
#            h = h(*v)
#            return x-np.sign(h)*np.power(np.abs(h))
#        return reduce(l, zip(X,Y),0)/M

    def train(self, train_set, train_set_labels, eta=0.1):
        for x,y in zip(train_set,train_set_labels):
            #x /= np.sqrt(np.dot(x,x))
            x = np.r_[1,x]
            sigma = np.sign(np.dot(x,self._w))
            if y == sigma:
                self._w +=  eta*y*x
                self._w /= np.sqrt(np.dot(self._w,self._w))

    def validate(self, val_set,):
        """
        input:  X a vector or a matrix with features in the rows (M,N)
        output: (M,)
        """
        val_set = np.array([np.r_[1,x] for x in val_set])
        return len(val_set.shape)==1 and\
                np.array([np.dot(val_set,self._w)]) or np.dot(val_set,self._w)

    
    def error(self, X, Y):
        return 1-sum(np.sign(self.validate(X)) == Y)/len(Y)

