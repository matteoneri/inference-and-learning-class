import numpy as np
import scipy as sp

def create_actractors(M, N):
    return np.array([np.random.choice([-1,1], size=N) for i in range(M)])

def create_weights(actractors):
    M = len(actractors)
    N = len(actractors[0])

    temp = np.dot(actractors.T,actractors)
    temp -= np.eye(N,dtype=np.int)*M

    return np.divide(temp,N)

def random_state(N):
    return np.random.choice([-1,1], size=N)

def perturb_state(state, n_errors):
    N = len(state)
    for i in range(n_errors):
        state = flip_spin(np.random.randint(N), state)
    return state

def flip_spin(i,state):
    state[i] = -np.sign(state[i])
    return state

def codistance(actractors,state):
    N = len(actractors[0])
    return np.divide(np.dot(actractors,state),N)

def evolveMCMC(actractors, weights, state, T, beta, keeptrack=False):
    M = len(actractors)
    N = len(state)

    if keeptrack:
        history = []

    for t in range(T):
        i = np.random.randint(N)
        prop = np.copy(state)
        flip_spin(i, prop)
        dE = -2*prop[i]*np.dot(weights[i], prop)

        if dE<0 or np.random.rand() < np.exp(-beta*dE):
            state = prop

        if keeptrack:
            history.append(codistance(actractors, state))
        
    if keeptrack:
        return history
    else:
        return(codistance(actractors, state))
    


