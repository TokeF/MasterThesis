import pandas as pd
import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt
import utilities.PC_anal
import utilities.data_visualize as dvis

#apply linear regression to points in window. Return slope of the fit and r value
def linear_deriv(timestamp: np.ndarray, dbdt : np.ndarray, wsize : int):
    shift = int(np.floor(wsize / 2))
    slope = np.zeros((dbdt.shape[0], dbdt.shape[1]))
    rvalue = np.zeros((dbdt.shape[0], dbdt.shape[1]))
    for j in range(dbdt.shape[1]): #iterate over gates
        for i in range(shift, dbdt.shape[0] - shift): #iterate over gate values, starting from shift to end-shift
            x = timestamp[i - shift : i + shift]
            y = dbdt[i - shift : i + shift, j]
            s = sp.linregress(x, y) #perform linear regression on window
            slope[i,j] = s.slope
            rvalue[i, j] = s.rvalue
            # line[1,i] = s.rvalue
    return slope, rvalue

#None of the methods account for gaps
#Compute difference quotient. OBS: currently not computing last value
def derivative(timestamp: np.ndarray, dbdt : np.ndarray):
    der = np.zeros((dbdt.shape[0], dbdt.shape[1]))
    for i in range(dbdt.shape[0] - 1):
        der[i,:] = (dbdt[i + 1,:] - dbdt[i, :])/(timestamp[i + 1] - timestamp[i])
    return der

# Compute rolling window average. Does not account for start and end values of array.
def mvg_var(gate : pd.DataFrame, wsize : int) :
    shift = -int(np.floor(wsize/2))
    window = gate.shift(shift).rolling(wsize).var()
    # gate = pd.concat([window, gate.DBDT_Ch2GT8], axis =1)
    return window.values

#compute variance with only a forward looking window. Do it in both directions
def mvg_var_leftright(gate : pd.DataFrame, wsize : int) :
    windowLR = gate[::-1].rolling(wsize).var() #reverse the values and later the window to do LR
    windowRL = gate.rolling(wsize).var() #do not reverse since this already is RL value (just going backwards)
    return windowLR[::-1].values, windowRL.values

# Compute rolling window average. Does not account for start and end values of array.
def mvg_avg(gate : pd.DataFrame, wsize : int) :
    shift = - int(np.floor(wsize/2))
    window = gate.shift(shift).rolling(wsize).mean()
    # gate = pd.concat([window, gate.DBDT_Ch2GT8], axis =1)
    return window.values

def row_ratio(timestamp : np.ndarray, dbdt : np.ndarray) -> np.ndarray:
    #compute ratio between neighbour gates
    #allocate space for ratio vbectors/matrix
    ratio = np.zeros((dbdt.shape[0], dbdt.shape[1] - 1))
    for i in range(dbdt.shape[1] - 1):
        ratio[:, i] = dbdt[:,i+1] / dbdt[:,i]
    return ratio

#Compute difference between rows, UNUSED!
def row_diff(dbdt : np.ndarray):
    #for each column, compute the distance to all columns on the right
    n = dbdt.shape[1] - 1
    triangularNum =  int(n * (n + 1)/2)
    a = np.zeros((dbdt.shape[0], triangularNum))
    j = 0
    for i in range(n):
        for h in  range(i,n):
            a[:, j] = dbdt[:,i+1] - dbdt[:,i]
            j += 1
    return a

    