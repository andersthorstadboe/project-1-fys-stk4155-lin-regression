import numpy as np
import matplotlib.pyplot as plt

def Franke(x,y):
    """Returns the Franke function for x and y arrays
       as f(x,y)
    """
    p1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    p2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    p3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    p4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)

    return p1 + p2 + p3 + p4

