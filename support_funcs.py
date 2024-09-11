import numpy as np
import matplotlib.pyplot as plt

### Test functions

def Franke(x,y,x_noise,y_noise,noise=0.0):
   """
   Returns the Franke function on a mesh grid (x,y)\n
   as test function, with or without noise added
   
   Parameters
   --------
   x, y : array, n x m
      Mesh grid data points
   x_noise, y_noise : array, n x m
      Mesh grid noise data points
   noise : float
      Scale for the amount of noise added to the output,\n
      same on both axis x and y. Default = 0.0

   Returns
   --------
   ndarray : 2D-function representing the Franke function

   """
   p1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
   p2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
   p3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
   p4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)

   return p1 + p2 + p3 + p4 + noise*x_noise + noise*y_noise

def exp1D(x,x_noise,a,b,noise=0.0):
   """
   Returns a 1D exponential test function,\n
   a*exp(-x²) + b*exp(-(x-2)²) + noise\n
   for a ndarray x, with or without noise given as ndarray x_noise

   Parameters
   --------
   x : ndarray, n
      Data points
   x_noise : ndarray, n
      Noise data array
   a, b : float
      Scale of first and second exponential term
   noise : float
      Scale for the amount of noise added to the output. Default = 0.0

   Returns
   --------
   ndarray : 1D exponential function
   """
   return a*np.exp(-x**2) + b*np.exp(-(x-2)**2) + noise*x_noise

def exp2D(x,y,x_noise,y_noise,a,b,noise=0.0):
   """
   Returns a 2D exponential test function,\n

   for a mesh grid (x,y) with or without noise 
   from mesh grid (x_noise, y_noise)

   Parameters
   ---
   x, y : array, n x m
      Mesh grid data points
   x_noise, y_noise : array, n x m
      Mesh grid noise data points
   a, b : float
      Scale of first and second exponential term
   noise : float
      Scale for the amount of noise added to the output,\n
      same on both axis x and y. Default = 0.0
   
   Returns
   ---
   ndarray : 2D exponential function
   """
   p1 = a*np.exp(-(x**2 + y**2)) + b*np.exp(-(x-2)**2 - (y-2)**2)

   return p1 + noise*x_noise + noise*y_noise

### Other supporting functions

def poly_model_1d(x: np.ndarray,poly_deg: int):
    """
    Returning a design matrix for a polynomial of a given degree in\n
    one variable, x. Starts at i = 1, so intercept column is ommitted\n 
    from design matrix\n

    Parameters
    --------
    x : array
        1d-numpy array, dimension n\n
    poly_deg : int
        degree of the resulting polynomial to be modelled, p
    
    Returns
    --------
    X : array, n x p 
        Design matrix of dimension n x p
    """
    X = np.zeros((len(x),poly_deg))
    for p_d in range(1, poly_deg+1):
        X[:,p_d-1] = x**p_d
    return X