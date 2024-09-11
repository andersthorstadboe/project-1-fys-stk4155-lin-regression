### Imports
import numpy as np

### Functions

def OLS(X):
    
    return 0

def Ridge(X):

    return 0

def Lasso(X):

    return 0


def mse_own(y_data, y_model):
    return np.sum((y_data - y_model)**2)/(np.size(y_model))

def poly_model_1d(x,poly_deg):
    X = np.zeros(len(x),poly_deg)
    for p_d in range(1, poly_deg+1):
        X[:,p_d-1] = x**p_d
    return X
