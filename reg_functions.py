### Imports
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from support_funcs import SVDcalc

### Metric functions

def mse_own(y_data: np.ndarray, y_model: np.ndarray):
    """
    Calculating the mean square error between a data set\n
    and a prediction model, returning the MSE - score.

    Parameters
    --------
    y_data : array
        Data set
    y_model : ndarray
        Prediction data set, model case

    Returns
    --------
    float: The mean square error between y_data and y_model
    
    """
    return np.sum((y_data - y_model)**2)/(np.size(y_model))

def r2score(y_data: np.ndarray, y_model: np.ndarray):
    """
    Calculating the R²-score for y_model compared to y_data.

    Parameters:
    ---
    y_data : array
        Data set
    y_model : array
        Prediction model
    Returns
    ---
    float : R²-score for y_model

    """
    y_mean = np.mean(y_data)
    return 1.0 - ((np.sum((y_data - y_model)**2))/(np.sum((y_data - y_mean)**2)))

### Regression Functions

def OLS(y_data, X, split=0.0, scaling=True, prnt=False):
    """
    Takes a design matrix, X, and data set, y, performing a\n
    ordinary least square (OLS) regression analysis on the data set.\n
    Calculates the MSE and R² - metrics for the resulting prediction model
    
    Parameters
    ---
    y_data : ndarray, n...
        Data set
    X : ndarray, n...
        Design matrix
    split : float
        True for splitting the data into test and training data\n
        False for doing analysis on entire data set
    scaling : bool
        True for scaling of the data
        False for not scaling
    prnt : bool
        For printing metrics to terminal
    
    Returns
    ---
    ndarray : predicted model for plotting

    """

    ## Splitting and scaling conditions  
    if split != 0.0:
        X_train, X_test, y_train, y_test = train_test_split(X,y_data,test_size=split)
        #print('splitting')
    else:
        #print('not splitting')
        y_train = y_data; y_test = y_data
        X_train = X; X_test = 0
    
    if scaling == True:
        #print('scaling')
        # Scaling with the mean value of columns of X
        y_train_mean = np.mean(y_train)
        X_train_mean = np.mean(X_train,axis=0)
        
    else: # NOT WORKING AS INTENDED
        #print('not scaling')
        y_train_mean = 0.0 #np.mean(y_train)
        X_train_mean = 0.0 #np.mean(X_train,axis=0)
        
    y_train_s = y_train - y_train_mean
    X_train_s = X_train - X_train_mean 
    X_test_s  = X_test  - X_train_mean
    
    ## Optimization
    # Optimizing with training data, using a SVD-method
    try:
        beta_ols = (np.linalg.inv(X_train_s.T @ X_train_s) @ X_train_s.T @ y_train_s)
    except np.linalg.LinAlgError:
        beta_ols = SVDcalc(X_train_s) @ y_train_s

    # Calculating the intercept
    if scaling == True:
        intcept_ols = np.mean(y_train_mean - X_train_mean @ beta_ols)
    else: # NOT WORKING
        intcept_ols = np.mean(np.mean(y_train_s) - np.mean(X_train_s,axis=0) @ beta_ols)

    # Predictions, including the intercept
    y_ols_train = X_train_s @ beta_ols + intcept_ols
    y_ols_test  = X_test_s @ beta_ols + intcept_ols

    mse_ols = [mse_own(y_train,y_ols_train),mse_own(y_test,y_ols_test)]
    mse_sk  = [mean_squared_error(y_train,y_ols_train),mean_squared_error(y_test,y_ols_test)]

    r2s_ols = [r2score(y_train,y_ols_train),r2score(y_test,y_ols_test)]
    r2s_sk  = [r2_score(y_train,y_ols_train),r2_score(y_test,y_ols_test)]
    if prnt == True:
        print('Error metrics')
        print('MSE (Own):\nTraining: %g | Test: %g' %(mse_ols[0],mse_ols[1]))
        print('MSE (Scikit):\nTraining: %g | Test: %g' %(mse_sk[0],mse_sk[1]))
        print('R² (Own):\nTraining: %g | Test: %g' %(r2s_ols[0],r2s_ols[1]))
        print('R² (Scikit):\nTraining: %g | Test: %g' %(r2s_sk[0],r2s_sk[1]))

    return y_ols_train,y_ols_test, intcept_ols, beta_ols, mse_ols, r2s_ols

def Ridge(y_data, X, lmbda, intcept, split=0.0, scaling=True, prnt=False):
    """
    
    Parameters
    ---


    Returns
    ---

    """

    ## Splitting and scaling conditions  
    if split != 0.0:
        X_train, X_test, y_train, y_test = train_test_split(X,y_data,test_size=split)
        #print('splitting')
    else:
        #print('not splitting')
        y_train = y_data; y_test = y_data
        X_train = X; X_test = 0
    
    if scaling == True:
        #print('scaling')
        # Scaling with the mean value of columns of X
        y_train_mean = np.mean(y_train)
        X_train_mean = np.mean(X_train,axis=0)
        
    else: # NOT WORKING AS INTENDED
        #print('not scaling')
        y_train_mean = 0.0 #np.mean(y_train)
        X_train_mean = 0.0 #np.mean(X_train,axis=0)
        
    y_train_s = y_train - y_train_mean
    X_train_s = X_train - X_train_mean 
    X_test_s  = X_test  - X_train_mean

    # Identity matrix
    id = np.eye(len(X[0,:]),len(X[0,:]))
    #print('I: ',id.shape)
    #print(id)

    ## Optimization
    # Loop for Ridge optimization with different lambdas
    MSE_ridge_train = np.zeros(len(lmbda)); r2_ridge_train = np.zeros(len(lmbda))
    MSE_ridge_test  = np.zeros(len(lmbda)); r2_ridge_test  = np.zeros(len(lmbda))
    beta_store = []
    for i, lmb in enumerate(lmbda):
        beta_ridge = (np.linalg.inv((X_train_s.T @ X_train_s) + lmb*id) @ X_train_s.T @ y_train_s)
        
        # Storing beta values for lambda
        beta_store.append(beta_ridge)
        
        # Prediction
        y_ridge_train = X_train_s @ beta_ridge + intcept[i]
        y_ridge_test  = X_test_s @ beta_ridge + intcept[i]

        # Storing MSE and R²-scores
        MSE_ridge_train[i] = mse_own(y_train, y_ridge_train)
        MSE_ridge_test[i] = mse_own(y_test, y_ridge_test)

        r2_ridge_train[i] = r2score(y_train, y_ridge_train)
        r2_ridge_test[i] = r2score(y_test, y_ridge_test)
    
    return y_ridge_train,y_ridge_test, intcept, beta_store, [MSE_ridge_train,MSE_ridge_test], [r2_ridge_train,r2_ridge_test]

def Lasso(X):

    return 0

