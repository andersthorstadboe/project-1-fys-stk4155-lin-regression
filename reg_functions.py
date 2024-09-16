### Imports
import numpy as np
import sklearn.linear_model as sk_lin
from sklearn.linear_model import Lasso
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

def RegOLS(y_data, X, split=0.0, scaling=True, prnt=False):
    """
    Performs a ordinary least square regression analysis based on input data, (y, X)\n
    The intercept column of X must be ommitted.\n
    Calculates β-values, intercept, and a prediction model, as well as the MSE and R²-metrics for the this model
    
    Parameters
    ---
    y_data : ndarray, n...
        Data set
    X : ndarray, n x (p-1)
        Design matrix
    split : float, Default = 0.0
        The amount of data that will be used for testing the model
    scaling: bool, Default = True
        Whether or not data will be scaled before prediction
    prnt : bool, Default = False
        Gives regression metric output in terminal if selected
    
    Returns
    ---
    y_ols_train,y_ols_test, intcept_ols, beta_ols, mse_ols, r2s_ols
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

def RegRidge(y_data, X, lmbda, intcept, split=0.0, scaling=True, prnt=False):
    """
    Performs a Ridge regression analysis based on input data (y, X), for input λ-values.\n
    The intercept column of X must be ommitted.\n
    Calculates 

    Parameters
    ---
    y_data : ndarray
        Data set
    X : ndarray, n x (p-1)
        Design matrix without the intercept included
    lmbda : list
        List of λ-values to do analysis with
    intcept : ndarray
        Intercept values for the different cases for y-model based on OLS-regression analysis
    split : float, Default = 0.0
        The amount of data that will be used for testing the model
    scaling: bool, Default = True
        Whether or not data will be scaled before prediction
    prnt : bool, Default = False
        Gives regression metric output in terminal if selected

    Returns
    ---
    y_ridge_train : ndarray

    y_ridge_test : 

    intcept : ndarray

    beta_store : list

    MSE_ridge : list

    R2_ridge : list
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

    ## Optimization
    # Loop for Ridge optimization with different lambdas
    MSE_ridge_train = np.zeros(len(lmbda)); r2_ridge_train = np.zeros(len(lmbda))
    MSE_ridge_test  = np.zeros(len(lmbda)); r2_ridge_test  = np.zeros(len(lmbda))
    beta_store = []
    for i, lmb in enumerate(lmbda):
        beta_ridge = (np.linalg.inv((X_train_s.T @ X_train_s) + lmb*id) @ X_train_s.T @ y_train_s)
        
        # Storing beta values for lambda
        beta_store.append(beta_ridge)
        
        # Prediction with added intercept (not centered anymore)
        y_ridge_train = X_train_s @ beta_ridge + intcept[i]
        y_ridge_test  = X_test_s @ beta_ridge + intcept[i]

        # Storing MSE and R²-scores
        MSE_ridge_train[i] = mse_own(y_train, y_ridge_train)
        MSE_ridge_test[i] = mse_own(y_test, y_ridge_test)

        r2_ridge_train[i] = r2score(y_train, y_ridge_train)
        r2_ridge_test[i] = r2score(y_test, y_ridge_test)
    
        # COPIED FROM OLS-FUNCTION, MUST BE REWRITTEN
        #if prnt == True:
        #    print('Error metrics')
        #    print('MSE (Own):\nTraining: %g | Test: %g' %(mse_ols[0],mse_ols[1]))
        #    print('R² (Own):\nTraining: %g | Test: %g' %(r2s_ols[0],r2s_ols[1]))
    
    return y_ridge_train,y_ridge_test, intcept, beta_store, [MSE_ridge_train,MSE_ridge_test], [r2_ridge_train,r2_ridge_test]

def RegLasso(y_data, X, lmbda, intcept, maxit=1000, split=0.0, scaling=True, prnt=False):
    """
    Performs a Lasso regression analysis based on input data (y, X), for input λ-values.\n 
    The intercept column of X must be ommitted.\n
    Uses the sklearn.linear_model Lasso-method for the prediction.
    
    Parameters
    ---
    y_data : ndarray
        Data set
    X : ndarray, n x (p-1)
        Design matrix without the intercept included
    lmbda : list
        List of λ-values to do analysis with
    intcept : ndarray
        Intercept values for the different cases for y-model based on OLS-regression analysis
    maxit : int
        Setting the maximum number of iterations for the Lasso-method. Increase with warned about convergence issues
    split : float, Default = 0.0
        The amount of data that will be used for testing the model
    scaling: bool, Default = True
        Whether or not data will be scaled before prediction
    prnt : bool, Default = False
        Gives regression metric output in terminal if selected

    Returns
    ---
    y_ridge_train : ndarray

    y_ridge_test : 

    intcept : ndarray

    beta_store : list

    MSE_ridge : list

    R2_ridge : list
    """
    ## Splitting and scaling conditions  
    if split != 0.0:
        X_train, X_test, y_train, y_test = train_test_split(X,y_data,test_size=split)
        #print('splitting')
    else: # Need to verify
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
    # Loop for Lasso optimization with different lambdas
    MSE_lasso_train = np.zeros(len(lmbda)); r2_lasso_train = np.zeros(len(lmbda))
    MSE_lasso_test  = np.zeros(len(lmbda)); r2_lasso_test  = np.zeros(len(lmbda))
    beta_store = []

    for i, lmb in enumerate(lmbda):
        
        # Importing and fitting Lasso model
        reg_lasso = sk_lin.Lasso(lmb,fit_intercept=False,max_iter=maxit)
        reg_lasso.fit(X_train_s,y_train_s)
        
        # Storing beta values for lambda
        #beta_lasso = reg_lasso.coef_
        beta_store.append(reg_lasso.coef_)
        
        # Prediction with added intercept (not centered anymore)
        y_lasso_train = reg_lasso.predict(X_train_s) + intcept[i]
        y_lasso_test  = reg_lasso.predict(X_test_s) + intcept[i]

        # Storing MSE and R²-scores
        MSE_lasso_train[i] = mse_own(y_train, y_lasso_train)
        MSE_lasso_test[i] = mse_own(y_test, y_lasso_test)

        r2_lasso_train[i] = r2score(y_train, y_lasso_train)
        r2_lasso_test[i] = r2score(y_test, y_lasso_test)

        # COPIED FROM OLS-FUNCTION, MUST BE REWRITTEN
        #if prnt == True:
        #    print('Error metrics')
        #    print('MSE (Own):\nTraining: %g | Test: %g' %(mse_ols[0],mse_ols[1]))
        #    print('R² (Own):\nTraining: %g | Test: %g' %(r2s_ols[0],r2s_ols[1]))

    return y_lasso_train,y_lasso_test, intcept, beta_store, [MSE_lasso_train,MSE_lasso_test], [r2_lasso_train,r2_lasso_test]

