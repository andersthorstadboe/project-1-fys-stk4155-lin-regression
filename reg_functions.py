### Imports
import numpy as np
import sklearn.linear_model as sk_lin
from sklearn.utils import resample
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score,mean_squared_error, root_mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler,MaxAbsScaler,RobustScaler,PowerTransformer
from support_funcs import SVDcalc,poly_model_1d, poly_model_2d

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

### Scaling
def dataScaler(data,type='',range=(0,1)):
    """
    Scales a data-array based on input type of common scaling methods in Scikit-learn.
    Returns scaled data-array. If no type is given, data-array is returned unscaled
    """
    if type == 'standard':
        scaler = StandardScaler(with_std=False)
    elif type == 'minmax':
        scaler = MinMaxScaler(feature_range=range)
    elif type == 'maxabs':
        scaler = MaxAbsScaler()
    elif type == 'robust':
        scaler = RobustScaler()
    elif type == 'power':
        scaler = PowerTransformer()
    else:
        data_scaled = data
    
    scaler.fit(data)
    data_scaled = scaler.transform(data)

    return data_scaled#, scaler

### Regression Functions
def RegOLS(y_data: list,x_data: list, polydeg: int, scale: str=None, prnt=False):
    """
    Performs a ordinary least square regression analysis based on input data, (x, y). 
    The intercept column of X will be omitted by default if not stated in support function. 
    Calculates β-values, intercept, and a prediction model, as well as the MSE and R²-metrics for the this model

    Parameters
    ---
    y_data : list
        Dataset function values, split in training and testing datasets
    x_data : list
        Dataset function variables, split in training and testing datasets
    polydeg : int
        Integer p, creating a design matrix of degree, p
    scale : bool
        Default = True. Whether or not data will be scaled before prediction
    prnt : bool
        Whether or not to give output to terminal of calculated metrics

    Returns
    ---
    Prediction models (y_train, y_test), β-values, intercept, MSE-value, R²-value

    """
    ## Making design matrices for training and test data (checking if we are in a 1d- or 2d-case)
    if len(x_data) > 2:
        X_train = poly_model_2d(x=x_data[0],y=x_data[2],poly_deg=polydeg)
        X_test = poly_model_2d(x=x_data[1],y=x_data[3],poly_deg=polydeg)
    else:
        X_train = poly_model_1d(x=x_data[0],poly_deg=polydeg)
        X_test = poly_model_1d(x=x_data[1],poly_deg=polydeg)
    print(X_train.shape)
    ## Scaling the training data by subtracting the mean
    y_tr_mean = np.mean(y_data[0])#,axis=0)
    X_tr_mean = np.mean(X_train,axis=0)
    
    if scale != None:
        print('here')
        #y_tr_s = dataScaler(y_data[0],type=scale)
        y_tr_s = (y_data[0] - y_tr_mean)/np.std(y_data[0])#,axis=0)
        X_tr_s = dataScaler(X_train,type=scale)
        #X_tr_s = (X_train - X_tr_mean)/np.std(X_train,axis=0)
    else: 
        y_tr_s = y_data[0] - y_tr_mean
        X_tr_s = X_train - X_tr_mean

    ## Optimization
    # Optimizing with training data, using a SVD-method
    try:
        beta_ols = (np.linalg.inv(X_tr_s.T @ X_tr_s) @ X_tr_s.T @ y_tr_s)
    except np.linalg.LinAlgError:
        #print('LinAlgError, singular matrix in (X^T X)^{-1}, using SVD-method to calc β-values')
        beta_ols = SVDcalc(X_tr_s) @ y_tr_s

    # Calculating the intercept
    intcept_ols = np.mean(y_tr_mean - X_tr_mean @ beta_ols)

    # Predictions, including the intercept (on unscaled data)
    y_ols_train = X_train @ beta_ols + intcept_ols
    y_ols_test  = X_test @ beta_ols + intcept_ols

    ## Regression metrics, MSE and R²-score
    mse_ols = [mse_own(y_data[0],y_ols_train),mse_own(y_data[1],y_ols_test)]
    mse_sk  = [mean_squared_error(y_data[0],y_ols_train),mean_squared_error(y_data[1],y_ols_test)]

    r2s_ols = [r2score(y_data[0],y_ols_train),r2score(y_data[1],y_ols_test)]
    r2s_sk  = [r2_score(y_data[0],y_ols_train),r2_score(y_data[1],y_ols_test)]    
    if prnt == True:
        print('\nError metrics, p =', polydeg)
        print('MSE (Own):\nTraining: %g | Test: %g' %(mse_ols[0],mse_ols[1]))
        print('MSE (Scikit):\nTraining: %g | Test: %g' %(mse_sk[0],mse_sk[1]))
        print('R² (Own):\nTraining: %g | Test: %g' %(r2s_ols[0],r2s_ols[1]))
        print('R² (Scikit):\nTraining: %g | Test: %g' %(r2s_sk[0],r2s_sk[1]))

    return y_ols_train,y_ols_test, intcept_ols, beta_ols, mse_ols, r2s_ols

def RegRidge(y_data: list,x_data: list,polydeg: int, lmbda: list, intcept: float, scale: str=None, prnt=False):
    """
    Performs a Ridge-regression analysis based on input data, (x, y), using different λ-values. 
    The intercept column of X will be omitted by default if not stated in support function. 
    Calculates β-values, intercept, and a prediction model, as well as the MSE and R²-metrics for the this model

    Parameters
    ---
    y_data : list
        Dataset function values, split in training and testing datasets
    x_data : list
        Dataset function variables, split in training and testing datasets
    polydeg : int
        Integer p, creating a design matrix of degree, p
    lmbda : list
        List of λ-values to do analysis with
    scale : bool
        Default = True. Whether or not data will be scaled before prediction
    prnt : bool
        Whether or not to give output to terminal of calculated metrics

    Returns
    ---
    Prediction models (y_train, y_test), β-values, intercept, MSE-value, R²-value

    """
    ## Making design matrices for training and test data (checking if we are in a 1d- or 2d-case)
    if len(x_data) > 2:
        X_train = poly_model_2d(x=x_data[0],y=x_data[2],poly_deg=polydeg)
        X_test = poly_model_2d(x=x_data[1],y=x_data[3],poly_deg=polydeg)
    else:
        X_train = poly_model_1d(x=x_data[0],poly_deg=polydeg)
        X_test = poly_model_1d(x=x_data[1],poly_deg=polydeg)

    print(X_train.shape)
    ## Scaling the training data by subtracting the mean
    y_tr_mean = np.mean(y_data[0])#,axis=0)
    X_tr_mean = np.mean(X_train,axis=0)
    
    if scale != None:
        #y_tr_s = dataScaler(y_data[0],type=scale)
        y_tr_s = (y_data[0] - y_tr_mean)/np.std(y_data[0])#,axis=0)
        X_tr_s = dataScaler(X_train,type=scale)
        #X_tr_s = (X_train - X_tr_mean)/np.std(X_train,axis=0)
    else: 
        y_tr_s = y_data[0] - y_tr_mean
        X_tr_s = X_train - X_tr_mean

    # Identity matrix
    id = np.eye(len(X_tr_s[0,:]),len(X_tr_s[0,:]))

    ## Optimization
    # Loop for Ridge optimization with different lambdas
    MSE_ridge_train = np.zeros(len(lmbda)); r2_ridge_train = np.zeros(len(lmbda))
    MSE_ridge_test  = np.zeros(len(lmbda)); r2_ridge_test  = np.zeros(len(lmbda))
    beta_store = []
    for i, lmb in enumerate(lmbda):
        beta_ridge = (np.linalg.inv((X_tr_s.T @ X_tr_s) + lmb*id) @ X_tr_s.T @ y_tr_s)
        
        # Storing beta values for lambda
        beta_store.append(beta_ridge)
        
        intcept = np.mean(y_tr_mean - X_tr_mean @ beta_store[i])

        # Prediction with added intercept (not centered anymore)
        y_ridge_train = X_train @ beta_ridge + intcept
        y_ridge_test  = X_test @ beta_ridge + intcept

        # Storing MSE and R²-scores
        MSE_ridge_train[i] = mean_absolute_error(y_data[1],y_ridge_test) #mse_own(y_data[0], y_ridge_train)
        MSE_ridge_test[i] = mse_own(y_data[1], y_ridge_test)

        r2_ridge_train[i] = r2score(y_data[0], y_ridge_train)
        r2_ridge_test[i] = r2score(y_data[1], y_ridge_test)

    if prnt == True:
        print('\nError metrics, p =', polydeg)
        print('MSE (Own):\nTraining: ',MSE_ridge_train)
        print('Test    : ',MSE_ridge_test)
        print('R²-score (Own):\nTraining: ',r2_ridge_train)
        print('Test    : ',r2_ridge_test)
        
    return y_ridge_train,y_ridge_test, intcept, beta_store, [MSE_ridge_train,MSE_ridge_test], [r2_ridge_train,r2_ridge_test]

def RegLasso(y_data: list,x_data: list,polydeg: int, lmbda: list, intcept: float, maxit=1000, scale: str=None, prnt=False):
    """
    Performs a Ridge-regression analysis based on input data, (x, y), using different λ-values. 
    The intercept column of X will be omitted by default if not stated in support function. 
    Calculates β-values, intercept, and a prediction model, as well as the MSE and R²-metrics for the this model

    Parameters
    ---
    y_data : list
        Dataset function values, split in training and testing datasets
    x_data : list
        Dataset function variables, split in training and testing datasets
    polydeg : int
        Integer p, creating a design matrix of degree, p
    lmbda : list
        List of λ-values to do analysis with
    maxit : int
        Default = 1000. Number of max iterations for input to sklearn.linear_model.Lasso
    scale : bool
        Default = True. Whether or not data will be scaled before prediction
    prnt : bool
        Whether or not to give output to terminal of calculated metrics

    Returns
    ---
    Prediction models (y_train, y_test), β-values, intercept, MSE-value, R²-value
    """
    ## Making design matrices for training and test data (checking if we are in a 1d- or 2d-case)
    if len(x_data) > 2:
        X_train = poly_model_2d(x=x_data[0],y=x_data[2],poly_deg=polydeg)
        X_test = poly_model_2d(x=x_data[1],y=x_data[3],poly_deg=polydeg)
    else:
        print('here')
        X_train = poly_model_1d(x=x_data[0],poly_deg=polydeg)
        X_test = poly_model_1d(x=x_data[1],poly_deg=polydeg)

    print(X_train.shape)

    ## Scaling the training data by subtracting the mean
    y_tr_mean = np.mean(y_data[0])#,axis=0)
    X_tr_mean = np.mean(X_train,axis=0)
    
    if scale != None:
        #y_tr_s = dataScaler(y_data[0],type=scale)
        y_tr_s = (y_data[0] - y_tr_mean)/np.std(y_data[0])#,axis=0)
        X_tr_s = dataScaler(X_train,type=scale)
        #X_tr_s = (X_train - X_tr_mean)/np.std(X_train,axis=0)
    else: 
        y_tr_s = y_data[0] - y_tr_mean
        X_tr_s = X_train - X_tr_mean

    ## Optimization
    # Loop for Lasso optimization with different lambdas
    MSE_lasso_train = np.zeros(len(lmbda)); r2_lasso_train = np.zeros(len(lmbda))
    MSE_lasso_test  = np.zeros(len(lmbda)); r2_lasso_test  = np.zeros(len(lmbda))
    beta_store = []

    for i, lmb in enumerate(lmbda):
        # Importing and fitting Lasso model
        #print('lmb:', lmb)
        reg_lasso = sk_lin.Lasso(lmb,fit_intercept=False,max_iter=maxit)
        reg_lasso.fit(X_tr_s,y_tr_s)
        
        # Storing beta values for lambda
        beta_store.append(reg_lasso.coef_)

        intcept = np.mean(y_tr_mean - X_tr_mean @ beta_store[i])
        
        # Prediction with added intercept (not centered anymore)
        y_lasso_train = reg_lasso.predict(X_train) + intcept
        y_lasso_test  = reg_lasso.predict(X_test) + intcept

        # Storing MSE and R²-scores
        MSE_lasso_train[i] = root_mean_squared_error(y_data[1],y_lasso_test)#mean_squared_error(y_data[0], y_lasso_train) #mse_own(y_data[0], y_lasso_train)
        MSE_lasso_test[i] = mean_squared_error(y_data[1], y_lasso_test)   #mse_own(y_data[1], y_lasso_test)

        r2_lasso_train[i] = r2_score(y_data[0], y_lasso_train) #r2score(y_data[0], y_lasso_train)
        r2_lasso_test[i] = r2_score(y_data[1], y_lasso_test)   #r2score(y_data[1], y_lasso_test)
    
    if prnt == True:
        print('\nError metrics, p =', polydeg)
        print('MSE (sklearn):\nTraining: ',MSE_lasso_train)
        print('Test    : ',MSE_lasso_test)
        print('R²-score (sklearn):\nTraining: ',r2_lasso_train)
        print('Test    : ',r2_lasso_test)

    return y_lasso_train,y_lasso_test, intcept, beta_store, [MSE_lasso_train,MSE_lasso_test], [r2_lasso_train,r2_lasso_test]

def RegOLS_boot(y_data: list, x_data: list, polydeg: int, n_boots=1, scale: str=None, prnt=False):
    """
    Performs a ordinary least square regression analysis based on input data, (x, y), using bootstrapping\n
    The intercept column of X must be ommitted.\n'
    Calculates β-values, intercept, and a prediction model as output

    Parameters
    ---
    y_data : list
        Dataset function values, split in training and testing datasets
    x_data : list
        Dataset function variables, split in training and testing datasets
    polydeg : int
        Integer p, creating a design matrix of degree, p
    n_boots : int
        Default = 1. Number of iterations to bootstrap over. 
    scale : bool
        Default = True. Whether or not data will be scaled before prediction
    prnt : bool
        Whether or not to give output to terminal of calculated metrics

    Returns
    ---
    y_pred, β-values, intercept

    """
    ## Making design matrices for training and test data (checking if we are in a 1d- or 2d-case)
    if len(x_data) > 2:
        X_train = poly_model_2d(x=x_data[0],y=x_data[2],poly_deg=polydeg)
        X_test = poly_model_2d(x=x_data[1],y=x_data[3],poly_deg=polydeg)
    else:
        X_train = poly_model_1d(x=x_data[0],poly_deg=polydeg)
        X_test = poly_model_1d(x=x_data[1],poly_deg=polydeg)

    ## Scaling the training data by subtracting the mean
    y_tr_mean = np.mean(y_data[0])#,axis=0)
    X_tr_mean = np.mean(X_train,axis=0)
    
    if scale != None:
        print('here')
        #y_tr_s = dataScaler(y_data[0],type=scale)
        y_tr_s = (y_data[0] - y_tr_mean)/np.std(y_data[0])#,axis=0)
        X_tr_s = dataScaler(X_train,type=scale)
        #X_tr_s = (X_train - X_tr_mean)/np.std(X_train,axis=0)
    else: 
        y_tr_s = y_data[0] - y_tr_mean
        X_tr_s = X_train - X_tr_mean
  
    ## Optimization with bootstrapping
    y_pred = np.empty((y_data[1].shape[0],n_boots))
    for i in range(n_boots):

        # Resampling training data
        X_train_i, y_train_i = resample(X_tr_s,y_tr_s)

        # Prediction
        beta_i = SVDcalc(X_train_i) @ y_train_i

        # Calculating the intercept
        intcept_ols = np.mean(y_tr_mean - X_tr_mean @ beta_i[:,0])

        # Predictions, including the intercept (on unscaled data)
        y_pred[:,i] = X_test @ beta_i[:,0] + intcept_ols
    
    return y_pred, intcept_ols, beta_i

def Reg_kfold(y_data: list, x_data: list, polydeg: np.ndarray, folds: int=2, lmbda: list=[], scale: str=None, maxit=1000):
    """ 
    Performs a regression analysis based on the K-fold cross-validation method, using OLS-, Ridge- and Lasso regression cost-functions.

    Parameters
    ---
    y_data : list
        Dataset function values, split in training and testing datasets
    x_data : list
        Dataset function variables, split in training and testing datasets
    polydeg : NDArray
        Array of integers, p, for creating a design matrix of degree, p
    folds : int
        Number of folds to split training dataset into. Default = 2
    lmbda : list
        List of λ-values to do analysis with
    scale : bool 
        Default = True. Whether or not data will be scaled before prediction
    maxit : int
        Default = 1000. Number of max iterations for input to sklearn.linear_model.Lasso

    Returns
    ---
    k-fold method MSE-scores for the three methods, tuple
    
    """
    kfold = KFold(n_splits=folds)
    scores_ols, scores_ridge, scores_lasso = {},{},{}
    for i, p_d in enumerate(polydeg):
        print('kfold')
        print('p = ',p_d)

        ## Making design matrices for training and test data (checking if we are in a 1d- or 2d-case)
        if len(x_data) > 2:
            X_train = poly_model_2d(x=x_data[0],y=x_data[2],poly_deg=p_d)
        else:
            X_train = poly_model_1d(x=x_data[0],poly_deg=p_d)

        scores_kfold_ols = np.zeros((len(lmbda),folds))
        scores_kfold_ridge = np.zeros((len(lmbda),folds))
        scores_kfold_lasso = np.zeros((len(lmbda),folds))

        for j, lmb in enumerate(lmbda):

            for k, (tr_idx, te_idx) in enumerate(kfold.split(x_data[0])):
                X_tr_k, X_te_k = X_train[tr_idx], X_train[te_idx]
                y_tr_k, y_te_k = y_data[0][tr_idx], y_data[0][te_idx]

                ## Identity matrix for Ridge
                id = np.eye(len(X_tr_k[0,:]),len(X_tr_k[0,:]))

                ## Scaling the training data by subtracting the mean
                y_tr_mean = np.mean(y_tr_k)#,axis=0)
                X_tr_mean = np.mean(X_tr_k,axis=0)
                if scale != None:
                    print('here')
                    #y_tr_k = dataScaler(y_tr_k,type=scale)
                    y_tr_k = (y_tr_k - y_tr_mean)/np.std(y_data[0])#,axis=0)
                    X_tr_k = dataScaler(X_tr_k,type=scale)
                    #X_tr_s = (X_train - X_tr_mean)/np.std(X_train,axis=0)
                else:
                    y_tr_k = (y_tr_k - y_tr_mean)
                    X_tr_k = (X_tr_k - X_tr_mean)

                # OLS regression
                beta_ols = SVDcalc(X_tr_k) @ y_tr_k
                intcept_ols = np.mean(y_tr_mean - X_tr_mean @ beta_ols)
                y_ols = X_te_k @ beta_ols[:,0] + intcept_ols

                # Ridge regression
                beta_ridge = (np.linalg.inv((X_tr_k.T @ X_tr_k) + lmb*id) @ X_tr_k.T @ y_tr_k)
                intcept_ridge = np.mean(y_tr_mean - X_tr_mean @ beta_ridge)
                y_ridge  = X_te_k @ beta_ridge + intcept_ridge

                # Lasso regression
                reg_lasso = sk_lin.Lasso(lmb,fit_intercept=False,max_iter=maxit)
                reg_lasso.fit(X_tr_k,y_tr_k)
                intcept_lasso = np.mean(y_tr_mean - X_tr_mean @ reg_lasso.coef_)
                y_lasso = reg_lasso.predict(X_te_k) + intcept_lasso

                scores_kfold_ols[j,k] = mean_squared_error(y_te_k,y_ols)#np.sum((y_ols - y_te_k)**2)/np.size(y_ols)
                scores_kfold_ridge[j,k] = mean_squared_error(y_te_k,y_ridge)#np.sum((y_ridge - y_te_k)**2)/np.size(y_ridge)
                scores_kfold_lasso[j,k] = mean_squared_error(y_te_k,y_lasso)#np.sum((y_lasso - y_te_k)**2)/np.size(y_lasso)
                #end k for
            #end j for
        
        scores_ols['p_'+str(p_d)] = np.mean(scores_kfold_ols,axis=1)
        scores_ridge['p_'+str(p_d)] = np.mean(scores_kfold_ridge,axis=1)
        scores_lasso['p_'+str(p_d)] = np.mean(scores_kfold_lasso,axis=1)
        #end i for

    return scores_ols,scores_ridge,scores_lasso