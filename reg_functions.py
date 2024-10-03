### Imports
import numpy as np
import sklearn.linear_model as sk_lin
from sklearn.utils import resample
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
#from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
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

### Regression Functions

def RegOLS_skl(y_data: list,x_data: list,polydeg: int, scale=True, prnt=False):
    """
    
    """

    if len(x_data) > 2:
        X_train = poly_model_2d(x=x_data[0],y=x_data[2],poly_deg=polydeg)
        X_test = poly_model_2d(x=x_data[1],y=x_data[3],poly_deg=polydeg)
    else:
        X_train = poly_model_1d(x=x_data[0],poly_deg=polydeg)
        X_test = poly_model_1d(x=x_data[1],poly_deg=polydeg)

    y_tr_mean = np.mean(y_data[0],axis=0)
    X_tr_mean = np.mean(X_train,axis=0)
    
    if scale == True:
        y_tr_s = y_data[0] - y_tr_mean
        X_tr_s = X_train - X_tr_mean
    else: 
        y_tr_s = y_data[0]
        X_tr_s = X_train
    #y_tr_s = y_data[0] - y_tr_mean; y_ts_s = y_data[1] - y_ts_mean
    #X_tr_s = X_train - X_tr_mean; X_ts_s = X_test - X_ts_mean 

    # Scikit-learn linear regression method, no intercept, scaled data
    reg_OLS = sk_lin.LinearRegression(fit_intercept=scale)

    if scale == True:
        reg_OLS.fit(X_train,y_data[0])
    else:
        reg_OLS.fit(X_tr_s,y_tr_s)

    y_ols_train = reg_OLS.predict(X_train)
    y_ols_test = reg_OLS.predict(X_test)

    intcept_ols = reg_OLS.intercept_[0]
    beta_ols = reg_OLS.coef_
    print(beta_ols)

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

def RegOLS(y_data: list,x_data: list,polydeg: int, scale=True, prnt=False):
    """
    Performs a ordinary least square regression analysis based on input data, (x, y)\n
    'The intercept column of X must be ommitted.\n'
    Calculates β-values, intercept, and a prediction model, as well as the MSE and R²-metrics for the this model
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
    
    if scale == True:
        y_tr_s = (y_data[0] - y_tr_mean)#/np.std(y_data[0])#,axis=0)
        X_tr_s = (X_train - X_tr_mean)#/np.std(X_train,axis=0)
    else: 
        y_tr_s = y_data[0]
        X_tr_s = X_train

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

def RegRidge(y_data: list,x_data: list,polydeg: int, lmbda: list, intcept: float, scale=True, prnt=False):
    """
    
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
    
    if scale == True:
        y_tr_s = (y_data[0] - y_tr_mean)#/np.std(y_data[0])#,axis=0)
        X_tr_s = (X_train - X_tr_mean)#/np.std(X_train,axis=0)
    else: 
        y_tr_s = y_data[0]
        X_tr_s = X_train

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
        
        # Prediction with added intercept (not centered anymore)
        y_ridge_train = X_train @ beta_ridge + intcept
        y_ridge_test  = X_test @ beta_ridge + intcept

        # Storing MSE and R²-scores
        MSE_ridge_train[i] = mse_own(y_data[0], y_ridge_train)
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

def RegLasso(y_data: list,x_data: list,polydeg: int, lmbda: list, intcept: float, maxit=1000, scale=True, prnt=False):
    """
    
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
    
    if scale == True:
        #scaler_y = StandardScaler(); scaler_X = StandardScaler()
        #scaler_y.fit(y_data[0]); scaler_X.fit(X_train)
        #y_tr_s = scaler_y.transform(y_data[0])
        #print(np.std(y_data[0][:],axis=0))
        y_tr_s = (y_data[0] - y_tr_mean)#/np.std(y_data[0])#,axis=0)
 
        #X_tr_s = scaler_X.transform(X_train)
        #print(X_train)
        #print(np.std(X_train,axis=0))
        X_tr_s = (X_train - X_tr_mean)#/np.std(X_train,axis=0)
    else: 
        y_tr_s = y_data[0]
        X_tr_s = X_train

    ## Optimization
    # Loop for Lasso optimization with different lambdas
    MSE_lasso_train = np.zeros(len(lmbda)); r2_lasso_train = np.zeros(len(lmbda))
    MSE_lasso_test  = np.zeros(len(lmbda)); r2_lasso_test  = np.zeros(len(lmbda))
    beta_store = []

    for i, lmb in enumerate(lmbda):
        #print('On iteration, %i, l = %.3f' %(i,lmb))
        # Importing and fitting Lasso model
        reg_lasso = sk_lin.Lasso(lmb,fit_intercept=False,max_iter=maxit)
        reg_lasso.fit(X_tr_s,y_tr_s)
        
        # Storing beta values for lambda
        #beta_lasso = reg_lasso.coef_
        beta_store.append(reg_lasso.coef_)
        intcepter = reg_lasso.intercept_
        #print(intcept)
        #print(reg_lasso.coef_)
        
        # Prediction with added intercept (not centered anymore)
        y_lasso_train = reg_lasso.predict(X_train) + intcept
        y_lasso_test  = reg_lasso.predict(X_test) + intcept

        # Storing MSE and R²-scores
        #MSE_lasso_train[i] = 
        #MSE_lasso_test[i] = 
        MSE_lasso_train[i] = mean_squared_error(y_data[0], y_lasso_train) #mse_own(y_data[0], y_lasso_train)
        MSE_lasso_test[i] = mean_squared_error(y_data[1], y_lasso_test)   #mse_own(y_data[1], y_lasso_test)

        r2_lasso_train[i] = r2_score(y_data[0], y_lasso_train) #r2score(y_data[0], y_lasso_train)
        r2_lasso_test[i] = r2_score(y_data[1], y_lasso_test)   #r2score(y_data[1], y_lasso_test)
    
    if prnt == True:
        print('\nError metrics, p =', polydeg)
        print('MSE (Own):\nTraining: ',MSE_lasso_train)
        print('Test    : ',MSE_lasso_test)
        print('R²-score (Own):\nTraining: ',r2_lasso_train)
        print('Test    : ',r2_lasso_test)

    return y_lasso_train,y_lasso_test, intcept, beta_store, [MSE_lasso_train,MSE_lasso_test], [r2_lasso_train,r2_lasso_test]

def RegOLS_boot(y_data: list, x_data: list, polydeg: int, n_boots: int, scale=True, prnt=False):
    """
    Performs a ordinary least square regression analysis based on input data, (x, y)\n
    'The intercept column of X must be ommitted.\n'
    Calculates β-values, intercept, and a prediction model, as well as the MSE and R²-metrics for the this model
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
    
    if scale == True:
        y_tr_s = y_data[0] - y_tr_mean
        X_tr_s = X_train - X_tr_mean
    else: 
        y_tr_s = y_data[0]
        X_tr_s = X_train
  
    ## Optimization with bootstrapping
    y_pred = np.empty((y_data[1].shape[0],n_boots))
    for i in range(n_boots):

        # Resampling training data
        X_train_i, y_train_i = resample(X_tr_s,y_tr_s)

        # Prediction
        beta_i = SVDcalc(X_train_i) @ y_train_i

        # Calculating the intercept
        intcept_ols = np.mean(y_tr_mean - X_tr_mean @ beta_i)

        # Predictions, including the intercept (on unscaled data)
        y_pred[:,i] = X_test @ beta_i[:,0] + intcept_ols
    

    return y_pred, intcept_ols, beta_i

def Reg_kfold(y_data: list, x_data: list, polydeg: np.ndarray, folds: int=2, lmbda: list=[], scale=True, maxit=1000):
    """ 

    
    """
    kfold = KFold(n_splits=folds)
    scores_ols, scores_ridge, scores_lasso = {},{},{}
    for i, p_d in enumerate(polydeg):


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
                if scale == True:
                    y_tr_k = y_tr_k - y_tr_mean
                    X_tr_k = X_tr_k - X_tr_mean

                # OLS regression
                beta_ols = SVDcalc(X_tr_k) @ y_tr_k
                intcept_ols = np.mean(y_tr_mean - X_tr_mean @ beta_ols)
                y_ols = X_te_k @ beta_ols[:,0] + intcept_ols

                # Ridge regression
                beta_ridge = (np.linalg.inv((X_tr_k.T @ X_tr_k) + lmb*id) @ X_tr_k.T @ y_tr_k)
                y_ridge  = X_te_k @ beta_ridge + intcept_ols

                # Lasso regression
                reg_lasso = sk_lin.Lasso(lmb,fit_intercept=False,max_iter=maxit)
                reg_lasso.fit(X_tr_k,y_tr_k)
                y_lasso = reg_lasso.predict(X_te_k) + intcept_ols

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

def Reg_Kfold(y_data: list, x_data: list, polydeg: int, lmbda: list=[], scale=True, prnt=False, maxit=1000):
    """
    Performs a ordinary least square regression analysis based on input data, (x, y)\n
    'The intercept column of X must be ommitted.\n'
    Calculates β-values, intercept, and a prediction model, as well as the MSE and R²-metrics for the this model
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
    
    if scale == True:
        y_tr_s = y_data[0] - y_tr_mean
        X_tr_s = X_train - X_tr_mean
    else: 
        y_tr_s = y_data[0]
        X_tr_s = X_train

    id = np.eye(len(X_tr_s[0,:]),len(X_tr_s[0,:]))
    score_ols, score_ridge, score_lasso = np.zeros(len(lmbda)),np.zeros(len(lmbda)),np.zeros(len(lmbda))
    for i, lmb in enumerate(lmbda):

        # OLS regression
        beta_ols = SVDcalc(X_tr_s) @ y_tr_s
        intcept_ols = np.mean(y_tr_mean - X_tr_mean @ beta_ols)
        y_ols = X_test @ beta_ols[:,0] + intcept_ols

        # Ridge regression
        beta_ridge = (np.linalg.inv((X_tr_s.T @ X_tr_s) + lmb*id) @ X_tr_s.T @ y_tr_s)
        y_ridge  = X_test @ beta_ridge + intcept_ols

        # Lasso regression
        reg_lasso = sk_lin.Lasso(lmb,fit_intercept=False,max_iter=maxit)
        reg_lasso.fit(X_tr_s,y_tr_s)
        y_lasso = reg_lasso.predict(X_test) + intcept_ols

        score_ols[i] = (mean_squared_error(y_data[1],y_ols))
        score_ridge[i] = (mean_squared_error(y_data[1],y_ridge))
        score_lasso[i] = (mean_squared_error(y_data[1],y_lasso))

    return score_ols, score_ridge, score_lasso
































def RegLasso1(y_data, X, lmbda, intcept, maxit=1000, split=0.0, scaling=True, prnt=False):
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
    intcept : float
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






# ------------------------------------------ #
# ---------------- Obsoletes --------------- #
# ------------------------------------------ #
def RegOLS_1(y_data, X, split=0.0, scaling=True, prnt=False):
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
        print('scaling')
        # Scaling with the mean value of columns of X
        y_train_mean = np.mean(y_train,axis=0)
        X_train_mean = np.mean(X_train,axis=0)
        print('y_mean: ',y_train_mean)
        print('X_mean: ',X_train_mean)
        
    else: # NOT WORKING AS INTENDED
        print('not scaling')
        y_train_mean = 0.0 #np.mean(y_train)
        X_train_mean = 0.0 #np.mean(X_train,axis=0)
        
    y_train_s = y_train - y_train_mean
    X_train_s = X_train - X_train_mean 
    X_test_s  = X_test  - X_train_mean
    
    ## Optimization
    # Optimizing with training data, using a SVD-method
    try:
        beta_ols = (np.linalg.inv(X_train_s.T @ X_train_s) @ X_train_s.T @ y_train_s)
        print('Beta: ',beta_ols.ravel())
    except np.linalg.LinAlgError:
        print('LinAlgError, singular matrix in (X^T X)^{-1}')
        beta_ols = SVDcalc(X_train_s) @ y_train_s
        print('Beta: ',beta_ols)

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
        print('\nError metrics, p =',len(X[0,:]))
        print('MSE (Own):\nTraining: %g | Test: %g' %(mse_ols[0],mse_ols[1]))
        print('MSE (Scikit):\nTraining: %g | Test: %g' %(mse_sk[0],mse_sk[1]))
        print('R² (Own):\nTraining: %g | Test: %g' %(r2s_ols[0],r2s_ols[1]))
        print('R² (Scikit):\nTraining: %g | Test: %g' %(r2s_sk[0],r2s_sk[1]))

    return y_ols_train,y_ols_test, intcept_ols, beta_ols, mse_ols, r2s_ols

def RegRidge1(y_data, X, lmbda, intcept, split=0.0, scaling=True, prnt=False):
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
