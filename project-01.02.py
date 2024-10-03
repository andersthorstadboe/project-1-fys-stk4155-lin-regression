# Importing supporting functions
from reg_functions import *
from support_funcs import *

# Importing relevant packages
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split

#Default for plots
plt.rcParams["figure.figsize"] = (15,7)

# Random seed
np.random.seed(2018)

# Case selection (Test function, z)
case_s = ['1d', '2d', 'Franke']
case = case_s[0]
shw = 'y'

# Grid and data setup
a, b   = 1.0, 1.5                                           # Coefficients for exponential model
c0, c1 = 0.01, 0.0                                          # Noise scaling    
x0, xN = 0, 1                                               # Start and end of domain, x-axis
y0, yN = 0, 0.8                                             # Start and end of domain, y-axis
Nx, Ny = 100, 100                                           # Number of sample points

x   = np.sort(np.random.uniform(x0,xN,Nx)).reshape(-1,1)    # Mesh points on x-axis (uniformly distributed, sorted values)
y   = np.sort(np.random.uniform(y0,yN,Ny)).reshape(-1,1)    # Mesh points on y-axis (uniformly distributed, sorted values) (try different length arrays in x and y if singular values are an issue)
x_n = np.random.normal(0, c0, x.shape)                      # Noise for x-axis
y_n = np.random.normal(0, c0, y.shape)                      # Noise for y-axis

# Test function selection
if case == '1d':
    z_data = exp1D(x,x_n,a=a,b=b,noise=c1)
    if shw == 'y':
        fig,ax = plt.subplots(1,1)
        ax.scatter(x,z_data)
        ax.set_title('Test function'); ax.set_xlabel('x'); ax.set_ylabel('y',rotation=0,labelpad=15)
elif case == '2d':
    xx, yy = np.meshgrid(x,y)
    z_data = exp2D(x,y,x_n,y_n,a=a,b=b,noise=c1)
    zz_data = exp2D(xx,yy,x_n,y_n,a=a,b=b,noise=c1)
    if shw == 'y':
        plot2D(xx,yy,zz_data,labels=['Test function','','X','Y','Z'])
elif case == 'Franke':
    xx, yy = np.meshgrid(x,y)
    z_data = Franke(x,y,x_n,y_n,noise=c1)
    zz_data = Franke(xx,yy,x_n,y_n,noise=c1)
    if shw == 'y':
        plot2D(xx,yy,zz_data,labels=['','','x','y','z'])

## Polynomial degree setup
maxdegree   = 5
poly_deg    = np.arange(1,maxdegree+1,1)

## Training and test data ratio
train_split = 4/5
test_split  = 1.0 - train_split 
train_l     = int(np.round(len(x)*train_split))
test_l      = int(np.round(len(x)*test_split))

# Dataset splitting
if case == '1d':
    x_train,x_test,z_train,z_test = train_test_split(x,z_data,test_size=test_split)
    x_data = [x_train,x_test]; z_data = [z_train,z_test]
else:
    x_train,x_test,y_train,y_test,z_train,z_test = train_test_split(x,y,z_data,test_size=test_split)
    x_data = [x_train,x_test,y_train,y_test]; z_data = [z_train,z_test]

## Dictionaries for storing values from regression analysis
mod_train, mod_test = {},{}
intcept, betas  = {},{}
mse_s, r2_s     = {},{}

## ---------- OLS-regression ---------- ##
# Loop storage for output values from regression
mod_tr_ols, mod_ts_ols = {},{}
itcpt_ols,btas_ols = {},{}
mse_ols,r2_ols     = {},{}

# Storing intercept values for OLS-analysis for using in Ridge and Lasso-analysis
beta_0 = np.zeros(maxdegree)

# OLS loop
for i, p_d in enumerate(poly_deg):

    #y_train_tmp,y_test_tmp,intcept_tmp,beta_tmp,mse_s_tmp,r2_s_tmp = RegOLS_skl(y_data=z_data,x_data=x_data,polydeg=p_d)
    y_train_tmp,y_test_tmp,intcept_tmp,beta_tmp,mse_s_tmp,r2_s_tmp = RegOLS(y_data=z_data,x_data=x_data,polydeg=p_d)

    mod_tr_ols['train_p_'+str(p_d)] = y_train_tmp; mod_ts_ols['test_p_'+str(p_d)] = y_test_tmp
    itcpt_ols['p_'+str(p_d)] = intcept_tmp; btas_ols['beta_p_'+str(p_d)] = beta_tmp
    mse_ols['p_'+str(p_d)] = mse_s_tmp; r2_ols['p_'+str(p_d)] = r2_s_tmp 

    beta_0[i] = intcept_tmp

# Storing results for OLS analysis
mod_train['y_tr_ols'] = mod_tr_ols; mod_test['y_ts_ols'] = mod_ts_ols; 
intcept['intercept_ols'] = itcpt_ols; betas['betas_ols'] = btas_ols
mse_s['mse_ols'] = mse_ols; r2_s['r2_ols'] = r2_ols

## ---------- Ridge-regression ---------- ##
lmbda = np.logspace(-3,4,10)
#lmbda = [1e-5,1e-4,1e-3,1e-2,1e-1,1e0]

# Loop storage for output values from regression
y_tr_ridge, y_ts_ridge = {},{}
itcpt_ridge,btas_ridge = {},{}
mse_ridge,r2_ridge     = {},{}
#'''
# Ridge loop
for i, p_d in enumerate(poly_deg):

    y_train_tmp,y_test_tmp,intcept_tmp,beta_tmp,mse_s_tmp,r2_s_tmp = RegRidge(y_data=z_data,x_data=x_data,polydeg=p_d,
                                                                        lmbda=lmbda, intcept=beta_0[i], prnt=False)

    y_tr_ridge['train_p_'+str(p_d)] = y_train_tmp; y_ts_ridge['test_p_'+str(p_d)] = y_test_tmp
    itcpt_ridge['p_'+str(p_d)] = intcept_tmp; btas_ridge['beta_p_'+str(p_d)] = beta_tmp
    mse_ridge['p_'+str(p_d)] = mse_s_tmp; r2_ridge['p_'+str(p_d)] = r2_s_tmp 

# Storing result from Ridge analysis
mod_train['y_tr_ridge'] = y_tr_ridge; mod_test['y_ts_ridge'] = y_ts_ridge; 
intcept['intercept_ridge'] = itcpt_ridge; betas['betas_ridge'] = btas_ridge
mse_s['mse_ridge'] = mse_ridge; r2_s['r2_ridge'] = r2_ridge

## ---------- Lasso Regression ---------- ##
# Loop storage for output values from regression
y_tr_lasso, y_ts_lasso = {},{}
itcpt_lasso,btas_lasso = {},{}
mse_lasso,r2_lasso     = {},{}

#Lasso loop
maxiter = 2000
for i, p_d in enumerate(poly_deg):

    y_train_tmp,y_test_tmp,intcept_tmp,beta_tmp,mse_s_tmp,r2_s_tmp = RegLasso(y_data=z_data,x_data=x_data, polydeg=p_d,
                                                                        lmbda=lmbda, intcept=beta_0[i], maxit=maxiter, scale=True)#, prnt=True)
    
    y_tr_lasso['train_p_'+str(p_d)] = y_train_tmp; y_ts_lasso['test_p_'+str(p_d)] = y_test_tmp
    itcpt_lasso['p_'+str(p_d)] = intcept_tmp; btas_lasso['beta_p_'+str(p_d)] = beta_tmp
    mse_lasso['p_'+str(p_d)] = mse_s_tmp; r2_lasso['p_'+str(p_d)] = r2_s_tmp 

# Storing results from Lasso analysis
mod_train['y_tr_lasso'] = y_tr_lasso; mod_test['y_ts_lasso'] = y_ts_lasso; 
intcept['intercept_lasso'] = itcpt_lasso; betas['betas_lasso'] = btas_lasso
mse_s['mse_lasso'] = mse_lasso; r2_s['r2_lasso'] = r2_lasso

## ---------- Bias-Variance tradeoff with bootstrapping ---------- ##
# Storage for error, bias and variance values
#'''
err_b,bias_b,var_b = np.zeros(len(poly_deg)),np.zeros(len(poly_deg)),np.zeros(len(poly_deg))
beta = []

n_boots = 1000

for i,p_d in enumerate(poly_deg):
    
    y_test_tmp,intcept_tmp,beta_tmp = RegOLS_boot(y_data=z_data, x_data=x_data, polydeg=p_d, n_boots=n_boots)

    err_b[i]  = np.mean( np.mean((z_test - y_test_tmp)**2, axis=1, keepdims=True) )
    bias_b[i] = np.mean( (z_test - np.mean(y_test_tmp, axis=1, keepdims=True))**2 )
    var_b[i]  = np.mean( np.var(y_test_tmp, axis=1, keepdims=True) )
    beta.append(beta_tmp)

## ---------- K-fold cross validation ---------- ##
from sklearn.model_selection import KFold
folds = 5; kfold = KFold(n_splits=folds)

# Restating lambda in same range, with more values to get a smoother line
lmbda = np.logspace(-3,4,100)

## Kfold-method
scores_ols, scores_ridge, scores_lasso = Reg_kfold(y_data=z_data,x_data=x_data,polydeg=poly_deg,folds=folds,lmbda=lmbda,maxit=maxiter)

## ---------- Post-processing data ---------- ##
## Averaging the kfold-scores per polynomial degree to compare to OLS-error from bootstrapping
err_ols_kfold = np.zeros(len(scores_ols))
err_ridge_kfold = np.zeros(len(scores_ols))
err_lasso_kfold = np.zeros(len(scores_ols))

i = 0
for f in list(scores_ols):
    err_ols_kfold[i] = np.mean(scores_ols[f])
    err_ridge_kfold[i] = np.mean(scores_ridge[f])
    err_lasso_kfold[i] = np.mean(scores_lasso[f])
    
    i+=1

## ---------- Plotting results ---------- ##
clrmap = ['inferno','twilight','viridis','gray','coolwarm']
'''
## Heatmaps of Ridge and Lasso results
plot_heatmap(x_data=poly_deg,y_data=lmbda,values=mse_s['mse_lasso'],labels=['Lasso','MSE','poly.deg','λ','Training','Test'],clrmap=clrmap[1])
plot_heatmap(x_data=poly_deg,y_data=lmbda,values=r2_s['r2_lasso'],labels=['Lasso','R²','poly.deg','λ','Training','Test'],clrmap=clrmap[2])
plot_heatmap(x_data=poly_deg,y_data=lmbda,values=mse_s['mse_ridge'],labels=['Ridge','MSE','poly.deg','λ','Training','Test'],clrmap=clrmap[3])
plot_heatmap(x_data=poly_deg,y_data=lmbda,values=r2_s['r2_ridge'],labels=['Ridge','R²','poly.deg','λ','Training','Test'],clrmap=clrmap[4])

## Line plot of OLS results
plot_OLS(poly_deg,y_data=[mse_s['mse_ols'],r2_s['r2_ols']],labels=['OLS','poly.deg','MSE','R²'])

## Plotting beta-values for OLS
#beta_plot(poly_deg,betas_ols,x_label='poly. degree',y_label=r'$\hat{\beta}$')

## Plotting final result from bootstrapping vs. polynomial degree
fig,bx = plt.subplots(1,1)
bx.plot(poly_deg,err_b, label='Error')
bx.plot(poly_deg,bias_b,label='Bias')
bx.plot(poly_deg,var_b, label='Var')
bx.legend(); bx.set_title('Bias-Variance tradeoff\n#Resamples = %i' %(n_boots))
bx.set_xlabel('poly.deg'); bx.set_ylabel('Error',rotation=0,labelpad=15)
bx.grid(); 
#'''

## K-fold scores for Ridge and Lasso
fig,ax = plt.subplots(3,1)
for i in list(scores_ridge):
    ax[0].plot(np.log10(lmbda),scores_ridge[i],label=i)
    ax[1].plot(np.log10(lmbda),scores_lasso[i],label=i)
    ax[2].plot(np.log10(lmbda),scores_ols[i],label=i)
ax[2].set_xlabel(r'log$_{10}$(λ)'); ax[1].set_ylabel('MSE',rotation=0,labelpad=15)
ax[0].legend(); ax[1].legend(); ax[0].grid(),ax[1].grid()

## Comparison between bootstrap OLS and Kfold OLS
fig,cx = plt.subplots(1,1)
cx.plot(poly_deg,err_b,label='Boot')
cx.plot(poly_deg,err_ols_kfold,label='OLS-Kfold')
cx.plot(poly_deg,err_ridge_kfold,label='Ridge-Kfold')
cx.plot(poly_deg,err_lasso_kfold,label='Lasso-Kfold')
cx.set_xlabel('poly.deg'); cx.set_ylabel('MSE',rotation=0,labelpad=15)
cx.set_title('Comparison, bootstrap vs. Kfold')
cx.legend(); cx.grid()


## Plots to show resulting prediction model
#plot_reg1D(x_data=x,y_data=z_data,b_data=betas_ols,b0=intcept_ols,labels=['OLS','x',r'$\tilde{y}$(x)'])





plt.show()