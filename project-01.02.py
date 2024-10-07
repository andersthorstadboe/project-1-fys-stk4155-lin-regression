# Importing supporting functions
from reg_functions import *
from support_funcs import *

# Importing relevant packages
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split, KFold

#Default for plots
plt.rcParams["figure.figsize"] = (5,4)
plt.rcParams["font.size"] = 10

# Random seed
np.random.seed(2018)

# Case selection (Test function, z)
case_s = ['1d', '2d', 'Franke']
case = case_s[2]
scaler = None #'minmax'
shw = 'n'; plotting = 'y'

# Grid and data setup
a, b   = 1.0, 1.5                                           # Coefficients for exponential model
c0, c1 = 0.01, 0.95                                         # Noise scaling    
x0, xN = 0, 1                                               # Start and end of domain, x-axis
y0, yN = 0, 1                                               # Start and end of domain, y-axis
Nx, Ny = 15, 15                                             # Number of sample points

x   = np.sort(np.random.uniform(x0,xN,Nx)).reshape(-1,1)    # Mesh points on x-axis (uniformly distributed, sorted values)
y   = np.sort(np.random.uniform(y0,yN,Ny)).reshape(-1,1)    # Mesh points on y-axis (uniformly distributed, sorted values) (try different length arrays in x and y if singular values are an issue)
x_n = np.random.normal(0, c0, x.shape)                      # Noise for x-axis
y_n = np.random.normal(0, c0, y.shape)                      # Noise for y-axis

# Test function selection
if case == '1d':
    z = exp1D(x,x_n,a=a,b=b,noise=c1)
    if shw == 'y':
        fig,ax = plt.subplots(1,1)
        ax.scatter(x,z)
        ax.set_title('Test function'); ax.set_xlabel('x'); ax.set_ylabel('y',rotation=0,labelpad=15)
elif case == '2d':
    xx, yy = np.meshgrid(x,y)
    z = exp2D(xx,yy,x_n,y_n,a=a,b=b,noise=c1)
    if shw == 'y':
        plot2D(xx,yy,z,labels=['Test function','','X','Y','Z'])
elif case == 'Franke':
    xx, yy = np.meshgrid(x,y)
    z = Franke(xx,yy,x_n,y_n,noise=c1)
    if shw == 'y':
        plot2D(xx,yy,z,labels=['','','X','Y','Z'])

## Parameter setup
maxdegree   = 10
poly_deg    = np.arange(1,maxdegree+1,1)

n_lmbda = 6
l_min,l_max = -10,10
lmbda = np.logspace(l_min,l_max,n_lmbda)

## Resampling setup
n_boots = 1000
folds = 8; kfold = KFold(n_splits=folds)
# Restating lambda in same range, with more values to get a smoother line
N_lmbda = n_lmbda*5
Lmbda = np.logspace(l_min,l_max,N_lmbda)

## Training and test data ratio
train_split = 4/5
test_split  = 1.0 - train_split 
train_l     = int(np.round(len(x)*train_split))
test_l      = int(np.round(len(x)*test_split))

# Dataset splitting
if case == '1d':
    x_train,x_test,z_train,z_test = train_test_split(x,z,test_size=test_split)
    x_data = [x_train,x_test]; z_data = [z_train,z_test]
else:
    xf = xx.reshape(-1,1); yf = yy.reshape(-1,1); zf = z.reshape(-1,1)
    x_train,x_test,y_train,y_test,z_train,z_test = train_test_split(xf,yf,zf,test_size=test_split)
    x_data = [x_train,x_test,y_train,y_test]; z_data = [z_train,z_test]

## Dictionaries for storing values from regression analysis
mod_train, mod_test = {},{}
intcept, betas      = {},{}
mse_tr_s, mse_te_s  = {},{}
r2_tr_s,r2_te_s     = {},{}

## ---------- OLS-regression ---------- ##
# Loop storage for output values from regression
mod_tr_ols,mod_ts_ols = {},{}
itcpt_ols,btas_ols    = {},{}
mse_tr_ols,mse_te_ols = {},{}
r2_tr_ols,r2_te_ols   = {},{}

# Storing intercept values for OLS-analysis for using in Ridge and Lasso-analysis
beta_0 = np.zeros(maxdegree)

# OLS loop
for i, p_d in enumerate(poly_deg):

    y_train_tmp,z_test_tmp,intcept_tmp,beta_tmp,mse_s_tmp,r2_s_tmp = RegOLS(y_data=z_data,x_data=x_data,
                                                                            polydeg=p_d,scale=scaler)
    mod_tr_ols['train_p_'+str(p_d)] = y_train_tmp; mod_ts_ols['test_p_'+str(p_d)] = z_test_tmp
    itcpt_ols['p_'+str(p_d)] = intcept_tmp; btas_ols['p_'+str(p_d)] = beta_tmp
    mse_tr_ols['p_'+str(p_d)] = mse_s_tmp[0]; mse_te_ols['p_'+str(p_d)] = mse_s_tmp[1]; 
    r2_tr_ols['p_'+str(p_d)] = r2_s_tmp[0]; r2_te_ols['p_'+str(p_d)] = r2_s_tmp[1] 

    beta_0[i] = intcept_tmp

# Storing results for OLS analysis
mod_train['y_tr_ols'] = mod_tr_ols; mod_test['y_ts_ols'] = mod_ts_ols; 
intcept['intercept_ols'] = itcpt_ols; betas['betas_ols'] = btas_ols
mse_tr_s['mse_ols'] = mse_tr_ols; mse_te_s['mse_ols'] = mse_te_ols; 
r2_tr_s['r2_ols'] = r2_tr_ols; r2_te_s['r2_ols'] = r2_te_ols

## ---------- Ridge-regression ---------- ##

# Loop storage for output values from regression
y_tr_ridge,y_ts_ridge     = {},{}
itcpt_ridge,btas_ridge    = {},{}
mse_tr_ridge,mse_te_ridge = {},{}
r2_tr_ridge,r2_te_ridge   = {},{}

# Ridge loop
for i, p_d in enumerate(poly_deg):

    y_train_tmp,z_test_tmp,intcept_tmp,beta_tmp,mse_s_tmp,r2_s_tmp = RegRidge(y_data=z_data,x_data=x_data,polydeg=p_d,
                                                                        lmbda=lmbda, intcept=beta_0[i],scale=scaler)
    y_tr_ridge['train_p_'+str(p_d)] = y_train_tmp; y_ts_ridge['test_p_'+str(p_d)] = z_test_tmp
    itcpt_ridge['p_'+str(p_d)] = intcept_tmp; btas_ridge['p_'+str(p_d)] = beta_tmp
    mse_tr_ridge['p_'+str(p_d)] = mse_s_tmp[0]; mse_te_ridge['p_'+str(p_d)] = mse_s_tmp[1]
    r2_tr_ridge['p_'+str(p_d)] = r2_s_tmp[0]; r2_te_ridge['p_'+str(p_d)] = r2_s_tmp[1]

# Storing result from Ridge analysis
mod_train['y_tr_ridge'] = y_tr_ridge; mod_test['y_ts_ridge'] = y_ts_ridge; 
intcept['intercept_ridge'] = itcpt_ridge; betas['betas_ridge'] = btas_ridge
mse_tr_s['mse_ridge'] = mse_tr_ridge; mse_te_s['mse_ridge'] = mse_te_ridge; 
r2_tr_s['r2_ridge'] = r2_tr_ridge; r2_te_s['r2_ridge'] = r2_te_ridge

## ---------- Lasso Regression ---------- ##
# Loop storage for output values from regression
y_tr_lasso, y_ts_lasso    = {},{}
itcpt_lasso,btas_lasso    = {},{}
mse_tr_lasso,mse_te_lasso = {},{}
r2_tr_lasso,r2_te_lasso   = {},{}

#Lasso loop
maxiter = 20000
for i, p_d in enumerate(poly_deg):

    y_train_tmp,z_test_tmp,intcept_tmp,beta_tmp,mse_s_tmp,r2_s_tmp = RegLasso(y_data=z_data,x_data=x_data, polydeg=p_d,
                                                                        lmbda=lmbda, intcept=beta_0[i], maxit=maxiter,scale=scaler)
    y_tr_lasso['train_p_'+str(p_d)] = y_train_tmp; y_ts_lasso['test_p_'+str(p_d)] = z_test_tmp
    itcpt_lasso['p_'+str(p_d)] = intcept_tmp; btas_lasso['p_'+str(p_d)] = beta_tmp
    mse_tr_lasso['p_'+str(p_d)] = mse_s_tmp[0]; mse_te_lasso['p_'+str(p_d)] = mse_s_tmp[1]; 
    r2_tr_lasso['p_'+str(p_d)] = r2_s_tmp[0]; r2_te_lasso['p_'+str(p_d)] = r2_s_tmp[1] 

# Storing results from Lasso analysis
mod_train['y_tr_lasso'] = y_tr_lasso; mod_test['y_ts_lasso'] = y_ts_lasso; 
intcept['intercept_lasso'] = itcpt_lasso; betas['betas_lasso'] = btas_lasso
mse_tr_s['mse_lasso'] = mse_tr_lasso; mse_te_s['mse_lasso'] = mse_te_lasso; 
r2_tr_s['r2_lasso'] = r2_tr_lasso; r2_te_s['r2_lasso'] = r2_te_lasso

## ---------- Bias-Variance tradeoff with bootstrapping ---------- ##
# Storage for error, bias and variance values
err_b,bias_b,var_b = np.zeros(len(poly_deg)),np.zeros(len(poly_deg)),np.zeros(len(poly_deg))
beta = []

for i,p_d in enumerate(poly_deg):
    print('p =',p_d)
    
    z_test_tmp,intcept_tmp,beta_tmp = RegOLS_boot(y_data=z_data, x_data=x_data, polydeg=p_d, n_boots=n_boots,scale=scaler)

    err_b[i]  = np.mean( np.mean((z_data[1] - z_test_tmp)**2, axis=1, keepdims=True) )
    bias_b[i] = np.mean( (z_data[1] - np.mean(z_test_tmp, axis=1, keepdims=True))**2 )
    var_b[i]  = np.mean( np.var(z_test_tmp, axis=1, keepdims=True) )
    beta.append(beta_tmp)

## ---------- K-fold cross validation ---------- ##
scores_ols, scores_ridge, scores_lasso = Reg_kfold(y_data=z_data,x_data=x_data,polydeg=poly_deg,folds=folds
                                                   ,lmbda=Lmbda,maxit=maxiter,scale=scaler)

## ---------- Post-processing data ---------- ##
## Averaging the kfold-scores per polynomial degree to compare to OLS-error from bootstrapping
err_ols_kfold = np.zeros(len(scores_ols))
err_ridge_kfold = np.zeros(len(scores_ridge))
err_lasso_kfold = np.zeros(len(scores_lasso))

i = 0
for f in list(scores_ols):
    err_ols_kfold[i] = np.mean(scores_ols[f])
    err_ridge_kfold[i] = np.mean(scores_ridge[f])
    err_lasso_kfold[i] = np.mean(scores_lasso[f])
    
    i+=1

## Final training with optimal values
mse_min_ridge = []; mse_min_lasso = []
idx_min_r = []; idx_min_l = []
for f in list(scores_ridge):
    min_idx_r = np.argmin(scores_ridge[f])
    min_idx_l = np.argmin(scores_lasso[f])
    
    idx_min_r.append(min_idx_r); idx_min_l.append(min_idx_l)
    mse_min_ridge.append(scores_ridge[f][min_idx_r]); mse_min_lasso.append(scores_ridge[f][min_idx_l]) 

## ---------- Plotting results ---------- ##
clrmap = ['inferno','twilight','viridis','gray','coolwarm']
plotting = 'n'
if plotting == 'y':

    lmbda = np.logspace(l_min,l_max,n_lmbda)
    ## Heatmaps of Ridge and Lasso results
    mse_ridge,mse_lasso = {},{}
    r2_ridge,r2_lasso   = {},{}
    mse_ridge['mse_tr_ridge'] = mse_tr_ridge; mse_ridge['mse_te_ridge'] = mse_te_ridge; 
    mse_lasso['mse_tr_lasso'] = mse_tr_lasso; mse_lasso['mse_te_lasso'] = mse_te_lasso; 
    r2_ridge['r2_tr_ridge'] = r2_tr_ridge; r2_ridge['r2_te_ridge'] = r2_te_ridge; 
    r2_lasso['r2_tr_lasso'] = r2_tr_lasso; r2_lasso['r2_te_lasso'] = r2_te_lasso; 
    
    plot_heatmap(x_data=poly_deg,y_data=lmbda,values=mse_lasso,labels=['Lasso','MSE','poly.deg','λ','Training','Test'])#,clrmap=clrmap[])
    plot_heatmap(x_data=poly_deg,y_data=lmbda,values=r2_lasso,labels=['Lasso','R²','poly.deg','λ','Training','Test'])#,clrmap=clrmap[2])
    plot_heatmap(x_data=poly_deg,y_data=lmbda,values=mse_ridge,labels=['Ridge','MSE','poly.deg','λ','Training','Test'])#,clrmap=clrmap[4])
    plot_heatmap(x_data=poly_deg,y_data=lmbda,values=r2_ridge,labels=['Ridge','R²','poly.deg','λ','Training','Test'])#,clrmap=clrmap[4])
    
    ## Line plot of OLS results
    OLS_data = {'mse_tr': mse_tr_s['mse_ols'],'mse_te': mse_te_s['mse_ols'],'r2_tr': r2_tr_s['r2_ols'],'r2_te': r2_te_s['r2_ols']}
    plot_OLS(poly_deg,y_data=OLS_data,labels=['OLS','poly.deg','MSE','R²'])

    ## Plotting beta-values for OLS
    beta_plot(poly_deg,betas['betas_ols'],labels=[r'$\hat{\beta}_{\text{OLS}}$',r'$\beta_{i}$',r'$\hat{\beta}$'])

    ## Plotting final result from bootstrapping vs. polynomial degree
    fig,bx = plt.subplots(1,1)
    bx.plot(poly_deg,err_b, label='Error')
    bx.plot(poly_deg,bias_b,label='Bias')
    bx.plot(poly_deg,var_b, label='Var')
    bx.legend(); bx.set_title('Bias-Variance tradeoff\n#Resamples = %i' %(n_boots))
    bx.set_xlabel('poly.deg'); bx.set_ylabel('Error',rotation=0,labelpad=15)
    bx.grid(); 

    ## Comparison between bootstrap OLS and Kfold OLS
    fig,cx = plt.subplots(1,1)
    cx.plot(poly_deg,err_b,label='Boot')
    cx.plot(poly_deg,err_ols_kfold,label='OLS-Kfold')
    cx.plot(poly_deg,err_ridge_kfold,label='Ridge-Kfold')
    cx.plot(poly_deg,err_lasso_kfold,label='Lasso-Kfold')
    cx.set_xlabel('poly.deg'); cx.set_ylabel('MSE',rotation=0,labelpad=15)
    cx.set_title('Comparison, bootstrap vs. Kfold')
    cx.legend(); cx.grid()
    fig.tight_layout()

    ## Comparison between regression and Kfold, Ridge and Lasso
    # Downsampling scores from Kfold to get a more readable heatmap-figure
    sc_ridge,sc_lasso = {},{}
    nn = int(N_lmbda/n_lmbda)
    for i in list(scores_ridge):
        sc_ridge[i] = scores_ridge[i][::nn]
        sc_lasso[i] = scores_lasso[i][::nn]

    # K-fold scores for Ridge and Lasso
    fig0,ax = plt.subplots(1,1)
    fig1,bx = plt.subplots(1,1)
    for i in list(scores_ridge):
        ax.plot(np.log10(Lmbda),scores_ridge[i],label=i)
        bx.plot(np.log10(Lmbda),scores_lasso[i],label=i)
    ax.set_xlabel(r'log$_{10}$(λ)'); ax.set_ylabel('MSE',rotation=0,labelpad=15)
    ax.set_ylabel('MSE',rotation=0,labelpad=15)
    bx.set_xlabel(r'log$_{10}$(λ)'); bx.set_ylabel('MSE',rotation=0,labelpad=15)
    bx.set_ylabel('MSE',rotation=0,labelpad=15)
    ax.legend(); ax.grid()
    bx.legend(); bx.grid()
    ax.set_title('Ridge-case'); bx.set_title('Lasso-case'); 
    fig0.suptitle('Kfold-scores, k = %i' %(folds),y=.98,x=.5225)
    fig0.tight_layout()
    fig1.suptitle('Kfold-scores, k = %i' %(folds),y=.98,x=.5225)
    fig1.tight_layout()

    # Heatmap-plots, K-fold
    heat_ridge_mse,heat_lasso_mse = {},{}
    heat_ridge_mse['mse_te_ridge'] = mse_te_ridge; heat_ridge_mse['mse_tr_ridge_kfold'] = sc_ridge
    heat_lasso_mse['mse_te_lasso'] = mse_te_lasso; heat_lasso_mse['mse_tr_lasso_kfold'] = sc_lasso

    plot_heatmap(x_data=poly_deg,y_data=lmbda,values=heat_ridge_mse,labels=['Ridge','MSE','poly.deg','λ','Regression','Kfold'])#,clrmap=clrmap[1])
    plot_heatmap(x_data=poly_deg,y_data=lmbda,values=heat_lasso_mse,labels=['Lasso','MSE','poly.deg','λ','Regression','Kfold'])#,clrmap=clrmap[1])

    plt.show()

## Fit with optimal parameters, OLS, 1D-case
if case == '1d':
    # OLS-fit
    idx_o_poly = np.argmin(err_b)
    p_ols = idx_o_poly+1; 
    X_ols = poly_model_1d(x=x,poly_deg=p_ols)
    b_ols = beta[p_ols-1]                                           # Picking b from bootstrapping results
    intcepter = np.mean(np.mean(z) - np.mean(X_ols,axis=0) @ b_ols) # Calc of intercept to scale back fit to correct starting value, used on all three
    y_pred_ols = X_ols @ b_ols + intcepter

    # Ridge-fit
    idx_r_poly = np.argmin(mse_min_ridge)
    p_ridge = idx_r_poly+1; p_r = 'p_'+str(p_ridge)
    lmb_r = Lmbda[idx_min_r[idx_r_poly]]
    idx_r_lmb = np.where(lmbda == lmb_r)[0]
    if idx_r_lmb.size == 0:
        idx_r_lmb = np.abs(lmbda - lmb_r).argmin()
    else:
        idx_r_lmb = idx_r_lmb[0]
    X_ridge = poly_model_1d(x=x,poly_deg=idx_r_poly+1)
    b_ridge = betas['betas_ridge'][p_r][idx_r_lmb]
    intcepter = np.mean(np.mean(z) - np.mean(X_ridge,axis=0) @ b_ridge)
    y_pred_ridge = X_ridge @ b_ridge + intcepter

    # Lasso-fit
    idx_l_poly = np.argmin(mse_min_lasso)
    p_lasso = idx_l_poly+1; p_l = 'p_'+str(p_lasso)
    lmb_l = Lmbda[idx_min_l[idx_l_poly]]
    idx_l_lmb = np.where(lmbda == lmb_l)[0]
    if idx_l_lmb.size == 0:
        idx_l_lmb = np.abs(lmbda - lmb_l).argmin()
    else:
        idx_l_lmb = idx_l_lmb[0]
    X_lasso = poly_model_1d(x=x,poly_deg=idx_l_poly+1)
    b_lasso = betas['betas_lasso'][p_l][idx_l_lmb]
    intcepter = np.mean(np.mean(z) - np.mean(X_lasso,axis=0) @ b_lasso)
    y_pred_lasso = X_lasso @ b_lasso + intcepter 
    
    # Plotting regression models against dataset, 1D
    fig,ax = plt.subplots(1,1)
    ax.scatter(x,z,alpha=0.5)
    ax.plot(x,y_pred_ols,'C1',label='OLS')
    ax.plot(x,y_pred_ridge,'C2',label='Ridge')
    ax.plot(x,y_pred_lasso,'--C3',label='Lasso')
    ax.legend(); ax.grid()
    plt.show()
    
else: # Used if dataset is 2D
    # OLS-fit
    idx_o_poly = np.argmin(err_b)
    p_ols = idx_o_poly+1; 
    X_ols = poly_model_2d(x=xf,y=yf,poly_deg=p_ols)
    b_ols = beta[p_ols-1]                                           # Picking b from bootstrapping results
    intcepter = np.mean(np.mean(z) - np.mean(X_ols,axis=0) @ b_ols) 
    y_pred_ols = X_ols @ b_ols + intcepter

    # Ridge-fit
    idx_r_poly = np.argmin(mse_min_ridge)
    p_ridge = idx_r_poly+1; p_r = 'p_'+str(p_ridge)
    lmb_r = Lmbda[idx_min_r[idx_r_poly]]
    idx_r_lmb = np.where(lmbda == lmb_r)[0]
    if idx_r_lmb.size == 0:
        idx_r_lmb = np.abs(lmbda - lmb_r).argmin()
    else:
        idx_r_lmb = idx_r_lmb[0]
    X_ridge = poly_model_2d(x=xf,y=yf,poly_deg=p_ridge)
    b_ridge = betas['betas_ridge'][p_r][idx_r_lmb]
    intcepter = np.mean(np.mean(z) - np.mean(X_ridge,axis=0) @ b_ridge)
    y_pred_ridge = X_ridge @ b_ridge + intcepter

    # Lasso-fit
    idx_l_poly = np.argmin(mse_min_lasso)
    p_lasso = idx_l_poly+1; p_l = 'p_'+str(p_lasso)
    lmb_l = Lmbda[idx_min_l[idx_l_poly]]
    idx_l_lmb = np.where(lmbda == lmb_l)[0]
    if idx_l_lmb.size == 0:
        idx_l_lmb = np.abs(lmbda - lmb_l).argmin()
    else:
        idx_l_lmb = idx_l_lmb[0]
    X_lasso = poly_model_2d(x=xf,y=yf,poly_deg=p_lasso)
    b_lasso = betas['betas_lasso'][p_l][idx_l_lmb]
    intcepter = np.mean(np.mean(z) - np.mean(X_lasso,axis=0) @ b_lasso)
    y_pred_lasso = X_lasso @ b_lasso + intcepter  

    y_pred_ols = y_pred_ols.reshape(Nx,Ny)
    y_pred_ridge = y_pred_ridge.reshape(Nx,Ny)
    y_pred_lasso = y_pred_lasso.reshape(Nx,Ny)
    
    plot2D(xx,yy,y_pred_ols,labels=['OLS-prediction','X','Y','Z'])
    plot2D(xx,yy,y_pred_ridge,labels=['Ridge-prediction','X','Y','Z'])
    plot2D(xx,yy,y_pred_lasso,labels=['Lasso-prediction','X','Y','Z'])

    fig,ax = plt.subplots(1,1)
    ax.plot(poly_deg,err_b,label='ols')
    ax.plot(poly_deg,mse_min_lasso,label='lasso')
    ax.plot(poly_deg,mse_min_ridge,'--',label='ridge')
    ax.set_xlabel('poly_deg'); ax.set_ylabel('MSE')
    ax.set_title('Resulting MSE-values vs. model complexity')
    ax.legend(); ax.grid()

    plt.show()

## Printing values to terminal:
print()
print('MSE-values')
print('OLS   = ',err_b)
print('Ridge = ',mse_min_ridge)
print('Lasso = ',mse_min_lasso)
print('R²-values')
print('OLS: ', r2_te_s.pop('r2_ols'))
print('Ridge: ', r2_te_s.pop('r2_ridge'))
print('Lasso: ', r2_te_s.pop('r2_lasso'))
print('Chosen polynomial degrees')
print('OLS   = ',p_ols,' | Ridge = ',p_ridge, ' | Lasso = ',p_lasso)
print('Chosen lambdas')
print('Ridge = ',lmb_r, ' | Lasso = ',lmb_l)