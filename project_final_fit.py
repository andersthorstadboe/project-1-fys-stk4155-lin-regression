# Importing supporting functions
from reg_functions import *
from support_funcs import *

# Importing relevant packages
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from imageio.v3 import imread
from sklearn.model_selection import train_test_split

#Default for plots
plt.rcParams["figure.figsize"] = (4.5,(5*3/4)) #(15,7)
plt.rcParams["font.size"] = 10

# Random seed
np.random.seed(2018)

# Case selection (Test function, z)
case_s = ['1d', '2d', 'Franke','topo']
case = case_s[2]
shw = 'y'; save = False

# Grid and data setup
a, b   = 1.0, 1.5                                           # Coefficients for exponential model
c0, c1 = 0.01, 0.9                                          # Noise scaling    
x0, xN = 0, 1                                               # Start and end of domain, x-axis
y0, yN = 0, 1                                               # Start and end of domain, y-axis
Nx, Ny = 100, 100                                           # Number of sample points

x   = np.sort(np.random.uniform(x0,xN,Nx)).reshape(-1,1)    # Mesh points on x-axis (uniformly distributed, sorted values)
y   = np.sort(np.random.uniform(y0,yN,Ny)).reshape(-1,1)    # Mesh points on y-axis (uniformly distributed, sorted values) (try different length arrays in x and y if singular values are an issue)
x_n = np.random.normal(0, c0, x.shape)                      # Noise for x-axis
y_n = np.random.normal(0, c0, y.shape)                      # Noise for y-axis

# Test function selection
if case == '1d':
    z_data = exp1D(x,x_n,a=a,b=b,noise=c1)
    zz_data = z_data
    if shw == 'y':
        fig,ax = plt.subplots(1,1)
        ax.scatter(x,z_data)
        ax.set_title('Test function'); ax.set_xlabel('x'); ax.set_ylabel('y',rotation=0,labelpad=15)
    p_ols = 5; p_ridge = 3; p_lasso = 3
    lmbda_ridge = [1e-4]; lmbda_lasso = [6e-6]  
elif case == '2d':
    xx, yy = np.meshgrid(x,y)
    z_data = exp2D(xx,yy,x_n,y_n,a=a,b=b,noise=c1)
    if shw == 'y':
        plot2D(xx,yy,z_data,labels=['Test function','X','Y','Z'])
    p_ols = 5; p_ridge = 5; p_lasso = 4
    lmbda_ridge = [3e-5]; lmbda_lasso = [1e-10]
elif case == 'Franke':
    xx, yy = np.meshgrid(x,y)
    z_data = Franke(xx,yy,x_n,y_n,noise=c1)
    if shw == 'y':
        plot2D(xx,yy,z_data,labels=['Franke function','X','Y','Z'])
    p_ols = 5; p_ridge = 5; p_lasso = 5
    lmbda_ridge = [3e-1]; lmbda_lasso = [1e-15]  
else:
    f_name = ['Etnedal','Jotunheimen']
    z_data, xx, yy, Nx, Ny, im = imageData(f_name=f_name)
    if im == 1:
        p_ols = 6; p_ridge = 9; p_lasso = 9
        lmbda_ridge = [7e3]; lmbda_lasso = [6e1]
        plot2D(xx,yy,z_data,labels=[f_name[im-1],'X','Y','m.a.s'],save=save,f_name='data_etne_final')
    elif im == 2:
        p_ols = 6; p_ridge = 10; p_lasso = 10
        lmbda_ridge = [2e3]; lmbda_lasso = [1e-10]
        plot2D(xx,yy,z_data,labels=[f_name[im-1],'X','Y','m.a.s'],save=save,f_name='data_jotun_final')

if case == '1d':
    x_train,x_test,z_train,z_test = train_test_split(x,z_data,test_size=0.2)
    x_data = [x_train,x_test]; z_data = [z_train,z_test]
else:
    xf = xx.reshape(-1,1); yf = yy.reshape(-1,1); zf = z_data.reshape(-1,1)
    print(zf.mean())
    x_train,x_test,y_train,y_test,z_train,z_test = train_test_split(xf,yf,zf,test_size=0.2)
    x_data = [x_train,x_test,y_train,y_test]; z_data = [z_train,z_test]

y_train_ols,z_test_ols,intcept_ols,beta_ols,mse_s_ols,r2_s_ols = RegOLS(y_data=z_data,x_data=x_data,polydeg=p_ols)
y_train_ridge,z_test_ridge,intcept_ridge,beta_ridge,mse_s_ridge,r2_s_ridge = RegRidge(y_data=z_data,x_data=x_data,polydeg=p_ridge,
                                                                            lmbda=lmbda_ridge,intcept=0)
y_train_lasso,z_test_lasso,intcept_lasso,beta_lasso,mse_s_lasso,r2_s_lasso = RegLasso(y_data=z_data,x_data=x_data,polydeg=p_lasso,
                                                                          lmbda=lmbda_lasso,intcept=0,maxit=20000)

## Printout of assessment values
print()
print('MSE-value')
print('OLS  : ',mse_s_ols)
print('Ridge: ',mse_s_ridge)
print('Lasso: ',mse_s_lasso)
print('')
print('R²-score')
print('OLS  : ',r2_s_ols)
print('Ridge: ',r2_s_ridge)
print('Lasso: ',r2_s_lasso)
print('')

## Design matrix for full prediction model
if case == '1d':
    X_ols = poly_model_1d(x,poly_deg=p_ols)
    X_ridge = poly_model_1d(x,poly_deg=p_ridge)
    X_lasso = poly_model_1d(x,poly_deg=p_lasso)

    ## Final calculation of full prediction model
    z_ols = (X_ols @ beta_ols + intcept_ols)
    z_ridge = (X_ridge @ beta_ridge + intcept_ridge)
    z_lasso = (X_lasso @ beta_lasso[0] + intcept_lasso)
    
    ## Plotting 1D-data
    z = [zz_data,z_ols,z_ridge.reshape(-1,1),z_lasso.reshape(-1,1)]
    plot1D(x_data=x,y_data=z,labels=['Predictions','X','Y','Data','OLS','Ridge','Lasso'])

else:
    X_ols = poly_model_2d(xf,yf,poly_deg=p_ols)
    X_ridge = poly_model_2d(xf,yf,poly_deg=p_ridge)
    X_lasso = poly_model_2d(xf,yf,poly_deg=p_lasso)

    ## Final calculation of full prediction model
    z_ols = (X_ols @ beta_ols + intcept_ols).reshape(Nx,Ny)
    z_ridge = (X_ridge @ beta_ridge + intcept_ridge).reshape(Nx,Ny)
    z_lasso = (X_lasso @ beta_lasso[0] + intcept_lasso).reshape(Nx,Ny)

    ## Plotting 2D-data
    if case == 'topo':
        if im == 1:
            plot2D(xx,yy,z_ols,labels=['OLS-prediction','X','Y','m.a.s'],save=save,f_name='ols_pred_etne_final')
            plot2D(xx,yy,z_ridge,labels=['Ridge-prediction','X','Y','m.a.s'],save=save,f_name='ridge_pred_etne_final')
            plot2D(xx,yy,z_lasso,labels=['Lasso-prediction','X','Y','m.a.s'],save=save,f_name='lasso_pred_etne_final')
        elif im == 2:
            plot2D(xx,yy,z_ols,labels=['OLS-prediction','X','Y','m.a.s'],save=save,f_name='ols_pred_jotun_final')
            plot2D(xx,yy,z_ridge,labels=['Ridge-prediction','X','Y','m.a.s'],save=save,f_name='ridge_pred_jotun_final')
            plot2D(xx,yy,z_lasso,labels=['Lasso-prediction','X','Y','m.a.s'],save=save,f_name='lasso_pred_jotun_final')

    else:
        plot2D(xx,yy,z_ols,labels=['OLS-prediction','X','Y','Z'])
        plot2D(xx,yy,z_ridge,labels=['Ridge-prediction','X','Y','Z'])
        plot2D(xx,yy,z_lasso,labels=['Lasso-prediction','X','Y','Z'])

## Plotting final beta values
x1 = np.arange(1,len(beta_ols)+1)
x2 = np.arange(1,len(beta_ridge[0])+1)
fig,ax = plt.subplots(1,1)
ax.plot(x1,beta_ols,'--s',label='OLS')
ax.plot(x2,beta_ridge[0],'--p',label='Ridge')
ax.plot(x2,beta_lasso[0],'--x',label='Lasso')
ax.set_xticks(np.arange(1,len(x2)+1,2),labels=[f'${{\\beta_{{{x}}}}}$' for x in x2[::2]])
ax.legend(); ax.grid(); ax.set_title(r'$\beta$s for final fit'); ax.set_ylabel(r'$\beta_{i}$')

plt.show()