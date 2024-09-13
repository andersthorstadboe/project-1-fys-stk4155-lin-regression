# Imports from supporting functions
from reg_functions import *
from support_funcs import *

# Import of relevant packages
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Default for plots
plt.rcParams["figure.figsize"] = (15,7)

# Random seed
np.random.seed(1000)

# Case selection (Data function)
case_s = ['1d', '2d', 'Franke']
case = case_s[0]

a, b, c = 1.0, 1.5, 0.05
train_split = 4.0/5.0
test_split  = 1.0 - train_split 

x0,xN = 0, 1
y0,yN = 0, 1
N = 100
x = np.sort(np.random.uniform(x0,xN,N))#np.linspace(x0,xN,N)
y = np.sort(np.random.uniform(y0,yN,N))
x_n = np.random.normal(0, c, x.shape)#np.random.randn(N)
y_n = np.random.normal(0, c, x.shape)


maxdegree = 5
poly_deg = np.arange(1,maxdegree+1,1)
train_l = int(np.round(len(x)*train_split)); test_l = int(np.round(len(x)*test_split))
print(train_l)
print(test_l)

y_train_ols, y_test_ols = np.zeros((train_l,maxdegree)),np.zeros((test_l,maxdegree))
intcept_ols, betas_ols      = np.zeros(maxdegree),[]
mse_s_ols, r2s_s_ols    = np.zeros((2,maxdegree)),np.zeros((2,maxdegree))
for i, p_d in enumerate(poly_deg):
    #print(i, p_d)
    if case == '1d':
        z_data = exp1D(x,x_n,a=a,b=b,noise=c)
        X = poly_model_1d(x=x,poly_deg=p_d)
    elif case == '2d':
        z_data = exp2D(x,y,x_n,y_n,a=a,b=b,noise=c)
        X = poly_model2d(x=x,y=y,poly_deg=p_d)
    elif case == 'Franke':
        z = Franke(x,y,x_n,y_n,noise=c)
        X = poly_model2d(x=x,y=y,poly_deg=p_d)
    
    y_train_ols[:,i],y_test_ols[:,i],intcept_ols[i],beta_tmp,mse_s_ols[:,i],r2s_s_ols[:,i] = OLS(y_data=z_data,X=X,split=test_split)#,scaling=False)
    betas_ols.append(beta_tmp)
    #print(betas[i])

lmbda = [1e-4,1e-3,1e-2,1e-1,1e0]
y_train_ridge, y_test_ridge = np.zeros((train_l,maxdegree)),np.zeros((test_l,maxdegree))
intcept_ridge, betas_ridge  = np.zeros((maxdegree,maxdegree)),[]
mse_s_ridge, r2s_s_ridge    = [],[] #np.zeros((2,maxdegree)),np.zeros((2,maxdegree))
for i, p_d in enumerate(poly_deg):
    #print(i, p_d)
    if case == '1d':
        z_data = exp1D(x,x_n,a=a,b=b,noise=c)
        X = poly_model_1d(x=x,poly_deg=p_d)
    elif case == '2d':
        z_data = exp2D(x,y,x_n,y_n,a=a,b=b,noise=c)
        X = poly_model2d(x=x,y=y,poly_deg=p_d)
    elif case == 'Franke':
        z = Franke(x,y,x_n,y_n,noise=c)
        X = poly_model2d(x=x,y=y,poly_deg=p_d)

    y_train_ridge[:,i],y_test_ridge[:,i],intcept_ridge[:,i],beta_tmp,mse_tmp,r2s_tmp = Ridge(y_data=z_data,X=X,lmbda=lmbda,
                                                                                                             intcept=intcept_ols,split=test_split,prnt=True)
    betas_ridge.append(beta_tmp); mse_s_ridge.append(mse_tmp); r2s_s_ridge.append(r2s_tmp)

print(len(betas_ridge))
print(betas_ridge[1])


## Plotting results for OLS-regression
#plot_OLS(poly_deg,y_data=[mse_s_ols,r2s_s_ols],x_label='poly. degree',y_labels=['MSE','R²'])
#beta_plot(poly_deg,betas_ols,x_label='poly. degree',y_label=r'$\hat{\beta}$')
#plot_reg1D(x_data=x,y_data=z_data,b_data=betas_ols,b0=intcept_ols)

## Plotting results for Ridge-regression 
#plot_RiLa(poly_deg,y_data=[mse_s_ridge,r2s_s_ridge],x_label='poly.deg',y_labels=['MSE','R²'],lmbda=lmbda)

#plt.show()





