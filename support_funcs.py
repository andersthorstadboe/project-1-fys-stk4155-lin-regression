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
    x : numpy.ndarray
        Dimension n\n
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

def poly_model2d(x: np.ndarray, y: np.ndarray, poly_deg: int):
   """ From lecture notes
   Returning a design matrix for a polynomial of a given degree in\n
   two variables, x, y. Intercept column removed in final output, so that\n
   it is omitted from the design matrix\n
   
   Parameters
   ---
   x, y : np.ndarrays
      Dimensions n\n
   poly_deg : int
      degree of the resulting polynomial to be modelled, p
   
   Returns
   ---
   numpy.ndarray : Design matrix, X. Dimension: n x (0.5*(p+2)(p+1))
   """
   if len(x.shape) > 1:
      x = np.ravel(x); y = np.ravel(y)

   cols = int(((poly_deg + 2) * (poly_deg + 1))/2)
   X = np.ones((len(x),cols))
   cols = []                  # Storage for column vectors

   for p_dx in range(1, poly_deg+1):
      q = int((p_dx+1)*(p_dx)/2)
      for p_dy in range((p_dx + 1)):
         print('x^%g * y^%g' %((p_dx-p_dy),p_dy))
         X[:,q+p_dy] = (x**(p_dx-p_dy)) * (y**p_dy)
   
   #X = np.column_stack(cols)  # Stacking the columns into the design matrix
   #X = X[:,1:]                # Removing the intercept column

   return X

def SVDcalc(X):
   """
   Calculating the (X^T X)^-1 X^T - matrix product using a\n
   singular value decomposition, with X = U S V^T

   Parameters
   ---
   X : ndarray, n x p

   Returns
   ---
   ndarray : matrix product V (S^T S)^-1 S^T U^T 
   """
   U,S_tmp,V_T = np.linalg.svd(X)
   S = np.zeros((len(U),len(V_T)))
   S[:len(X[0,:]),:len(X[0,:])] = np.diag(S_tmp)
   STS = S.T @ S

   return V_T.T @ np.linalg.inv(STS) @ S.T @ U.T  #

### Plotting functions

def plot_OLS(x_data,y_data,x_label,y_labels):
   """
   Plotting an arbitrary number regression metrics aganist given x-axis\n 
   data

   Parameters
   ---
   x_data : ndarray
      x-axis data, length n
   y_data : list
      List of ndarrays of regression metrics
   x_label : string
      Label for x-axis, single value
   y_labels : list
      List of strings for labeling different y-axis, needs to match\n
      order of metrics in y_data

   Returns
   ---
   nothing : 0
   """
   
   print(len(y_data[0]))

   ax = []
   for i in range(len(y_data[0])):
      ax.append('ax' + str(i))
   
   fig,ax = plt.subplots(len(y_data),1)
   print(len(y_data))
   for i in range(len(y_data)):
      
      ax[i].plot(x_data,y_data[i][0],label='Training '+y_labels[i])
      ax[i].plot(x_data,y_data[i][1],label='Test '+y_labels[i])
      ax[i].set_xlabel(x_label); ax[i].set_ylabel(y_labels[i],rotation=0,labelpad=15)
      ax[i].set_title(y_labels[i]+' against polynomial degree')
      ax[i].grid(); ax[i].legend()

   fig.tight_layout(pad=2.0,h_pad=1.5)
   return 0

def plot_RiLa(x_data,y_data,x_label,y_labels,lmbda):

   fig,ax = [],[]
   print(len(y_data[0]))
   for i in range(len(y_data)):
      fig.append('fig' + str(i)); ax.append('ax' + str(i))
      fig[i],ax[i] = plt.subplots(1,1)
   
   for i in range(len(y_data)):
      for j in range(len(y_data[0])):   
         ax[i].plot(x_data,y_data[i][j][0],label='Training '+y_labels[i]+' λ = '+str(lmbda[j]))
         ax[i].plot(x_data,y_data[i][j][1],label='Test '+y_labels[i]+' λ = '+str(lmbda[j]))
         ax[i].set_xlabel(x_label); ax[i].set_ylabel(y_labels[i],rotation=0,labelpad=15)
         ax[i].set_title(y_labels[i]+' against polynomial degree')
         ax[i].grid(); ax[i].legend()
   return 0

#'''
def beta_plot(x_data,b_data,x_label,y_label):
   """
   Plotting regression parameters β against polynomial degree
   
   Parameters
   ---


   Returns
   ---
   nothing : 0
   
   """
   b = np.zeros(len(x_data))
   #fig,ax = 'fig'+str(i), 'ax'+str(i)
   fig,ax = plt.subplots(1,1)
   for i in range(len(b_data)):
      b[:i+1] = b_data[i]
      ax.plot(x_data,b,'--p',label='p = '+str(x_data[i]))


   ax.set_xlabel(x_label); ax.set_ylabel(y_label,rotation=0,labelpad=15)
   ax.set_title(r'$\hat{\beta}$ against polynomial degree')
   ax.grid(); ax.legend()

   return 0

def plot_reg1D(x_data,y_data,b_data,b0):
   """
   
   Parameters
   ---

   Returns
   ---
   nothing : 0
   """
   fig,ax = plt.subplots(1,1)
   ax.scatter(x_data,y_data)
   for i in range(len(b_data)):

      y_p = b0[i]
      for j in range(0,len(b_data[i])):
         y_p += b_data[i][j]*x_data**[j+1]

      ax.plot(x_data,y_p,label='p = '+str(len(b_data[i])))

   ax.set_xlabel('x'); ax.set_ylabel(r'$\tilde{y}$(x)',rotation=0,labelpad=15)
   ax.set_title('Regression model for p\'s'); ax.grid(); ax.legend()
   return 0
#'''
# ______________________# 

def poly_model_2d(x: np.ndarray, y: np.ndarray, poly_deg: int):
   """
   Returning a design matrix for a polynomial of a given degree in\n
   two variables, x, y. Intercept column removed in final output, so that\n
   it is omitted from the design matrix\n
   
   Parameters
   ---
   x, y : np.ndarrays
      Dimensions n\n
   poly_deg : int
      degree of the resulting polynomial to be modelled, p
   
   Returns
   ---
   numpy.ndarray : Design matrix, X. Dimension: n x (0.5*(p+2)(p+1))
   """
   cols = []                  # Storage for column vectors

   for p_dx in range(poly_deg+1):
      for p_dy in range((poly_deg + 1 - p_dx)):
         #print('x^%g * y^%g' %(p_dx,p_dy))
         cols.append(x**p_dx * y**p_dy)
   
   X = np.column_stack(cols)  # Stacking the columns into the design matrix
   X = X[:,1:]                # Removing the intercept column

   return X