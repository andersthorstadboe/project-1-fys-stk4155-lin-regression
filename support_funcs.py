import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

### Test functions
def Franke(x,y,x_noise,y_noise,noise=0.0):
   """
   Returns the Franke function on a mesh grid (x,y) as data function, with or without noise added
   
   Parameters
   --------
   x, y : array, n x m
      Mesh grid data points
   x_noise, y_noise : array, n x m
      Mesh grid noise data points
   noise : float
      Scale for the amount of noise added to the output, same on both axis x and y. Default = 0.0

   Returns
   --------
   ndarray : 2D-function representing the Franke function

   """
   p1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
   p2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
   p3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
   p4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)

   return p1 + p2 + p3 + p4 + noise*x_noise + noise*y_noise

def exp1D(x: np.ndarray, x_noise: np.ndarray,a: float, b: float, noise=0.0):
   """
   Returns a 1D exponential test function, "a*exp(-x²) + b*exp(-(x-2)²) + noise" for a ndarray x, with or without noise given as ndarray x_noise

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
   Returns a 2D exponential test function, for a mesh grid (x,y) with or without noise from mesh grid (x_noise, y_noise)

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
def poly_model_1d(x: np.ndarray, poly_deg: int):
   """
   Returning a design matrix for a polynomial of a given degree in one variable, x. Starts at i = 1, so intercept column is ommitted from design matrix.

   Parameters
   -------
   x : numpy.ndarray
      Dimension n x 1
   poly_deg : int
      degree of the resulting polynomial to be modelled, p
    
   Returns
   --------
   X : array, n x p 
      Design matrix of dimension n x p
   """

   X = np.zeros((len(x),poly_deg))
   for p_d in range(1,poly_deg+1):
      X[:,p_d-1] = x[:,0]**p_d
   
   return X

def poly_model_2d(x: np.ndarray, y: np.ndarray, poly_deg: int):
   """ From lecture notes
   Returning a design matrix for a polynomial of a given degree in two variables, x, y.\n
   Intercept column removed in final output, so that it is omitted from the design matrix\n
   
   Parameters
   ---
   x, y : ndarray
      Mesh in x and y direction
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

   for p_dx in range(poly_deg+1):
      
      q = int((p_dx+1)*(p_dx)/2)
      for p_dy in range((p_dx+1)):
         #print('q, p_dx, p_dy = ',q, p_dx, p_dy)
         #print('x^%g * y^%g' %((p_dx-p_dy),p_dy))
         X[:,q+p_dy] = (x**(p_dx-p_dy)) * (y**p_dy)

   #print(X)
   #print(X[:,1:])

   return X[:,1:]

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

   return V_T.T @ np.linalg.pinv(STS) @ S.T @ U.T

### Plotting functions
def plot_OLS(x_data=np.zeros(0),y_data=[],labels=['','','']):
   """
   Plotting an arbitrary number regression metrics aganist given x-axis data

   Parameters
   ---
   x_data : ndarray
      Data to plot regression metrix against
   y_data : list
      List of dicts of ndarrays of regression metrics
   labels : list
      0: title-label; 1: x_label; 2:end: y_labels, need to be as many as len(y_data) 

   Returns
   ---
   nothing : 0
   """

   ## Getting the dictionary keys and setting up lists
   list_vals = []; fig = []; ys1 = []; ys2 = []
   for i in range(len(y_data)):
      f = list(y_data[i])
      list_vals.append(f)
      fig.append('fig'+str(i))
      ys1.append([])
      ys2.append([])

   ## Unpacking dict-values
   for i, lst in enumerate(list_vals):
      for j, l in enumerate(lst):
         y = y_data[i][l]
         ys1[i].append(y[0])
         ys2[i].append(y[1])

   ## Plotting
   for i in range(len(y_data)):
      fig[i],ax = plt.subplots(1,1)
      fig[i].suptitle(labels[0]+'-regression')
      ax.plot(x_data,ys1[i],label='Training '+labels[i+2])
      ax.plot(x_data,ys2[i],label='Test '+labels[i+2])
      ax.set_xlabel(labels[1]); ax.set_ylabel(labels[i+2],rotation=0,labelpad=15)
      ax.set_title(labels[i+2]+' against polynomial degree')
      ax.grid(); ax.legend()

      fig[i].tight_layout(pad=2.0,h_pad=1.5)
   #'''
   return 0

def plot_RiLa(x_data,y_data,labels,lmbda):
   """
   Plotting an arbitrary number regression metrics aganist given x-axis data from Ridge/Lasso regression analysis
   
   Parameters
   ---
   x_data : ndarray
   ---   
   nothing : 0
   """
   ## Getting the dictionary keys and setting up lists
   list_vals = []; fig, ax, bx = [],[],[]; ys1 = []; ys2 = []
   #print(len(y_data[0]))
   for i in range(len(y_data)):
      f = list(y_data[i])
      #print(y_data[1][f[0]])
      list_vals.append(f)
      ax.append('ax'+str(i))
      ys1.append([])
      ys2.append([])
      
      fig.append('fig' + str(i)); ax.append('ax' + str(i))#; bx.append('bx' + str(i))
   
   ## Unpacking dict-values
   for i, lst in enumerate(list_vals):
      for j, l in enumerate(lst):
         y = y_data[i][l]
         ys1[i].append(y[0])
         ys2[i].append(y[1])
   
   print(len(ys1))
   print(ys1[0][0].shape)
   print(x_data.shape)
   
   for i in range(len(ys1)):
      fig[i],ax = plt.subplots(2,1)
      fig[i].suptitle(labels[0]+'-regression')
      for j in range(len(ys1[i])):   
         ax[0].plot(x_data,ys1[i][j],label='Training '+labels[i+2]+' λ = '+str(lmbda[j]))
         ax[1].plot(x_data,ys2[i][j],label='Test '+labels[i+2]+' λ = '+str(lmbda[j]))
         ax[0].set_xlabel(labels[1]); ax[0].set_ylabel(labels[i+2],rotation=0,labelpad=15)
         ax[0].set_title(labels[i+2]+' against polynomial degree')
         ax[0].grid(); ax[0].legend()
         ax[1].set_ylabel(labels[i+2],rotation=0,labelpad=15)
         #ax[1].set_title(labels[i+2]+' against polynomial degree')
         ax[1].grid(); ax[1].legend()

   return 0

def plot_heatmap(x_data: list,y_data: list, values: dict, labels: list, clrmap: str='bwr'):
   """
   
   """
   ## Creating heatmap 2d-array from values-dict
   f = list(values); vals = []; fig = []
   for i in range(len(values[f[0]])):
      vals.append(np.zeros((len(y_data),len(x_data))))
      fig.append('fig'+str(i))
      for j,lst in enumerate(f):
         vals[i][:,j] = values[lst][i]

   for i in range(len(vals)):
      fig[i],ax = plt.subplots(1,1,figsize=(5,4))
      im = ax.imshow(vals[i],cmap=clrmap)
      ax.set_aspect(aspect='auto',adjustable='box')

      ax.set_xticks(np.arange(len(x_data)),labels=x_data)
      ax.set_yticks(np.arange(len(y_data)),labels=[f'{y:.2e}' for y in y_data])

      cbar = ax.figure.colorbar(im, ax=ax, ticks=np.linspace(vals[i].min(), vals[i].max(), 10))
      cbar.ax.set_ylabel(labels[1], rotation=-90, va="bottom")
      ax.set_xlabel(labels[2]); ax.set_ylabel(labels[3],rotation=0)
      ax.set_title(labels[1]+' against polynomial degree')
      fig[i].suptitle(labels[0]+', '+labels[4+i]+'-data')
      fig[i].tight_layout()


   return 0

def beta_plot(x_data,b_data,labels):
   """
   Plotting regression parameters β against polynomial degree of prediction. Number of lines decided by length of b_data
   
   Parameters
   ---
   x_data : ndarray
      Data to plot β-values against
   b_data : list
      List of β-values from different regression runs
   labels : list
      0: title-label; 1: x_label; 2: y_label

   Returns
   ---
   nothing : 0
   """

   b = np.zeros(len(x_data))

   fig,ax = plt.subplots(1,1)
   for i in range(len(b_data)):
      b[:i+1] = b_data[i]
      ax.plot(x_data,b,'--p',label='p = '+str(x_data[i]))

   fig.suptitle(labels[0]+'-regression')
   ax.set_xlabel(labels[1]); ax.set_ylabel(labels[2],rotation=0,labelpad=15)
   ax.set_title(r'$\hat{\beta}$ against polynomial degree')
   ax.grid(); ax.legend()

   return 0

def plot_reg1D(x_data,y_data,b_data,b0,labels):
   """
   Plotting predictions against data from different regression analysis. Number of lines decided by the length of b_data

   Parameters
   ---
   x_data : ndarray
      x-axis mesh
   y_data : ndarray
      Data for plotting original data
   b_data : List 
      List of β-values from different regression runs
   b0 : ndarray
      Intercept values for OLS-regression
   labels : list
      0: title-label; 1: x_label; 2: y_label

   Returns
   ---
   nothing : 0
   """

   fig,ax = plt.subplots(1,1)
   fig.suptitle(labels[0]+'-regression')
   ax.scatter(x_data,y_data)
   for i in range(len(b_data)):

      y_p = b0[i]
      for j in range(0,len(b_data[i])):
         y_p += b_data[i][j]*x_data**[j+1]

      ax.plot(x_data,y_p,label='p = '+str(len(b_data[i])))

   ax.set_xlabel(labels[1]); ax.set_ylabel(labels[2],rotation=0,labelpad=15)
   ax.set_title('Regression model for p\'s'); ax.grid(); ax.legend()
   return 0

def plot2D(x_data, y_data, z_data,labels: list):
   """
   
   """
   #plt.rcParams["font.size"] = 5
   # Plotting initial data
   fig = plt.figure(figsize=(3.5,(5*3/4)))
   ax = fig.add_subplot(111,projection='3d')
   #bx = fig.add_subplot(122,projection='3d')
   f1 = ax.plot_surface(x_data,y_data,z_data,cmap='viridis')
   #f2 = bx.plot_surface(x_data,y_data,z_data,cmap='bwr')
   ax.view_init(elev=25, azim=-30)
   fig.suptitle(labels[0])
   ax.set_title(labels[1]); ax.set_xlabel(labels[2])#; bx.set_title('Noise'); plt.show()
   ax.set_ylabel(labels[3]); ax.set_zlabel(labels[4])
   ax.tick_params(axis='both', which='major', labelsize=4)
   #fig.savefig('franke-plot.png',dpi=300,bbox_inches='tight')

   return 0