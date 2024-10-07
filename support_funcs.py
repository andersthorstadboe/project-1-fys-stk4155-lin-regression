import numpy as np
import matplotlib.pyplot as plt
from imageio.v3 import imread

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
def poly_model_1d(x: np.ndarray, poly_deg: int, intcept=False):
   """
   Returning a design matrix for a polynomial of a given degree in one variable, x. 
   Includes the β0-column in building X, but this is taken out from the output by default.

   Parameters
   -------
   x : NDArray
      Dimension n x 1
   poly_deg : int
      degree of the resulting polynomial to be modelled, p
   intcept : bool
      If True, X is return with the β0-column
    
   Returns
   --------
   numpy.ndarray : Design matrix, X. Dimension: n x p or n x (p-1)
   """

   X = np.zeros((len(x),poly_deg+1))
   for p_d in range(poly_deg+1):
      X[:,p_d] = x[:,0]**p_d

   if intcept == True:
      return X
   else:
      return X[:,1:]

def poly_model_2d(x: np.ndarray, y: np.ndarray, poly_deg: int, intcept=False):
   """ From lecture notes
   Returning a design matrix for a polynomial of a given degree in two variables, x, y.
   Includes the β0-column in building X, but this is taken out from the output by default.
   
   Parameters
   ---
   x, y : NDArray
      Mesh in x and y direction
   poly_deg : int
      degree of the resulting polynomial to be modelled, p
   intcept : bool
      If True, X is return with the β0-column
   
   Returns
   ---
   numpy.ndarray : Design matrix, X. Dimension: n x (0.5*(p+2)(p+1))-1
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

   if intcept == True:
      return X
   else:
      return X[:,1:]

def SVDcalc(X):
   """
   Calculating the (X^T X)^-1 X^T - matrix product using a singular value decomposition, SVD
   of X, as X = U S V^T

   Parameters
   ---
   X : ndarray, n x p

   Returns
   ---
   NDArray : matrix product V (S^T S)^-1 S^T U^T 
   """
   U,S_tmp,V_T = np.linalg.svd(X)
   S = np.zeros((len(U),len(V_T)))
   S[:len(X[0,:]),:len(X[0,:])] = np.diag(S_tmp)
   STS = S.T @ S

   return V_T.T @ np.linalg.pinv(STS) @ S.T @ U.T

### Plotting functions
def plot_OLS(x_data=np.zeros(0),y_data: dict={},labels=['','','']):
   """
   Plotting an arbitrary number regression metrics aganist given x-axis data

   Parameters
   ---
   x_data : ndarray
      Data to plot regression metrics against
   y_data : dict
      Nested dicts of regression metrics from OLS-analysis
   labels : list
      0: title-label; 1: x_label; 2:end: y_labels, need to be as many as len(y_data) 

   Returns
   ---
   Figure is initialized, must be shown explicitly
   """
   key = list(y_data)
   ys = np.zeros((len(y_data),len(y_data[key[0]])))

   for i,f in enumerate(list(y_data)):
      for j,(l,s) in enumerate(y_data[f].items()):
         ys[i,j] = s

   i = 0; j = 0; k = 1
   while k <= len(ys):
      fig,ax = plt.subplots(1,1)
      ax.plot(x_data,ys[i],label='Training '+labels[j+2])
      ax.plot(x_data,ys[k],label='Test '+labels[j+2])
      ax.set_xlabel(labels[1]); ax.set_ylabel(labels[j+2],rotation=0,labelpad=15)
      ax.set_title(labels[j+2]+' against polynomial degree')
      ax.grid(); ax.legend()
      fig.tight_layout(pad=2.0,h_pad=1.5)
      
      i+=2; j+=1; k+=2
   
   return 0

def plot_heatmap(x_data: list, y_data: list, values: dict, labels: list, clrmap: str='bwr'):
   """
   Returns a heatmap based on x,y-axis data, and values from a dict, using the plt.imshow-function
   Made for plotting error-values based on λ-values and polynomial degree.
   Length of list(values) decides number of figures, length of x-data,y-data gives size of grid. 
   Length of list(list(values)) must correspond to this lengh

   Parameters
   ---
   x_data : list
      x-axis data, also used to set the xticks
   y_data : list
      y-axis data, also used to set yticks, but formatted as .e-values
   values : dict
      Dict of values to populate grid of heatmap.
   labels : list
      List of labels for plot. 0: suptitle; 1: cbar-label/axes-title; 2,3: x-,y-labels
   clrmap : str
      Default = 'bwr'. Matplotlib and similar colormap-names
     
   Returns
   ---
   Figure is initialized, must be shown explicitly
   """
   ## Creating heatmap 2d-array from values-dict
   f = list(values); vals = []; fig = []
   for i, p in enumerate(f):
      vals.append(np.zeros((len(y_data),len(x_data))))
      fig.append('fig'+str(i))
      for j,lst in enumerate(list(values[f[i]])):

         vals[i][:,j] = values[p][lst]

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

def beta_plot(x_data: np.ndarray,b_data: dict,labels: list=['','','']):
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
   f = list(b_data)
   fig,ax = plt.subplots(1,1)
   for i, l in enumerate(f):
      b = np.arange(1,len(b_data[l])+1)
      ax.plot(b,b_data[l],'--p')
   ax.set_xticks(np.arange(1,len(b)+1),labels=[f'${{\\beta_{{{x}}}}}$' for x in b])
   ax.set_xlabel(labels[1]); ax.set_ylabel(labels[2],rotation=0,labelpad=15)
   fig.suptitle(labels[0]+r' for $p =$ %i' %(int(x_data[i])))
   ax.grid()
   
   return 0

def plot1D(x_data, y_data, labels: list=['','','','',''], save=False, f_name: str='generic name.png'):
   """
   Returns a surface plot of 2D-data functions

   Parameters
   ---
   x_data : NDArray
      np.ndarray of x-axis data created with np.meshgrid(x,y)
   y_data : NDArray
      np.ndarray of y-axis data created with np.meshgrid(x,y)
   z_data : NDArray
      np.ndarray to be plotted on (x_data,y_data)-gird
   labels : list
      List of figure labels. 0: axes-title; 1,2,3: x-,y-, z-axis labels
   save : bool
      Default = False. Saves figure to current folder if True
   f_name : str
      file-name, including file-extension

   Returns
   ---
   Figure is initialized, must be shown explicitly

   """
   if save == True:
      plt.rcParams["font.size"] = 10
      fig,ax = plt.subplots(1,1,figsize=(3.5,(5*3/4)))
   else:
      fig,ax = plt.subplots(1,1)
   # Plotting initial data
   for i in range(len(y_data)):
      if i <= 0:
         ax.scatter(x_data,y_data[i],label=labels[3+i])
      else:
         ax.plot(x_data,y_data[i],label=labels[3+i])

   ax.set_title(labels[0]) 
   ax.set_xlabel(labels[1]); ax.set_ylabel(labels[2])
   ax.legend(); ax.grid()
   if save == True:
      fig.savefig(f_name,dpi=300,bbox_inches='tight')

   return 0

def plot2D(x_data, y_data, z_data, labels: list=['','','',''], save=False, f_name: str='generic name.png'):
   """
   Returns a surface plot of 2D-data functions

   Parameters
   ---
   x_data : NDArray
      np.ndarray of x-axis data created with np.meshgrid(x,y)
   y_data : NDArray
      np.ndarray of y-axis data created with np.meshgrid(x,y)
   z_data : NDArray
      np.ndarray to be plotted on (x_data,y_data)-gird
   labels : list
      List of figure labels. 0: axes-title; 1,2,3: x-,y-, z-axis labels
   save : bool
      Default = False. Saves figure to current folder if True
   f_name : str
      file-name, including file-extension

   Returns
   ---
   Figure is initialized, must be shown explicitly

   """
   if save == True:
      plt.rcParams["font.size"] = 10
      fig = plt.figure(figsize=(4.5,(5*3/4)))
   else:
      fig = plt.figure()
   # Plotting initial data
   ax = fig.add_subplot(111,projection='3d')
   f1 = ax.plot_surface(x_data,y_data,z_data,cmap='viridis')
   ax.set_aspect(aspect='auto')
   ax.view_init(elev=25, azim=-30)
   ax.set_title(labels[0]); ax.set_xlabel(labels[1])
   ax.set_ylabel(labels[2]); ax.set_zlabel(labels[3])
   ax.tick_params(axis='both', which='major', labelsize=6)
   fig.tight_layout()
   if save == True:
      fig.savefig(f_name,dpi=300,bbox_inches='tight')

   return 0

def imageData(f_name):
   f_folder = '01-data/'
   f_path = [f_folder+f_name[0]+'.tif',f_folder+f_name[1]+'.tif']
   n = 1                                                           # Initial downsampling of .tif-data
   z_data = []
   for i in range(len(f_path)):
      z_data.append(imread(f_path[i])[::n,::n])
   print('Original shape, z: ',z_data[0].shape)
   n = int(input('Scale down dataset '))
   # Downsampling, new downsamplig parameter, n
   z_data = [z_data[0][::n,::n],z_data[1][::n,::n]]
   Nx,Ny = len(z_data[0]),len(z_data[0][1])
   x = np.linspace(0,Nx,Nx)
   y = np.linspace(0,Ny,Ny)
   xx,yy = np.meshgrid(x,y,indexing='ij')

   ## Choosing with dataset to go forward with
   print('Number of images in z_data-list: %i' %(len(z_data)))
   case = int(input('Choose image, input an image number, starting at 1: '))
   try:
      z_data = z_data[case-1]
   except IndexError:
      print('Image number out-of-range, try again')
      print('Number of images in z_data-list: %i' %(len(z_data)))
      case = int(input('New image number: '))
      z_data = z_data[case-1]

   return z_data, xx, yy, Nx, Ny, case