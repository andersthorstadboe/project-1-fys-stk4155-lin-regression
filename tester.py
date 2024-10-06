import numpy as np
from support_funcs import *
from reg_functions import dataScaler
from imageio.v3 import imread
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, MaxAbsScaler

#print(plt.rcParams.keys)
#'''

# Grid and data setup
a, b   = 1.0, 1.5                                           # Coefficients for exponential model
c0, c1 = 0.01, 0.95                                         # Noise scaling    
x0, xN = 0, 1                                               # Start and end of domain, x-axis
y0, yN = 0, 0.95                                            # Start and end of domain, y-axis
Nx, Ny = 5, 5                                             # Number of sample points
np.random.seed(2018)
x   = np.sort(np.random.uniform(x0,xN,Nx)).reshape(-1,1)    # Mesh points on x-axis (uniformly distributed, sorted values)
y   = np.sort(np.random.uniform(y0,yN,Ny)).reshape(-1,1)    # Mesh points on y-axis (uniformly distributed, sorted values) (try different length arrays in x and y if singular values are an issue)
x_n = np.random.normal(0, c0, x.shape)                      # Noise for x-axis
y_n = np.random.normal(0, c0, y.shape)                      # Noise for y-axis
'''
a, b   = 1.0, 1.5                                           # Coefficients for exponential model
c0, c1 = 0.01, 0.95                                         # Noise scaling    
x0, xN = 0, 1                                               # Start and end of domain, x-axis
y0, yN = 0, 0.95                                            # Start and end of domain, y-axis
Nx, Ny = 50, 50                                             # Number of sample points

x   = np.sort(np.random.uniform(x0,xN,Nx)).reshape(-1,1)    # Mesh points on x-axis (uniformly distributed, sorted values)
y   = np.sort(np.random.uniform(y0,yN,Ny)).reshape(-1,1)    # Mesh points on y-axis (uniformly distributed, sorted values) (try different length arrays in x and y if singular values are an issue)
x_n = np.random.normal(0, c0, x.shape)                      # Noise for x-axis
y_n = np.random.normal(0, c0, y.shape)                      # Noise for y-axis
xx,yy = np.meshgrid(x,y,indexing='ij')
z = Franke(xx,yy,x_n,y_n,0.0)

print(xx.shape)
'''

# Test function selection
f_folder = '01-data/'
f_name = ['Etnedal','Jotunheimen']
f_path = [f_folder+f_name[0]+'.tif',f_folder+f_name[1]+'.tif']
n = 85
z = imread(f_path[0])[::n,::n]

Nx,Ny = len(z),len(z[0])
print(Nx,Ny)
x = np.linspace(0,Nx,Nx)
y = np.linspace(0,Ny,Ny)
xx,yy = np.meshgrid(x,y,indexing='ij')


xf,yf,zf = xx.reshape(-1,1), yy.reshape(-1,1),z.reshape(-1,1)
X = poly_model_2d(xf,yf,2)

scale = 'minmax'
std = StandardScaler()
Xs = dataScaler(X,scale,range=(1,2))
std.fit(X)
Xstd = std.transform(X)
zs = dataScaler(zf,scale,range=(1,2))
std_z = MaxAbsScaler()
std_z.fit(zf)
zstd = std_z.transform(zf)
print(std.mean_)
print(np.min(Xs))
print(np.max(Xs))
print(np.min(zs))
print(np.max(zs))

#XX = Xs.reshape(Nx,Ny)
zz = zs.reshape(Nx,Ny)
zzstd = zstd.reshape(Nx,Ny)

#fig,ax = plt.subplots(2,1)
#ax[0].plot()

plot2D(xx,yy,z)
plot2D(xx,yy,zz)
plot2D(xx,yy,zzstd)

plt.show()


'''
'''
print(Xstd[:10,:5])
print(Xs[:10,:5])
print(Xstd[:10,:5] - Xs[:10,:5])
scaler_STD_X = StandardScaler()
scaler_STD_z = StandardScaler()
scaler_minmax = MinMaxScaler()
sc_mmz = MinMaxScaler()
sc_rob = RobustScaler()
sc_robz = RobustScaler()
sc_pow = PowerTransformer()
sc_powz = PowerTransformer()

Xs = scaler_STD_X.fit(X)
zs = scaler_STD_z.fit(z)
scaler_minmax.fit(X)
sc_mmz.fit(z)
sc_rob.fit(X)
sc_robz.fit(z)
sc_pow.fit(X)
sc_powz.fit(z)

zz = zf.reshape(Nx,Ny)
XX = Xs.reshape(Nx,Ny)

print(scaler_STD_X.transform(X)[:,0])
print(scaler_minmax.transform(X)[:,0])
print(sc_rob.transform(X)[:,0])
print(sc_pow.transform(X)[:,0])
print()
print(scaler_STD_z.transform(z)[:,0])
print(sc_mmz.transform(z)[:,0])
print(sc_robz.transform(z)[:,0])
print(sc_powz.transform(z)[:,0])
#'''