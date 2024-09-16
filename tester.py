import numpy as np
import matplotlib.pyplot as plt
from support_funcs import poly_model_1d, exp1D, exp2D, poly_model2d, Franke
from reg_functions import RegOLS
from mpl_toolkits.mplot3d import Axes3D

r = {}
s = {}
a = 1 + 2
b = 1

r['a'] = a; r['b'] = b
s['c'] = b + a; s['d'] = a*2
#print(r)
print(list(r),list(s))
q = {}
q['r'] = r; q['s'] = s
print(list(q['r']))
print(type(q['r']))
print(len(q['r']))
#print(r['a'])




'''
mse = []
r2 = []

for i in range(5):
    a = np.random.uniform(0,2,10)
    b = np.random.uniform(0,2,5)
    mse.append(a); r2.append(b)

c = [mse,r2]

print(len(c[0]))

c2 = []

d = [c2.append(c[0][i]) for i in range(len(c[0]))]

print(d)













np.random.seed(2081)

b, c = 1.5, 0.02

x0,xN = 0, 1
y0,yN = 0, 1
N = 100
x = np.sort(np.random.uniform(x0,xN,N))#np.linspace(x0,xN,N)
y = np.sort(np.random.uniform(y0,yN,N))
x_n = np.random.normal(0, c, x.shape)#np.random.randn(N)
y_n = np.random.normal(0, c, x.shape)#np.random.randn(N)

noise_x,noise_y = np.meshgrid(x_n,y_n)
xx,yy = np.meshgrid(x,y)

z = Franke(xx,yy,noise_x,noise_y,noise=0.0)
z_1 = Franke(xx,yy,x_n,y_n,noise=0.5)

#y = x**2 + 10 + np.random.normal(0, c, x.shape)
#y = np.exp(-x**2) + b*np.exp(-(x-2)**2) + np.random.normal(0, c, x.shape)
#y = exp1D(x,x_n,1.0,1.5,0.0)
#Y = exp2D(xx,yy,noise_x,noise_y,1.0,1.5,0.01)

# Plotting initial data
fig = plt.figure(figsize=(12,7))
ax = fig.add_subplot(121,projection='3d')
bx = fig.add_subplot(122,projection='3d')
f1 = ax.plot_surface(xx,yy,z,cmap='bwr')
f2 = bx.plot_surface(xx,yy,z_1,cmap='bwr')
fig.suptitle('Test function')
ax.set_title('No noise'); bx.set_title('Noise'); plt.show()

#X = poly_model_1d(x,5)
X = poly_model2d(x,y,8)
#X = np.eye(len(x),3)
#print(X)

y_p_train, y_p_test, betas = OLS(y_data=y,X=X,split=True,scaling=True)

y_p = betas[0]
for i in range(0,len(betas[1])):
    y_p += betas[1][i]*x**[i+1]

fig,ax = plt.subplots(1,1)
ax.scatter(x,y)
ax.plot(x,y_p,'r')
plt.show()
#'''












































'''
def create_design_matrix(x, y, p):
    # Ensure x and y are numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)
    
    # List to hold each column for the design matrix
    columns = []
    
    # Generate polynomial terms up to degree p
    for i in range(1,p + 1):
        for j in range(p - 1 + i):
            print('x^%g * y^%g' %(i,j))
            columns.append(x**(i) * y**(j))
    
    # Stack columns to create the design matrix
    X = np.column_stack(columns)
    
    return X


def create_design_matrix(x, y, p):
    # Ensure x and y are numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)
    
    # List to hold each column for the design matrix
    columns = []
    
    # Generate polynomial terms up to degree p
    for i in range(p + 1):
        for j in range(p + 1 - i):
            print('x^%g * y^%g' %(i,j))
            columns.append(x**i * y**j)
    
    # Stack columns to create the design matrix
    X = np.column_stack(columns)
    
    return X

#x = np.linspace(0,1,5)#
x = np.array([1, 2, 3])
#y = np.linspace(0,1,5)
y = np.array([2, 3, 4])

# Degree of the polynomial
p = 2

# Generate the design matrix
X = poly_model_2d(x, y, p)
#X = create_design_matrix(x,y,p)

# Output the design matrix
print(X)
















x, y = np.linspace(0,1,5),np.linspace(0,1,5)

poly_deg = 2

cols = []

for p_dx in range(1,poly_deg+1):
    for p_dy in range(1,(poly_deg+1 - p_dx)):
        cols.append(x**p_dx * y**p_dy)
        print(cols[p_dx-1])
   

X = np.column_stack(cols)
print(X)

degree = int(((p_d + 2) * (p_d + 1))/(2))
xx = np.zeros((len(x),p_d+1)); yy = np.zeros(p_d+1)
l = []
for i in range(1,len(xx)+1):
    print(i)
    l.append('x^%g' %(i))
    
    xx[:,i-1] = x**i

print(l)
print(xx)

for p_dx in range(1,p_d+1):
    for p_dy in range(1,p_d+1 - p_dx):
        print('px: ',p_dx-1)
        print('py: ',p_dy-1)
        print('x^%g y^%g' %((p_dx-1),(p_dy)))
        '''