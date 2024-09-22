from support_funcs import poly_model_1d
import numpy as np
np.random.seed(2018)
x   = 2*np.ones(10).reshape(-1,1)#np.sort(np.random.uniform(0,1,10)).reshape(-1,1)
print(x)
y = 2*x**3 + x + np.random.normal(0,0.1,x.shape)

degree = 5

X = poly_model_1d(x,degree)

print(X)