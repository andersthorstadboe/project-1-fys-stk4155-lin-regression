# Imports from supporting functions
from reg_functions import *
from support_funcs import *

# Import of relevant packages
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from imageio.v3 import imread
#from PIL import Image
#from tifffile import imread
from sklearn.model_selection import train_test_split

#Default for plots
plt.rcParams["figure.figsize"] = (15,7)

# Random seed
np.random.seed(2018)

## Data import
# Load the terrain
#z_data = imread('01-data/6701_1_10m_z33.tif')
#z_data = imread('01-data/SRTM_data_Norway_2.tif')
#z_data = imread('01-data/dtm1_33_119_124.tif')
#z_data = imread('01-data/dtm1_33_119_122.tif')
#dtm1_33_122_124
#img = Image.open('01-data/dom1_33_119_124.tif')[::,::100,::100]
#plt.imshow(img)
#plt.axis('off')
#plt.show()
#print(z_data.shape)

#f_name = ['01-data/dom10_6801_3_10m_z33.tif','01-data/dtm10_6801_3_10m_z33.tif']
#f_name = ['01-data/dom10_6702_4_10m_z33.tif','01-data/dtm10_6702_4_10m_z33.tif']
#f_name = ['01-data/dtm10_6702_4_10m_z33.tif','01-data/dtm10_6801_3_10m_z33.tif']
f_folder = '01-data/'
f_name = ['Etnedal','Jotunheimen']
#f_name = ['01-data/dtm1_33_120_123.tif','01-data/dtm1_33_113_126.tif']
#f_name = ['01-data/dtm1_33_114_126.tif','01-data/dtm1_33_114_127.tif']
#f_name = ['01-data/dtm1_33_120_123.tif','01-data/dtm1_33_120_124.tif']
#f_name = ['01-data/dtm1_33_121_123.tif','01-data/dtm1_33_121_124.tif']
f_path = [f_folder+f_name[0]+'.tif',f_folder+f_name[1]+'.tif']
# Show the terrain
fig,ax = plt.subplots(1,2)
fig.suptitle('Terrains')
j = 0
z_data = []
n = 20
for i, bx in enumerate(ax):
    
    z_data.append(imread(f_path[i])[::n,::n])#,imread(f_name[j+1])]
    print(z_data[i].shape)
    bx.imshow(z_data[i])

    bx.set_title(f_name[i]) #; cx.set_title(f_name[j+1])
    j += 3


print('Shape, z: ',z_data[0][::n,::n].shape)
Nx,Ny = len(z_data[0]),len(z_data[0][1])
print(Nx,Ny)
x = np.linspace(0,Nx,Nx)
y = np.linspace(0,Ny,Ny)
x, y = x[::n],y[::n]
z = [z_data[0][::n,::n],z_data[1][::n,::n]]
print('Shape, x: ',x.shape)
print('Shape, y: ',y.shape)
print('Shape, z0: ',z[0].shape)
print('Shape, z0: ',z[1].shape)
xx,yy = np.meshgrid(x,y,indexing='ij')

plot2D(xx,yy,z[0],labels=['',f_name[0],'','',''])
plot2D(xx,yy,z[1],labels=['',f_name[1],'','',''])
plt.show()

