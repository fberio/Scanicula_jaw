###Importation

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import fsolve
from mpl_toolkits.mplot3d import Axes3D
import csv

### fonctions

#k est un array contenant les coefficients de l'equation de la surface z=ax+by+cxy+dx²+ey²+fx²y+gy²x+hx²x+iy²y+k

def f(x,y,k):
    return k[0]*x+k[1]*y+k[2]*x*y+k[3]*x**2+k[4]*y**2+k[5]*x**2*y+k[6]*x*y**2+k[7]*x**3+ k[8]*y**3+k[9]

### données de bases

list_file_name=['200118A_inf_subsx3.csv','200118A_sup_subsx3.csv','200118B_inf_subsx3.csv','200118B_sup_subsx3.csv','200118C_inf_subsx3.csv','200118C_sup_subsx3.csv','200118C_inf_subsx3.csv','200118D_inf_subsx3.csv','200118D_sup_subsx3.csv','200118E_inf_subsx3.csv','200118E_sup_subsx3.csv','200118F_inf_subsx3.csv','200118F_sup_subsx3.csv','200118G_inf_subsx3.csv','200118G_sup_subsx3.csv','200118H_inf_subsx3.csv','200118H_sup_subsx3.csv','200118I_inf_subsx3.csv','200118I_sup_subsx3.csv']

for l in list_file_name:
    data= pd.read_csv('C:/Users/yddgr/Desktop/stage été 2018/S.canicula/input_python/csv_file/'+l, sep=";")
    data_brute = data.values[:,1:]
    n=np.shape(data_brute)[0]
    M=np.zeros([n,10])
    Z=np.zeros([n,1])
    
    ### Partie 1 : Regression non linéraire pour la surface
    
    for i in range(n):
        M[i,0]=data_brute[i,0]
        M[i,1]=data_brute[i,1]
        M[i,2]=data_brute[i,0]*data_brute[i,1]
        M[i,3]=data_brute[i,0]**2
        M[i,4]=data_brute[i,1]**2
        M[i,5]=data_brute[i,0]**2*data_brute[i,1]
        M[i,6]=data_brute[i,1]**2*data_brute[i,0]
        M[i,7]=data_brute[i,0]**3
        M[i,8]=data_brute[i,1]**3
        M[i,9]=1
        Z[i,0]=data_brute[i,2]
        
    #print(M),print(Z)
    
    M_tr=M.T
    M_car=np.dot(M_tr,M)
    M_inv=np.linalg.inv(M_car)
    #print('vérificaion de l inversion, ' ,np.dot(M_car,M_inv))
    K1=np.dot(M_inv,M_tr)
    K=np.dot(K1,Z)
    
    #vérif de la qualité de la régression = plot 3d du modèle
    fig=plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    xmin=np.min(data_brute[:,0])
    xmax=np.max(data_brute[:,0])
    ymin=np.min(data_brute[:,1])
    ymax=np.max(data_brute[:,1])
    
    
    x=np.linspace(xmin,xmax,100)
    y=np.linspace(ymin,ymax,100)
    x,y=np.meshgrid(x,y)
    z=f(x,y,K)
    
    ax.plot_surface(x,y,z, cmap='rainbow_r')
    ax.scatter(data_brute[:,0],data_brute[:,1],data_brute[:,2], c='k', marker= 'o')
    
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.title(l[0:11])
    
    plt.show()
    
    
    
    
