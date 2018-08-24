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

def f1(k,point):
#calcule la dérivée première par rapport à x de f en point = [x0,y0]
Return k[0]+k[2]*point[1]+2*k[3]*point[0]+2*k[5]*point[0]*point[1]+k[6]*point[1]**2+3*k[7]*point[0]**2
def f2(k,point):
#calcule la dérivée première par rapport à y de f en point = [x0,y0]
return k[1]+k[2]*point[0]+2*k[4]*point[1]+k[5]*point[0]**2+2*k[6]*point[0]*point[1]+3*k[8]*point[1]**2
def f3(k,point):
#calcule la dérivée croisée de f en point = [x0,y0]
return k[2]+2*k[5]*point[0]+2*k[6]*point[1]
def f4(k,point):
#calcule la dérivée seconde par rapport à x de f en point = [x0,y0]
return 2*k[3]+2*k[5]*point[1]+6*k[7]*point[0]
def f5(k,point):
#calcule la dérivée seconde par rapport à y de f en point = [x0,y0]
return 2*k[4]+2*k[6]*point[0]+6*k[8]*point[1]

def systeme(x,k,point):
out=np.array([0.,0.])
out[0]=2*(x[0]-point[0])+2*(k[0]+k[2]*x[1]+2*k[3]*x[0]+2*k[5]*x[0]*x[1]+k[6]*x[1]**2+3*k[7]*x[0]**2)*(k[0]*x[0]+k[1]*x[1]
+k[2]*x[0]*x[1]+k[3]*x[0]**2+k[4]*x[1]**2+k[5]*x[0]**2*x[1]+k[6]*x[0]*x[1]**2+k[7]*x[0]**3+k[8]*x[1]**3+k[9]-point[2])
out[1]=(2*(x[1]-point[1])+2*(k[1]+k[2]*x[0]+2*k[4]*x[1]+2*k[6]*x[0]*x[1]+k[5]*x[0]**2+3*k[8]*x[1]**2)*(k[0]*x[0]+ k[1]*x[1]+k[2]*x[0]*x[1]+k[3]*x[0]**2+k[4]*x[1]**2+k[5]*x[0]**2*x[1]+k[6]*x[0]*x[1]**2+k[7]*x[0]**3+k[8]*x[1]**3+k[9]-point[2]))
return out

def courbure_gauss(k,point):
#calcule la courbure de gauss de la surface définie par z=f(x,y) (l'info f étant contenu dans l'array colonne k) en point
return (f5(k,point)*f4(k,point)-f3(k,point))/(1+f1(k,point)**2+f2(k,point)**2)**2

def courbure_moy(k,point):
#calcule la courbure moyenne de la surface définie par z=f(x,y) (l'info f étant contenu dans l'array colonne k) en point
a=(1+f1(k,point)**2)*f5(k,point)-2*f1(k,point)*f2(k,point)*f3(k,point)+(1+f2(k,point)**2)*f4(k,point)
b=2*(1+f1(k,point)**2+f2(k,point)**2)**3/2
return a/b
### données de bases
list_file_name=['200118A_inf_subsx3.csv','200118A_sup_subsx3.csv','200118B_inf_subsx3.csv','200118B_sup_subsx3.csv','200118C_inf_subsx3.csv','200118C_sup_subsx3.csv','200118C_inf_subsx3.csv','200118D_inf_subsx3.csv','200118D_sup_subsx3.csv','200118E_inf_subsx3.csv','200118E_sup_subsx3.csv','200118F_inf_subsx3.csv','200118F_sup_subsx3.csv','200118G_inf_subsx3.csv','200118G_sup_subsx3.csv','200118H_inf_subsx3.csv','200118H_sup_subsx3.csv','200118I_inf_subsx3.csv','200118I_sup_subsx3.csv']

for l in list_file_name:
data= pd.read_csv(+l, sep=";") #lecture
#gestion du dataframe
t_pos=[]
t_generation=[]
for i in range(len(data['nom_dent'])):
tooth_name=data['nom_dent'][i]
t_pos=t_pos+[[tooth_name[5:7]]]
t_generation=t_generation+[[tooth_name[-1]]]
np_t_pos=np.array(t_pos)
np_t_generation=np.array(t_generation)
add_data=np.concatenate((np_t_pos,np_t_generation), axis=1)
data=data.drop('nom_dent', axis=1)
add_data=pd.DataFrame(add_data,columns=['t_pos','t_generation'])
data=pd.concat([add_data,data], axis=1)
#début du calcul
data_brute = data.values[:,2:]
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

### Partie 2 : Projection des points sur la surface
coor_proj=np.zeros(np.shape(data_brute))
for i in range(np.shape(data_brute)[0]):
point=np.array(data_brute[i,:],dtype='float64')
p=fsolve(systeme, x0 = point[0:2], args=(K,point))
coor_proj[i,0],coor_proj[i,1]=p[0],p[1]
coor_proj[i,2]=K[0]*p[0]+K[1]*p[1]+K[2]*p[0]*p[1]+K[3]*p[0]**2+K[4]*p[1]**2+K[5]*p[0]**2*p[1]+K[6]*p[0]*p[1]**2+K[7]*p[0]**3+K[8]*p[1]**3+K[9]
datas=np.concatenate((data_brute,coor_proj), axis=1)
### Partie 3 : Calcul des courbures en tous points
curv_gauss=np.zeros([np.shape(datas)[0],1])
curv_moy=np.zeros([np.shape(datas)[0],1])
for i in range(np.shape(datas)[0]):
point=coor_proj[i,:]
curv_gauss[i]=courbure_gauss(K,point)
curv_moy[i]=courbure_moy(K,point)
data_finale=np.concatenate((datas,curv_gauss), axis=1)
data_finale=np.concatenate((data_finale,curv_moy), axis=1)

### Partie 4 : Export des données
data_finale=pd.DataFrame(data_finale, columns=['x','y','z','xproj','yproj','zproj','k_gauss','k_moy'])
result=pd.concat([data,data_finale],axis=1)
result=result.drop('xmoy',axis=1)
result=result.drop('ymoy',axis=1)
result=result.drop('zmoy',axis=1)
with open(l, 'w', newline='') as f:
writer = csv.writer(f)
result.to_csv(l,sep=',') 
