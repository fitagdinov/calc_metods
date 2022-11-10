# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 22:27:45 2021

@author: USER
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
dt=0.005
dx=0.01
a=-2
b=5
t_max=4
x_data= np.arange(a,b,dx)
t_data=np.arange(0,t_max,dt)
c=1


def fi_x(x,m=100,d=0.5):
    fi=np.exp(-(x/d)**m)
    return fi
fi=fi_x(x_data)
U= np.zeros((len(x_data),len(t_data)))
#U[0]=fi
for t in range(len(t_data)):
    U[0][t]=fi_x(a)
for x in range(len(x_data)):
    U[x][0]=fi_x(x_data)[x]
#fig = plt.figure()
#ax = fig.add_subplot(111, projection = '3d')
T,X = np.meshgrid(t_data,x_data)
#T=np.zeros_like(U)
#X=np.zeros_like(U)
#for i in range()

print("метод номер \n 1- треугольник\n 2- TVD \n 3- L \n 4-true sokution")
inp=int(input())

if inp==2 or inp==3:
    for t in range(len(t_data)):
        U[len(x_data)-1][t]=fi_x(b)

def true_triangular(U,x_data,t_data,c,dx,dt):
    for n in range(0,len(t_data)-1):
        for j in range(1,len(x_data)):
            U[j][n+1]=U[j][n]-c*dt/dx*(U[j][n]-U[j-1][n])
    return(U)
#plt.plot(t_data,fi_x(t_data))
    
def psi(q):
    res=max(0,min(1,2*q),min(2,q))
    return res

def f_plas(f_j,f_j1,r,q):
    f_p=f_j+(f_j1-f_j)*(1-r)/2*psi(q)
    return (f_p)

r=c*dt/dx    
    
def TVD():
    for n in range(0,len(t_data)-1):
        for j in range(1,len(x_data)-1):
            #print(j)
            q=(U[j][n]-U[j-1][n])/(U[j+1][n]-U[j][n])
            f_p=c*f_plas(U[j][n],U[j+1][n],r,q)
            f_n=c*f_plas(U[j-1][n],U[j][n],r,q)
            
            
            
            U[j][n+1]=-(f_p-f_n)/dx*dt+U[j][n]
    return (U)
    

def f_plas_L(f_j,f_j1,r):
    f_p=f_j+(f_j1-f_j)*(1-r)/2
    return (f_p)
def L():
    for n in range(0,len(t_data)-1):
        for j in range(1,len(x_data)-1):
            f_p=c*f_plas_L(U[j][n],U[j+1][n],r)
            f_n=c*f_plas_L(U[j-1][n],U[j][n],r)
            
            U[j][n+1]=-(f_p-f_n)/dx*dt+U[j][n]  
    return(U)
    
    
def true_solution():
    for j in range(0,len(x_data)):
        for n in range(0,len(t_data)):
            t=n*dt
            U[j][n]=fi_x(x_data-c*t*np.ones_like(x_data))[j]
    return (U)
if inp==1:
    U=true_triangular(U,x_data,t_data,c,dx,dt)
if inp==2:
    U=TVD()
if inp ==3:
    U=L()
if inp ==4:
    U=true_solution()
print(T.shape,'T_shape\n')
print(X.shape,'X_shape\n')
print(U.shape,'U_shape')
#surf = ax.plot_surface(T,X, U)
#ax.set_xlabel("T")
#ax.set_ylabel("X")
#ax.set_zlabel("U")



# plot 2D
fig, axs= plt.subplots(4)
t_point_index=len(t_data)//5
x_point_index=len(x_data)//4*3

# plot t=const

for j in range(1,5):
    x=[]
    y=[]
    T_point=(len(t_data)//4*j)-1
    for i in range(len(x_data)):
        x.append(x_data[i])
        y.append(U[i][T_point])
    
    axs[j-1].plot(x,y)    
    axs[j-1].set_title("t=const")
    axs[j-1].set(xlabel="x")
     

#fig, axs= plt.subplots(4)
fig=plt.figure()
Z=[]
for j in range(1,5):
    TV=0
    T_point=(len(t_data)//4*j)-1
    for i in range(len(x_data)-1):
        TV+=abs(U[i+1][T_point]-U[i][T_point])
    Z.append(TV)
#    axs[j-1].plot(Z)    
#    axs[j-1].set_title("t=range(1,4,1")
#    axs[j-1].set(xlabel="TV")

plt.plot(Z,'b')



# plot x=const
#x=[]
#y=[]
#
#for i in range(len(t_data)):
#    x.append(t_data[i])
#    y.append(U[x_point_index][i])
#
#axs[0].plot(x,y)    
#axs[0].set_title("x=const")
#axs[0].set(xlabel="t")
     

                 
fig2= plt.figure()
plt.plot(x_data,fi_x(x_data)) 

true_mat=np.zeros((len(x_data),len(t_data)))

for j in range(0,len(x_data)):
   for n in range(0,len(t_data)):
      t=n*dt
      true_mat[j][n]=fi_x(x_data-c*t*np.ones_like(x_data))[j]
        

t=t_point_index


def max_loss(t):
    loss=[]
    for i in range(len(x_data)):
        fake=U[i][t]
        true=true_mat[i][t]
        step_loss=abs(fake-true)
        loss.append(step_loss)
    return(max(loss))
    
def rms_loss(t):
    loss=[]
    for i in range(len(x_data)):
        fake=U[i][t]
        true=true_mat[i][t]
        step_loss=abs(fake-true)
        loss.append(step_loss**2)
    return((sum(loss)/len(x_data))**0.5)
        
    


print(max_loss(t),'max_loss')
print(rms_loss(t),'rms_loss')
