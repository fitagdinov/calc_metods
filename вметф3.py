# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 18:31:11 2021

@author: USER
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
a=1
b=0.25
T_max=1000000
dx=0.001
dt=10000
T_c=293
T_inf=300
sigma=5*(10**(-8))
c_inside=500
p_inside=8000
k_inside=2

c_outside=1000
p_outside=1000
k_outside=10
alpha=0
q=6*(10**4)#3

r=np.arange(dx,a+dx,dx)

t=np.arange(0,T_max,dt)
len_r=len(r)
len_t=len(t)
print(len_t,len_r)
cen=int(len_r*b/a)
print(cen)


plt.figure()
#точное решение
c1=-q*(b**3)/3
v1=-q/6/k_inside
c2=np.arange(-1000,1000,1)
poh=c1/(a**2)+alpha*(-c1/k_outside/a + c2 -T_inf) + sigma*(-c1/k_outside/a +c2)**4
plt.plot(c2,poh)
plt.grid(True)

s2=249.91
v2=-c1/k_outside/b+s2+q/6/k_inside*b*b
real=np.zeros_like(r)
for i in range(len_r):
    if r[i]<b:
        real[i]=v1*(r[i]**2)+v2
#        print(real[i])
    else:
        real[i]=-c1/k_outside/r[i] +s2
#        print(real[i])





def koef(c,p,k):
    D=k/(c*p)
#    mu=D*dt/(dx**2)
    return(D)
    

d1=koef(c_inside,p_inside,k_inside)
d2=koef(c_outside,p_outside,k_outside)
def progon(A,B):# тренхдиоганальная матрица A, A*x=B. Цель найти решение.
    # порядок в матрице a,c,b
    # находим коэфиценты для прогонки. прямая прогонка
    f=-B
    n=len(A)
    alfaq=np.zeros(n-1)
    beta=np.zeros(n)
    alfaq[0]=-1*A[0][1]/A[0][0]
    beta[0]=-1*f[0]/A[0][0]
    for i in range(1,n):
        a=A[i][i-1]
        c=-1*A[i][i]
        if i!= n-1:
            b=A[i][i+1]
            alfaq[i]=b/(c-a*alfaq[i-1])
        beta[i]=(f[i]+a*beta[i-1])/(c-a*alfaq[i-1])
        
    # получаем решение исходноего линейного уравнения 
    
#    print(alfa)
#    print(beta)
    #input()
    x=np.zeros(n)
    x[n-1]=beta[n-1]
    for i in range(n-2,-1,-1):
        #print(i)
        x[i]=alfaq[i]*x[i+1]+beta[i]
    return (x)    

d1=koef(c_inside,p_inside,k_inside)
d2=koef(c_outside,p_outside,k_outside)
#r=np.arange(dx,a*5+dx,dx)



U=np.zeros((len_t,len_r))
T_0=np.ones_like(r)*300
U[0]=T_0


matrix=np.zeros((len_r,len_r))
matrix[0][0]=-1
matrix[0][1]=1
for i in range(1,len_r-1):
    if r[i]<b:
        d=d1
    else:
        d=d2
    matrix[i][i-1]=d*dt/dx /r[i] - d*dt/(dx**2)
    matrix[i][i]=1+2*d*dt/(dx**2)
    matrix[i][i+1]=-d*dt/dx /r[i] - d*dt/(dx**2)
matrix[-1][-2]=-k_outside
matrix[-1][-1]=k_outside+alpha*dx

matrix[cen][cen-1]=-k_inside
matrix[cen][cen]=k_inside+k_outside
matrix[cen][cen+1]=-k_outside

for i in range(1,len_t):
    f=np.ones_like(r)
    for j in range(1,len_r-1):
        if r[j]<b:
            e=q/c_inside/p_inside
        else:
            e=0
        f[j]=e*dt+U[i-1][j]
    if i>1:
        f[-1]=T_inf*alpha*dx-sigma*((2*U[i-1][-1]-U[i-2][-1])**4) *dx   #interpolation
    else:
        f[-1]=T_inf*alpha*dx-sigma*((U[i-1][-1])**4)*dx
    f[0]=0
    f[cen]=0
    U[i]=progon(matrix,f)
    
plt.figure()
plt.plot(r,real,'r',r,U[-1],'g')
plt.legend(['real','apr'])
plt.grid(True)

plt.figure()
plt.plot(r,abs(real-U[-1]))
plt.grid(True)

#c1=-q*(b**3)/3
#d1=-q*(b**2)/6
#c2=np.arange(97,97.25,0.01)
#poh=c1/(a**2)+alpha-(-c1/k_outside/a + c2 -T_inf) + sigma*(-c1/k_outside/a +c2)**4
##plt.plot(c2,poh)
#plt.grid(True)
#
#s2=97.094
#d2=-c1/k_outside/b+s2+q/6/k_inside*b*b

