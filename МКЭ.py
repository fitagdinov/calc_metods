# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 17:53:00 2022

@author: USER
"""
import numpy as np
import matplotlib.pyplot as plt

n=4
k_list=(0.1,10)
phi=1000
f1=2000
g=20
num=1024

def basis_fun(n,num=64):
    x=np.linspace(0, 1, num+1)
    # h=1/n
    # down=np.zeros(num//n)
    
    if num % n != 0:
        print("ОСТОРОЖНО. ДЕЛЕНИЕ ПРОИЗВОДИТСЯ НЕ НА РАВНЫЕ УЧАСТКИ")
    N=np.zeros((n+1,num+1))
    N[-1][-1]=1
    for i in range(0,n):
        for j in range(num//n):
            N[i][i*(num//n) +j ]=N[i][i*(num//n) +j ]+ (((i+1)/n)-x[i*(num//n) +j])*n
            # print((((i+1)/h)-x[i*(num//n) +j])*h)
            # print(x[i*(num//n) +j],'\n')
    for i in range(1,n+1):
        for j in range(num//n):
            N[i][(i-1)*(num//n) +j ]=N[i][(i-1)*(num//n) +j ]+ ((-(i-1)/n)+x[(i-1)*(num//n) +j])*n       
    
    return (N)
                
# x=np.linspace(0, 1, 17)
# N= basis_fun(4,16)
# plt.plot(x,N[2])        
# plt.grid(True)        
    
    
def FEM(n,k_list,phi,f1,g):
    k1,k2=k_list
    A=np.zeros((n+1,n+1))
    A[0][0]=n*k1
    A[0][1]=-n*k1
    A[1][0]=-n*k1
    for i in range(1,n):
        if i< n/2:
            k=k1
        else:
            k=k2
            
        A[i][i+1]=-n*k
        A[i][i]=2*n*k
        A[i+1][i]=-n*k
    A[n//2][n//2]=(k1+k2)*n    # center 
    
    A[-1][-1]=n*k + phi # right 
    
    
    f=np.zeros((n+1))
    f[n//2]=1/2/n*f1
    f[0]=1/2/n*f1
    for i in range(1,n//2):
        f[i]=1/n*f1
    f[-1]=g
    alpha=np.linalg.solve(A,f)
    return(alpha)
    
N=basis_fun(n,num)
alpha=FEM(n,k_list,phi,f1,g)
x=np.linspace(0, 1, num+1)
y= np.zeros(num+1)
for i in range(n+1):
    y=y+N[i]*alpha[i]
plt.plot(x,y)

k1,k2=k_list
true_sol=np.zeros(num+1)
B1=-f1/2/k2
B2=(g+(phi+1)*f1/(2*k2))/phi#+0.1*f1/20+10/phi
C2=B1*0.5+B2+f1/2/k1*0.25
dif=C2-y[0]
for i in range(num+1):
    if x[i]<0.5:
        true_sol[i]=-f1/2/k1*(x[i]**2)+C2-dif
    else:
        true_sol[i]=B1*x[i]+B2-dif
plt.plot(x,true_sol,'r')
plt.legend([str(n)])
plt.grid(True)

for i in range(n+1):
    ind=i*num//n
    print(true_sol[ind]-y[ind],x[ind])
    