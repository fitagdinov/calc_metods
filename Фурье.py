# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 20:28:05 2022

@author: USER
"""
import numpy as np
import matplotlib.pyplot as plt
import math as m


from scipy.fft import fft, ifft 

# def fft(F):
#     f= np.zeros_like(F)
#     for i in range(len(F)):
#         k=0
#         for j in range(1,len(F)-1):
#             k+=F[j]*m.sin(m.pi*i*j/len(F))
#         f[i]=k
#     return(f)

# def ifft(f):
#     F= np.zeros_like(f)
#     for i in range(1,len(F)-1):
#         k=0
#         for j in range(1,len(f)-1):
#             k+=f[j]*m.sin(m.pi*i*j/len(F))
#         F[i]=k*2/len(f)
#     return(F)

def sigma(x):
    return( (-1/(1-x**2)))
# def dif(t):
#     k1=((-2)/((1-t**2)**2))**2
#     k2=((-2)*(1-t**2)-8*t**2)/((1-t**2)**3)
#     res=m.exp(sigma(t))*(k1+k2)
#     return(res)

def dif(x):
    return(m.exp(sigma(x))*((2*x/((1-x**2)**2))+ (-2*(1-x**2)-8*x**2)/((1-x**2)**3)))

N=100
M=100
a=1
b=1
hx=a/(N-1)
hy=b/(M-1)
x=np.linspace(0, a, N)
y=np.linspace(0, b, M)

f=np.zeros((N,M))
for i in range(1,N-1):
    for j in range(1,M-1):
        f[i,j]=-4*dif(2*x[i]-1)*m.exp(sigma(2*y[j]-1))-4*dif(2*y[j]-1)*m.exp(sigma(2*x[i]-1))
# z=ifft(f[:,50])
# q=fft(z)

real=np.zeros((N,M))
for i in range(1,N-1):
    for j in range(1,M-1):
        real[i,j]=m.exp(sigma(2*x[i]-1))*m.exp(sigma(2*y[j]-1))
        
        
        
# a=m.pi
# b=m.pi
# hx=a/(N-1)
# hy=b/(M-1)
# x=np.linspace(0, a, N)
# y=np.linspace(0, b, M)
# f=np.zeros((N,M))
# for i in range(N):
#     for j in range(M):
#         f[i,j]=2*m.sin(x[i])*m.sin(y[j])
        
# real=np.zeros((N,M))
# for i in range(N):
#     for j in range(M):
#         real[i,j]=m.sin(x[i])*m.sin(y[j])



F1=np.zeros((N,M),dtype=complex)
for i in range(N):
    F1[i]=ifft(f[i])
F2=np.zeros((N,M),dtype=complex)    
for j in range(M):
    F2[:,j]=ifft(F1[:,j])
    
# print(F2[:,50])
V=np.zeros((N,M),dtype=complex)
for i in range(1,N-1):
    for j in range(1,M-1):
        g=4*(m.sin(m.pi*i/(2*N))**2)
        l=4*(m.sin(m.pi*j/(2*M))**2)
        V[i,j]=(g/hx/hx+l/hy/hy)**(-1)*F2[i,j]

F1=np.zeros((N,M),dtype=complex)
for j in range(M):
    F1[:,j]=fft(V[:,j])
F2=np.zeros((N,M))    
for i in range(N):
    F2[i]=fft(F1[i])
print(fft(F1[50]))
print(F2[0,50])
plt.plot(x,F2[:,50],'b+')
plt.plot(x,real[:,50],'r')
