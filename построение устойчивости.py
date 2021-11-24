# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 13:18:06 2021

@author: USER
"""

import numpy as np
import matplotlib.pyplot as plt
import cmath
import math as m
a=np.array([[]])
b=np.array([])


#
#fi=np.linspace(0,3.14*2,100)
#j=complex(0,1)
#R=[]
#x=[]
#y=[]
#z=[]
#for i in range(0,100):
#    R.append(cmath.exp(fi[i]*j))
#plt.grid()
#for i in range(100):
#    z.append((5*R[i]-5)/(1+4*R[i]))
#    x.append(z[i].real)
#    y.append(z[i].imag)
#plt.plot(x,y)
#plt.grid(True)
if False:
    X_plot=[]
    Y_plot=[]
    f,ax=plt.subplots(1,2)
    
    x=np.linspace(-10,10,1000)
    y=np.linspace(-10,10,1000)
    for i in x:
        for j in y:
            z=complex(i,j)
            R=(1+m.sqrt(3)*z/3+(1+2*m.sqrt(3))/12*z*z)/((0.5-m.sqrt(3)/6)**2*z*z-(1-m.sqrt(3)/3)*z+1)
            if abs(R)<1:
                X_plot.append(z.real)
                Y_plot.append(z.imag)
    ax[0].plot(X_plot,Y_plot,'o')
    ax[0].grid(True)
    #plt.figure(1)
    r=np.linspace(-100,0,100)
    R_plot=[]
    for i in r:
        z=i
        R=(1+m.sqrt(3)*z/3+(1+2*m.sqrt(3))/12*z*z)/((0.5-m.sqrt(3)/6)**2*z*z-(1-m.sqrt(3)/3)*z+1)
        R_plot.append(R)
    ax[1].plot(r,R_plot)
    ax[1].grid(True)


if True:
    a=1
    tau=np.linspace(0.0001,1,1000)
    h=np.linspace(0.0001,1,1000)
    X=[]
    Y=[]
    j=complex(0,1)
    fi=np.linspace(0,2*3.14,100)
    for x in tau:
        for y in h:
            flag=True
            for i in fi:
                bracket_1=cmath.exp(j*i)+1
                bracket_2=cmath.exp(j*i)-1
                mem_1=1/(2*x)*bracket_1
                mem_2=1/(2*y)*bracket_2
                labda=abs((mem_1-mem_2)/(mem_1+mem_2))
                print(labda)
                if labda>1:
                    flag=False
                    break
            if flag:
                X.append(x)
                Y.append(y)
                #print(x,y)
    plt.plot(X,Y,'go')
                
        
        