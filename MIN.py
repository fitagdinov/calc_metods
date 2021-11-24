# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 09:50:54 2020

@author: USER
"""

import numpy as np
import matplotlib.pyplot as plt
import math as m
from mpl_toolkits.mplot3d import Axes3D
if 0==1:
    def f(x):
        return((x-5)*m.exp(x))
    def bar(a,b,d,n):
        return((b-a)/(2**n)+(1-1/(2**n))*d)
    
    x=np.linspace(-2,6,1000)
    y=(x-5)*np.exp(x)
    plt.plot(x,y,'r')
    plt.grid()
    a_0=-2
    b_0=6
    a=2
    b=6
    e=0.01
    n=0
    d=0.001
    datax=[(a+b)/2]
    datay=[f((a+b)/2)]
    while bar(a_0,b_0,d,n)>e:
        print(bar(a_0,b_0,d,n))
        t1=(a+b-d)/2
        t2=(a+b+d)/2
        n+=1
        if f(t1)<f(t2):
            a=a
            b=t2
        else:
            a=t1
            b=b
        datax.append((a+b)/2)
        datay.append(f((a+b)/2))
        
            
        
            
    print((a+b)/2,'+-',bar(a_0,b_0,d,n)/2)
    plt.plot(datax,datay,'og')
    
    
    print('\n\nf\'(x)=e^x*(x-4)=0')
    def g(x):
        return((x-4)*m.exp(x))
        
    def h(x):
        return((x-3)*m.exp(x))
        
    x_=7
    datax2=[x_]
    datay2=[f(x_)]
    while(abs(g(x_))>e/2):
        print(x_,'point')
        x_=x_-g(x_)/h(x_)
        datax2.append(x_)
        datay2.append(f(x_))
        
    print(x_)
    plt.plot(datax2,datay2,'+b')
    
else:
    def f(x,y):
        return(x**2+y**2+x*y+x-y+1)
        
    x=np.linspace(-2,2,1000)
    y=np.linspace(-2,2,1000)
    X, Y = np.meshgrid(x, y)

    Z=f(X,Y)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(X,Y,Z)
    x_0=0
    y_0=0

    def g_x(x,y):
        return(2*x+y+1)
    def g_y(x,y):
        return(2*y+x-1)
        
    def g_xx(x,y):
        return(2)
    def g_yy(x,y):
        return(2)
    def memory(datax,datay,dataz,x,y):
        datax.append(x)
        datay.append(y)
        dataz.append(f(x,y))
        
        
    datax=[x_0]
    datay=[y_0]
    dataz=[(f(x_0,y_0))]    
    while(abs(g_x(x_0,y_0))>0.001 or abs(g_y(x_0,y_0))>0.001):
        x_0=x_0-g_x(x_0,y_0)/g_xx(x_0,y_0)
        memory(datax,datay,dataz,x_0,y_0)
        y_0=y_0-g_y(x_0,y_0)/g_yy(x_0,y_0)
        memory(datax,datay,dataz,x_0,y_0)
    print(x_0,y_0,'result')
    ax.plot(datax,datay,dataz,'#ff0000ff')
    plt.figure()
    plt.plot(datax,datay)
    


         
            
