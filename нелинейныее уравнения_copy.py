# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 05:11:30 2020

@author: USER
"""

import numpy as np
import matplotlib.pyplot as plt
import math as m
print('HI\nwhat are you want')

### first
want=int(input())
if want==1:
    def f(x):
        f_=4*x**3-12*x**2+3*x-5
        return(f_)
    x=np.linspace(-3,5,1000)
    y=4*x**3-12*x**2+3*x-5
    #plt.plot(x,y)
    plt.grid()
    print('видно что решение где то х=2,8\n',
          'решаем x_{n+1}=x_n-a*f(x)=g(x)\n\n',
          '|g\'(x)|=|1-3*a+24*a*x*(x-1)|<1\n',
          'a=1/(5*161) подходит')
    a=1/(3*162)
    x_0=0
    x_=0
    iteration=[x_]
    num=[1]
    i=0
    print(f(0))
    while abs(f(x_))>0.001:
        i+=1
        x_=x_-a*f(x_)
        num.append(i)
        iteration.append(x_)
        print('num',x_,'nev',abs(f(x_)),'\n')
        
    #plt.figure()
    
    plt.plot(num,iteration,'.')

if want==2:
    print('sin(x+2)-y=1.5\nx+cos(y-2)=0.5')
    def f(x,y):
        f_1=m.sin(x+2)-y-1.5
        f_2=x+m.cos(y-2)-0.5
        A=np.array([[f_1],[f_2]])
        return(A)
    def revers_J(x,y):
        j=np.array([[m.cos(x+2),-1],[1,-m.sin(y-2)]])
        return(np.linalg.inv(j))
    x1=np.linspace(-10,10,1000)
    y1=np.sin(x1+2)-1.5
    
    y2=np.linspace(-10,10,1000)
    x2=-np.cos(y2-2)+0.5
    plt.plot(x1,y1,'r',x2,y2,'b')
    x_0=0.0
    y_0=0.0
    e=0.001
    x=[x_0]
    y=[y_0]
    f_0=f(x_0,y_0)
    print(f_0[0]>e)
    print(f_0[1]>e)
    while abs(f_0[0])>e or abs(f_0[1])>e:
        J=revers_J(x_0,y_0)
        f_0=f(x_0,y_0)
        print(f_0,'f')
        
        A=J@f_0
        x_0=x_0-A[0][0]
        y_0=y_0-A[1][0]
        x.append(x_0)
        y.append(y_0)
    print(f_0)
    plt.plot(x,y,'g')
        
        

    
    
    