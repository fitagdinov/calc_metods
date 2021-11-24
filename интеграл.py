# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 16:37:49 2020

@author: USER
"""

import numpy as np 
import matplotlib.pyplot as plt
import math as m
print('9.5g\n\n\n')

x=np.array([0,0.25,0.5,0.75,1,1.25,1.5,1.75,2])
f=np.array([1,0.979915,0.927295,0.858001,0.785398,0.716844,0.716844,0.716844,0.716844])
def trap(f,dx):
    I=0
    for i in range(len(f)-1):
        I+=(f[i+1]+f[i])*dx/2
    return(I)
print(trap(f,0.25))
Ih=trap(f,0.25)
f2=np.array([1,0.927295,0.785398,0.716844,0.716844])
I2h=trap(f2,0.5)

print(I2h,'I2h')
Is=Ih+(Ih-I2h)/3
print(Is,'IS')

print('9.10g\n\n\n\n')
x=np.linspace(0,m.pi/4,500)
y=np.sin(x**2)
plt.plot(x,y)
print(trap(y,x[1]))

print('9.13b\n\n\n\n')


