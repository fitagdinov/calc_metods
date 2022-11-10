# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 21:45:28 2021

@author: USER
"""

        

#import math as m
#n= int(input())
#l=set(str(input()).split())
#l=[int(x) for x in l]
#l=set(l)
#s=set()
#for i in l:
#    q=m.cos(i)
#    s.add(q)
#print(len(s))
  
def d(a,b):
    q=a*x0+b*x1
    w=a*y0 + b*y1
    return(q,w)
import numpy as np    
t= int(input())
for j in range(t):
    z=list(str(input()).split())
    z=[float(x) for x in z]
    [n,a0,b0,x0,x1,y0,y1]=z
    m=np.array([[x0,x1],[y0,y1]])
    n= int(n)
    w=np.linalg.matrix_power(m,n)
    q=a0*w[0][0]+b0*w[0][1]
    print(int(q%1000000007))
        
        
        