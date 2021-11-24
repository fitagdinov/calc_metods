# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 11:48:50 2020

@author: USER
"""

#import numpy as np
#import matplotlib.pyplot as plt
#import math
#fx=np.array([-0.229,-0.205,-0.077,0.159])
#x=np.array([0.5,0.6,0.8,1])
#
#
#def dif_fun(mas,x,n):
#    res=np.zeros(len(mas)-1)
#    for i in range (len(mas)-1):
#        res[i]=(mas[i+1]-mas[i])/(x[i+1+n]-x[i])
##        print(x[i+1+n],'n')
##        print(x[i],'l\n\n')
#    return(res)
# 
##def inter(dif,):
#    
##r1=dif_fun(x,fx,0)
##print(r1)
##r2=dif_fun(r1,fx,1)
##print(r2)
##r3=dif_fun(r2,fx,2)
##print(r3)
##
##x_0=x[0]+r1[0]*(0-fx[0])+r2[0]*(0-fx[0])*(0-fx[1])+r3[0]*(0-fx[0])*(0-fx[1])*(0-fx[2])
##print(x_0,'for')
##real=x_0**2-math.sin(x_0)
##x_l=x[-1]+r1[-1]*(0-fx[-1])+r2[-1]*(0-fx[-1])*(0-fx[-2])*r3[-1]*(0-fx[-1])*(0-fx[-2])*(0-fx[-3])
##print(x_l,'back')
##q=np.linspace(0,1,500)
##t=q**2-np.sin(q)
##plt.plot(q,t)
##plt.grid()
##print(real)
##r_b=real=x_l**2-math.sin(x_l)
##print(r_b)
#
#
#
#x1=np.array([0,0.1,0.2,0.3,0.4])
#f1=np.array([5,2.5,3,-2.5,-0.2])
#i=0
#data=[]
#data.append(f1)
#while i!=4:
#    res=dif_fun(data[-1],x1,i)
#    data.append(res)
#    i+=1
#print(data[1])
#def f(data,x):
#    dif=data[1][-1]+data[2][-1]*(x-x1[-1]+x-x1[-2])+data[3][-1]*((x-x1[-1])*(x-x1[-3]))+data[4][-1]*(x-x1[-1])*(x-x1[-3])*(x-x1[-4])
#    return(dif)
#print(f(data,0.3))
#h=0.1
#A=np.array([[1,-3*h,(-3*h)**2/2,(-3*h)**3/3,(-3*h)**4/4],
#            [1,-2*h,(-2*h)**2/2,(-2*h)**3/3,(-2*h)**4/4],
#            [1,-1*h,(-1*h)**2/2,(-1*h)**3/3,(-1*h)**4/4],
#            [1,0,0,0,0],
#            [1,h,(h)**2/2,(h)**3/3,(h)**4/4]])
##F=np.array([[0],[1],[0],[0],[0]])
#F=np.array([0,1,0,0,0])
#a=np.linalg.solve(np.transpose(A),F)
#print(a)
#print(f1)
#print(sum(a*f1))

def permutations(iterable, r=None):
    # permutations('ABCD', 2) --> AB AC AD BA BC BD CA CB CD DA DB DC
    # permutations(range(3)) --> 012 021 102 120 201 210
    pool = tuple(iterable)
    n = len(pool)
    r = n if r is None else r
    if r > n:
        return
    indices = list(range(n))
    cycles = list(range(n, n-r, -1))
    yield tuple(pool[i] for i in indices[:r])
    print(tuple(pool[i] for i in indices[:r]))
    while n:
        for i in reversed(range(r)):
            cycles[i] -= 1
            if cycles[i] == 0:
                indices[i:] = indices[i+1:] + indices[i:i+1]
                cycles[i] = n - i
            else:
                j = cycles[i]
                indices[i], indices[-j] = indices[-j], indices[i]
                yield tuple(pool[i] for i in indices[:r])
                print(tuple(pool[i] for i in indices[:r]))
                break
        else:
            return
        
a= permutations([1,2,3],2)
print(list(a))
