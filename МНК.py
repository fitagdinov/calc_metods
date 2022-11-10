# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 18:41:22 2020

@author: USER
"""

import matplotlib.pyplot as plt
import numpy as np

def gaus(b,n,f1):
    a=np.copy(b)
    f=np.copy(f1)
    for i in range(n-1):
        for j in range(i+1,min(i+3,n)):
            kof=a[j][i]/a[i][i]
            for k in range(i,n):
                a[j][k]=a[j][k]-kof*a[i][k]
            f[j][0]=f[j][0]-kof*f[i][0]
                
    return((a,f))

def back_gaus(a,n,f):
    u=np.zeros((n,1))
    u[n-1][0]=f[n-1][0]/a[n-1][n-1]
    for i in range(n-2,-1,-1):
        s=0
        for j in range(i+1,n):
            s+=a[i][j]*u[j][0]
        u[i][0]=(f[i][0]-s)/a[i][i]
    return(u)
    
def all_gaus(a,n,f):
    (q,w)=gaus(a,n,f)
    u=back_gaus(q,n,w)
    return(u)
    

def MNK(function,y,line=True):
    n=len(function)
    f=function
    A=np.ones((n,n))
    F=np.ones((n,1))
    for i in range(n):
        for j in range(n):
            A[i][j]=sum(f[i]*f[j])
        F[i][0]=sum(f[i]*y)
    kof=all_gaus(A,n,F)
    if line==True:
        new_line=np.zeros_like(y)
        for i in range(n):
            new_line=new_line+kof[i]*f[i]
        
        return(new_line)
    else:
        return(kof)
    
    
def max_defferent (line,y):
    deff=line-y
    return(max(deff))
    
def mean_defferent (line,y):
    deff=line-y
    return(np.mean(deff))
            
print('HI\nwhat are you want')

### first
want=int(input())
if want==1:
    x=np.linspace(0,1,100)
    y=x**2
    plt.plot(x,y)
    plt.grid()
    f1=x
    f2=np.ones(100)
    new_line=MNK([f1,f2],y)
    plt.plot(x,new_line)
    
    print('график в отдельном окне')

## second    
if want==2:
    x=np.arange(20,46)
    y=np.array([431,409,429,422,530,505,459,499,526,563,587,595,647,669,746,760,778,828,
                846,836,916,956,1014,1076,1134,1024])
    x1=x[0:9]
    y1=y[0:9]
    x2=x[8:20]
    y2=y[8:20]
    x3=x[19:]
    y3=y[19:]
    x_n=[x1,x2,x3]
    y_n=[y1,y2,y3]
    if 1==1:
        plt.figure()
        plt.plot(x,y,'.r')
        f1=x
        f2=np.ones_like(x)
        new_line=MNK([f1,f2],y)
        plt.plot(x,new_line,'b')
        plt.text(25,1100,'max_deff  '+str(max_defferent(new_line,y))+'\nmean   '+str(mean_defferent(new_line,y)))
    if 1==1:
        plt.figure()
        plt.plot(x,y,'.r')
        
        max_=[]
        mean=[]
        for i in range(3):
            f1=x_n[i]
            f2=np.ones_like(f1)
            new_line=MNK([f1,f2],y_n[i])
            color=['b','g','#000000ff']
            plt.plot(x_n[i],new_line,color[i])
            max_.append(max_defferent(new_line,y_n[i]))
            mean.append(mean_defferent(new_line,y_n[i]))
        plt.text(25,1100,'max_deff  '+str(max(max_))+'\nmean   '+str(np.mean(mean)))
            
    if 1==1:
        plt.figure()
        d1=np.array([1-(i-20)/(28-20) for i in x1])
        f1=np.concatenate((d1,np.zeros_like(x2[1:],dtype=float),np.zeros_like(x3[1:],dtype=float)),axis=None)
        
        d1=np.array([(i-20)/(28-20) for i in x1])
        d2=np.array([1-(i-28)/(39-28) for i in x2[1:]])
        f2=np.concatenate((d1,d2,np.zeros_like(x3[1:],dtype=float)),axis=None)
        
        d1=np.array([(i-28)/(39-28) for i in x2[1:]])
        d2=np.array([1-(i-39)/(45-39) for i in x3[1:]])
        f3=np.concatenate((np.zeros_like(x1,dtype=float),d1,d2),axis=None)
        
        d1=np.array([(i-39)/(45-39) for i in x3[1:]])
        f4=np.concatenate((np.zeros_like(x1),
                          np.zeros_like(x2[1:]),
                          d1),axis=None)
        
        new_line=MNK([f1,f2,f3,f4],y)
        
        plt.plot(x,new_line,'b')
        plt.plot(x,y,'.r')
        plt.text(25,1100,'max_deff  '+str(max_defferent(new_line,y))+'\nmean   '+str(mean_defferent(new_line,y)))
        
    print('график в отдельном окне')