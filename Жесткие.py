# -*- coding: utf-8 -*-

"""
Created on Tue Feb 23 17:43:30 2021

@author: USER
"""
import math as m
n=int(input())


def c(k,n):
    q= m.factorial(n)/m.factorial(k)/m.factorial(n-k)
    return (q)
data=[]
def pod (a,k,p,q):
    x=0
    for i in range(a,k+1):
        x+=c(i,k)*(p1**i)*(q1**(k-i))
    return(x)
    
    
for i in range(n):
    d=str(input())
    d=d.split(' ')
    data.append(d)

for i in range(n):
    d=data[i]
    k=int(d[0])
    x=int(d[1])
    y= int(d[2])
    v=int(d[3])
    u=int(d[4])
    a=int(d[5])
    b=int(d[6])
    
    p1=(y-x+1)/37
    p2=(u-v+1)/37
    q1=(1-p1)
    q2=(1-p2)
    #X1=pod(a,3,p1,q1)
    X1=c(a,k)*(p1**a)*(q1**(k-a))
    print(X1)
#    X2=pod(b,2,p2,q2)
    X2=c(b,k)*(p2**a)*(q2**(k-a))
    
    print(X1*X2)
    
    