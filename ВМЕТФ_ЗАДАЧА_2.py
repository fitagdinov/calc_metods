# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 22:46:59 2021

@author: Robert
"""
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
a=0.001
b=0.0005
T_max=1000
dx=0.00001
dt=1
T_c=293

t_5=int(1/dt)-1
print(t_5)
def blat(dt,dx):
    x=np.arange(0,a+dx,dx)
    print(dt)
    t_5=int(3/dt)-1
    t=np.arange(0,T_max,dt)
    len_x=len(x)
    len_t=len(t)
    #print(x[len_x//2])
    T_a=T_c
    c_inside=1000
    p_inside=1000
    k_inside=0.1
    
    c_outside=500
    p_outside=8000
    k_outside=10
    q=5*(10**6)#3
    def koef(c,p,k):
        D=k/(c*p)
        mu=D*dt/(dx**2)
        return(mu)
    mu_inside=koef(c_inside,p_inside,k_inside)
    mu_outside=koef(c_outside,p_outside,k_outside)
    #print(mu_inside,mu_outside,"ffffffffffffffffffffffffff")
    def T_0(x):
    #    if x<b:
    #        return 4
    #    else:
    #        return 2
        return 293
    #    return x*T_c/a
    U=np.zeros(shape=(len_t,len_x))
    for i in range(len_t):
        U[i][len_x-1]=T_c
    
    for i in range(len_x):
        U[0][i]=T_0(x[i])
        
    def progon(A,B):# тренхдиоганальная матрица A, A*x=B. Цель найти решение.
        # порядок в матрице a,c,b
        # находим коэфиценты для прогонки. прямая прогонка
        f=-B
        n=len(A)
        alfa=np.zeros(n-1)
        beta=np.zeros(n)
        alfa[0]=-1*A[0][1]/A[0][0]
        beta[0]=-1*f[0]/A[0][0]
        for i in range(1,n):
            a=A[i][i-1]
            c=-1*A[i][i]
            if i!= n-1:
                b=A[i][i+1]
                alfa[i]=b/(c-a*alfa[i-1])
            beta[i]=(f[i]+a*beta[i-1])/(c-a*alfa[i-1])
            
        # получаем решение исходноего линейного уравнения 
        
    #    print(alfa)
    #    print(beta)
        #input()
        x=np.zeros(n)
        x[n-1]=beta[n-1]
        for i in range(n-2,-1,-1):
            #print(i)
            x[i]=alfa[i]*x[i+1]+beta[i]
        return (x)
    
    
    mu=np.ones(len_x)
    for i in range(len(mu)):
        if i<len_x/2:
            mu[i]=mu_inside*mu[i]
        else:
            mu[i]=mu_outside*mu[i]
    #print(mu)
    # задаенм трехдиаганальную матрицуисходя из начальных и краевых условий
    #для неявного метода
    matrix=np.zeros(shape=(len_x,len_x))
    matrix[0][0]=1
    matrix[0][1]=-1# из условия производной на левой границе
    matrix[-1][-1]=1 # T(t,a)=T_c
    for i in range(1,len_x-1):
        if i<len_x/2:
            matrix[i][i-1]=-mu_inside
            matrix[i][i]=(2*mu_inside+1)
            matrix[i][i+1]=-mu_inside
        else:
            matrix[i][i-1]=-mu_outside
            matrix[i][i]=(2*mu_outside+1)
            matrix[i][i+1]=-mu_outside
            
    #прописать разрыв
    cen=len_x//2
    matrix[cen][cen-1]=-k_inside
    matrix[cen][cen]=k_inside+k_outside
    matrix[cen][cen+1]=-k_outside
    #print(matrix)    
    for i in range(1,len_t):
        f=np.zeros(len_x)
        for j in range(1,len_x):
            if j<len_x/2:
                e=q*dt/c_inside/p_inside
            else:
                e=0
            f[j]=U[i-1][j]+e
        f[-1]=T_c
        f[cen]=0
        U[i]=progon(matrix,f)
    #    print(U[i])
    #    print(np.linalg.solve(matrix,f))
    #    input()
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection = '3d')
    #X,T= np.meshgrid(x,t)
    ##print(T.shape)
    ##print(U.shape)
    #surf = ax.plot_surface(T,X, U)
    #ax.set_xlabel("T")
    #ax.set_ylabel("X")
    #ax.set_zlabel("U")
    
    
    def true_solution():
        c1=0#(q*b/k_inside*(b-a)-q*b**2/2/k_inside)/a
        d1=-q*b/k_outside#*k_outside
        c2=d1*(b-a) + T_c + q*b*b/2/k_inside#a-b
    
        d2=T_c-d1*a#+d1*a/k_outside
        real=np.zeros(len_x)
    #    real=np.zeros_like(x)
        for i in range(len(x)):
            if x[i]<b:
    #            real[i]=-q*(x[i]**2)/2/k_inside + c1*x[i]+c2
                real[i]=-q*(x[i]**2)/2/k_inside + c1*x[i]+c2
            else:
                real[i]=d1*x[i]+d2#-d1
                
        return(real)
                
    real=true_solution()
    
    # K-L
    ###
    ###
    ###
    Z=np.zeros(shape=(len_t,len_x))
    for i in range(len_t):
        Z[i][len_x-1]=T_c
    
    for i in range(len_x):
        Z[0][i]=T_0(x[i])
        
        
        
    matrix=np.zeros(shape=(len_x,len_x))
    matrix[0][0]=1+mu_inside
    matrix[0][1]=-mu_inside# из условия производной на левой границе
    matrix[-1][-1]=1 # T(t,a)=T_c
    for i in range(1,len_x-1):
        if i<len_x/2:
            matrix[i][i-1]=-mu_inside/2
            matrix[i][i]=(mu_inside+1)
            matrix[i][i+1]=-mu_inside/2
        else:
            matrix[i][i-1]=-mu_outside/2
            matrix[i][i]=(mu_outside+1)
            matrix[i][i+1]=-mu_outside/2
    
    cen=len_x//2
    matrix[cen][cen-1]=-k_inside
    matrix[cen][cen]=k_inside+k_outside
    matrix[cen][cen+1]=-k_outside  
    
    
    for i in range(1,len_t):
        f=np.zeros(len_x)
        for j in range(1,len_x-1):
            if j< len_x/2:
                mu=mu_inside
                e=q*dt/c_inside/p_inside
            else:
                mu=mu_outside
                e=0
            
            f[j]=Z[i-1][j]*(1-mu)+Z[i-1][j+1]*mu/2+Z[i-1][j-1]*mu/2+e
        f[0]=e+Z[i-1][0]*(1-mu_inside)+mu_inside*Z[i-1][1]
        f[-1]=T_c
        f[cen]=0
        Z[i]=progon(matrix,f)
    #    print(U[i])
    #    print(np.linalg.solve(matrix,f))
    #    input()
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection = '3d')
    #X,T= np.meshgrid(x,t)
    ##print(T.shape)
    ##print(U.shape)
    #surf = ax.plot_surface(T,X,Z)
    #ax.set_xlabel("T")
    #ax.set_ylabel("X")
    #ax.set_zlabel("Z")
    
    
    
    
    #fig, axs= plt.subplots(4)
    #dt=dt/2
    #dx=dx/2
    #
    #
    #x=np.arange(0,a+dx,dx)
    #
    #t=np.arange(0,T_max,dt)
    #len_x=len(x)
    #len_t=len(t)
    #
    #def koef(c,p,k):
    #    D=k/(c*p)
    #    mu=D*dt/(dx**2)
    #    return(mu)
    #mu_inside=koef(c_inside,p_inside,k_inside)
    #mu_outside=koef(c_outside,p_outside,k_outside)
    ##print(mu_inside,mu_outside,"ffffffffffffffffffffffffff")
    #def T_0(x):
    ##    if x<b:
    ##        return 4
    ##    else:
    ##        return 2
    #    return T_c
    ##    return x*T_c/a
    #Q=np.zeros(shape=(len_t,len_x))
    #for i in range(len_t):
    #    Q[i][len_x-1]=T_c
    #
    #for i in range(len_x):
    #    Q[0][i]=T_0(x[i])
    #    
    #
    #
    #
    ## задаенм трехдиаганальную матрицуисходя из начальных и краевых условий
    ##для неявного метода
    #matrix=np.zeros(shape=(len_x,len_x))
    #matrix[0][0]=1
    #matrix[0][1]=-1# из условия производной на левой границе
    #matrix[-1][-1]=1 # T(t,a)=T_c
    #for i in range(1,len_x-1):
    #    if i<len_x/2:
    #        matrix[i][i-1]=-mu_inside
    #        matrix[i][i]=(2*mu_inside+1)
    #        matrix[i][i+1]=-mu_inside
    #    else:
    #        matrix[i][i-1]=-mu_outside
    #        matrix[i][i]=(2*mu_outside+1)
    #        matrix[i][i+1]=-mu_outside
    #        
    ##прописать разрыв
    #cen=len_x//2
    #matrix[cen][cen-1]=-k_inside
    #matrix[cen][cen]=k_inside+k_outside
    #matrix[cen][cen+1]=-k_outside
    ##print(matrix)    
    #for i in range(1,len_t):
    #    f=np.zeros(len_x)
    #    for j in range(1,len_x):
    #        if j<len_x/2:
    #            e=q*dt/c_inside/p_inside
    #        else:
    #            e=0
    #        f[j]=Q[i-1][j]+e
    #    f[-1]=T_c
    #    f[cen]=0
    #    Q[i]=progon(matrix,f)
    ##    print(U[i])
    ##    print(np.linalg.solve(matrix,f))
    ##    input()
    
    
    
#    fig=plt.figure()
#    plt.plot(x,real)
#    fig=plt.figure()
#    plt.plot(x,U[-1],'r',  x,Z[-1],'b', x,real,'g') #,x,Q[-1],'#000000',  x,real,'g')
#    plt.legend(['1','K-L','true'])
##    #
#    fig=plt.figure()
#    plt.plot(x,real-U[-1],'r',x,real-Z[-1],'b')#,x,Q[-1]-real,'#000000')
#    plt.legend(["1","K-L"])
    
#    print((real-U[t_5]).mean())
#    print((real-Z[t_5]).mean())
#    
#    print((real-U[-1]).max())
#    print((real-Z[-1]).max())
#    return((abs((real-U[-1])).mean(),(abs((real-Z[-1])).mean())))
    return(((np.power(real-U[-1],2).sum())**0.5,(np.power(real-Z[-1],2).sum())**0.5))
#    fig=plt.figure()
#    plt.plot(x,U[-1],'r',  x,Z[-1],'b', x,real,'g') #,x,Q[-1],'#000000',  x,real,'g')
#    plt.legend(['1','K-L','true'])
#print(U[-1][-1],Z[-1][-1],real[-1])
#    
#fig=plt.figure()
#plt.plot(x,U[-1]-Z[-1])
DT=[]
M_1=[]
M_2=[]
for i in np.arange(2,30,1):
    dt=i
    DT.append(dt)
    (m1,m2)=blat(dt,dx)
    M_1.append(m1)
    M_2.append(m2)
    print("flag")
plt.figure()
plt.plot(DT,M_1,'r',DT,M_2,"g")
    
    

