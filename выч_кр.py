## -*- coding: utf-8 -*-
#"""
#Created on Mon Mar 15 16:31:50 2021
#
#@author: USER
#"""
#
#from mpl_toolkits.mplot3d import axes3d
#import matplotlib.pyplot as plt
#from matplotlib import cm
#import numpy as np
#lim_t=1
#tau=0.00001
#lim_x=1
#h=0.01
#t=np.linspace(0,lim_t,lim_t/tau)
#x=np.linspace(0,lim_x,lim_x/h)
#len_t=len(t)
#len_x=len(x)
#u_t0=1/(x**2+1)
#u_x0=1/(2*t+1)
#u_x1=1/(2*t+1)
#
#U=np.zeros((len_t,len_x))
#
#U[-1]=u_t0
#for i in range(len_t):
#    U[i][0]=u_x0[i]
#    U[i][-1]=u_x1[i]
#    
#for n in range(0,len_t-1):
#    for m in range(1,len_x-1):
#        U[n+1][m]=U[n][m]+tau*((U[n][m+1]-2*U[n][m]+U[n][m+1])/(h**2)+4*m*h/((m*h)**2+2*n*tau+1))#*(U[n][m+1]-U[n][m-1])/(2*h))
#
#X,Y=np.meshgrid(x, t)
#Z=U
#ax = plt.figure().add_subplot(projection='3d')
##X, Y, Z = axes3d.get_test_data(0.05)
#print(max(U[5]))
## Plot the 3D surface
#ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
#
## Plot projections of the contours for each dimension.  By choosing offsets
## that match the appropriate axes limits, the projected contours will sit on
## the 'walls' of the graph
#cset = ax.contour(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
#cset = ax.contour(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
#cset = ax.contour(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)
#
#ax.set_xlim(-1, 1)
#ax.set_ylim(0, 1)
#ax.set_zlim(-100, 100)
#
#ax.set_xlabel('X')
#ax.set_ylabel('Y')
#ax.set_zlabel('Z')
#
#plt.show()

