# -*- coding: utf-8 -*-
"""
Created on Thu May  6 23:12:21 2021

@author: USER
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
from matplotlib.widgets import Slider,Button


def Simpl():
    U=np.zeros((len_t,len_x))

    U[0]=u_t0
    for i in range(len_t):
        U[i][0]=u_x0[i]
        U[i][-1]=u_x1[i]
        
    for n in range(0,len_t-1):
        for m in range(1,len_x-1):
            U[n+1][m]=U[n][m]+tau*((U[n][m+1]-2*U[n][m]+U[n][m+1])/(h**2))#+4*m*h/((m*h)**2+2*n*tau+1))#*(U[n][m+1]-U[n][m-1])/(2*h))
    return U

def t_2_h_2():
    U=np.zeros((len_t,len_x))

    U[0]=u_t0
    for i in range(len_t):
        U[i][0]=u_x0[i]
        U[i][-1]=u_x1[i]
    for m in range(1,len_x-1):    
        U[0+1][m]=U[0][m]+tau*((U[0][m+1]-2*U[0][m]+U[0][m+1])/(h**2)+4*m*h/((m*h)**2+2*0*tau+1))#*#(U[0][m+1]-U[0][m-1])/(2*h))
    
    for n in range(1,len_t-1):
        for m in range(1,len_x-1):
            U[n+1][m]=U[n][m]+1/(1/(2*tau)+1/(h**2))*((U[n][m+1]-U[n-1][m]+U[n][m+1])/(h**2))#+4*m*h/((m*h)**2+2*n*tau+1))#*(U[n][m+1]-U[n][m-1])/(2*h))
    return U
    
def t_2_h_2_ver2():
    U=np.zeros((len_t,len_x))

    U[0]=u_t0
    for i in range(len_t):
        U[i][0]=u_x0[i]
        U[i][-1]=u_x1[i]
    for m in range(1,len_x-1):    
        U[0+1][m]=U[0][m]+tau*((U[0][m+1]-2*U[0][m]+U[0][m+1])/(h**2)-4*(m*h)**2/(((m*h)**2+2*0*tau+1)**3))#*(U[0][m+1]-U[0][m-1])/(2*h))
    
    for n in range(1,len_t-1):
        for m in range(1,len_x-1):
            U[n+1][m]=U[n][m]+1/(1/(2*tau)+1/(h**2))*(((U[n][m+1]-U[n-1][m]+U[n][m+1])/(h**2))-4*(m*h)**2/(((m*h)**2+2*0*tau+1)**3))#*(U[n][m+1]-U[n][m-1])/(2*h))
    return U
    
#fig = plt.figure()
ax = plt.figure().add_subplot(projection='3d')
lim_t=1
tau=0.00001*5
lim_x=1
h=0.1*5
t=np.linspace(0,lim_t,lim_t/tau)
x=np.linspace(0,lim_x,lim_x/h)
len_t=len(t)
len_x=len(x)
u_t0=1/(x**2+1)
u_x0=1/(2*t+1)
u_x1=1/(2*t+2)
#
#U=np.zeros((len_t,len_x))
#
#U[0]=u_t0
##plt.plot(x,U[0],'+')
#for i in range(len_t):
#    U[i][0]=u_x0[i]
#    U[i][-1]=u_x1[i]
#    
#for n in range(0,len_t-1):
#    for m in range(1,len_x-1):
#        U[n+1][m]=U[n][m]+tau*((U[n][m+1]-2*U[n][m]+U[n][m+1])/(h**2)+4*m*h/((m*h)**2+2*n*tau+1)*(U[n][m+1]-U[n][m-1])/(2*h))
#
#U=t_2_h_2()
U=Simpl()
#U=t_2_h_2_ver2()
R, P = np.meshgrid(x, t)

ax.plot_surface(R, P, U, cmap=plt.cm.YlGnBu_r)
ax.set_xlabel('X')
ax.set_ylabel('T')
ax.set_zlabel('U')

plt.show()

#fig2=plt.figure()
#plt.plot(x,U[9])
#plt.plot(x,u_t0,'--')

pot_0=[]
pot_1=[]
pot_2=[]
#for i in range(len_t):
#    pot_0.append(U[i][0])
#    pot_1.append(U[i][1])
#    pot_2.append(U[i][2])
    
pot_0=U[0]
pot_1=U[1]
pot_2=U[2]
#plt.plot(np.zeros_like(t),t,pot_0,'g')
#plt.plot(np.zeros_like(t),t,pot_1,'b')
#plt.plot(np.zeros_like(t),t,pot_2,'y')
plt.plot(t,pot_0,'g')
plt.plot(t,pot_1,'b')
plt.plot(t,pot_2,'y')

#ax_time = plt.axes([0.25, 0.04, 0.5, 0.03])
#time_slider = Slider(
#    ax=ax_time,
#    label="TIME",
#    valmin=0,
#    valmax=lim_t/tau,
#    valinit=T,
#    valstep=1
#)
#def update(val):
#    time = time_slider.val
#    line.set_ydata(U[time])
#    print(time,'time')
#    print(U[time],'val')
#    fig.canvas.draw_idle()
#
#ax_reset = plt.axes([0.8, 0.025, 0.1, 0.04])
#button = Button(ax_reset, 'Reset', hovercolor='0.975')
#
#
#def reset(event):
#    time_slider.reset()
#    
#button.on_clicked(reset)
#time_slider.on_changed(update)
#plt.show()











        
