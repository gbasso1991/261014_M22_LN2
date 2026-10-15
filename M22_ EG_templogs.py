#%% Librerias y paquetes 
import numpy as np
from uncertainties import ufloat, unumpy
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import os
import chardet
import re
from datetime import datetime
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
pendiente_HvsI = 3716.3 # 1/m
ordenada_HvsI = 1297.0 # A/m
# from clase_resultados import ResultadosESAR
#%% Lector Templog
def lector_templog(path):
    '''
    Busca archivo *templog.csv en directorio especificado.
    muestras = False plotea solo T(dt).
    muestras = True plotea T(dt) con las muestras superpuestas
    Retorna arrys timestamp,temperatura
    '''
    data = pd.read_csv(path,sep=';',header=5,
                            names=('Timestamp','T_CH1','T_CH2'),usecols=(0,1,2),
                            decimal=',',engine='python')
    temp_CH1  = pd.Series(data['T_CH1']).to_numpy(dtype=float)
    temp_CH2  = pd.Series(data['T_CH2']).to_numpy(dtype=float)
    timestamp = np.array([datetime.strptime(date,'%Y/%m/%d %H:%M:%S') for date in data['Timestamp']])

    time = np.array([(t-timestamp[0]).total_seconds() for t in timestamp])
    return timestamp,time,temp_CH1, temp_CH2
#%% Curvatura
def curvatura(T):
    ancho_de_ventana=11
    dT=savgol_filter(T,window_length=ancho_de_ventana,polyorder=3,deriv=1,delta=1.0)
    dTT=savgol_filter(T,window_length=ancho_de_ventana,polyorder=3,deriv=2,delta=1.0)
    curv=np.abs(dTT)/(1+dT**2)**1.5
    return dT,dTT,curv
def expo(t,A,B,tau):
    return A + B*np.exp(-t/tau)
def biexpo(t,A,B1,tau1,B2,tau2):
    return A + B1*np.exp(-t/tau1) + B2*np.exp(-t/tau2)

#%% Imports 
nombre = '''M22
EG51% + FF49% --> LN2 --> RF
f = 300 kHz 
Idc = [15, 12.5, 10, 9 , 9.5, 8.5, 8, 7.5, 6] dA\n'''
print('-'*50,nombre)

temps_150 = glob("data_M22/*150*.csv",recursive=True)
temps_125 = glob("data_M22/*125*.csv",recursive=True)
temps_100 = glob("data_M22/*100*.csv",recursive=True)
temps_090 = glob("data_M22/*090*.csv",recursive=True)
temps_085 = glob("data_M22/*085*.csv",recursive=True)
temps_080 = glob("data_M22/*080*.csv",recursive=True)
temps_075 = glob("data_M22/*075*.csv",recursive=True)
temps_060 = glob("data_M22/*060*.csv",recursive=True)
#%% Ordeno y printeo
for t in [temps_150,temps_125,temps_100,temps_090,temps_085,temps_080,temps_075,temps_060]:
    t.sort()
    for tt in t:
        print('  -',os.path.basename(tt))

print('\nInformacion importante: ')
print('  CH1: Centro del criovial (temp_C)\n  CH2: Borde del criovial (temp_B)\n') 
Idc_values = [15, 12.5, 10, 9 , 9.5, 8.5, 8, 7.5, 6]
H0=[(h*pendiente_HvsI+ordenada_HvsI)/1000 for h in Idc_values] 
print('Idc (dA)  -->  H0 (kA/m)')
for i,e in enumerate(Idc_values):
    print(f'{e:4.1f} dA  -->  {H0[i]:3.1f} kA/m')

C = ['C0','C1','C2','C3','C4','C5','C6'] 
t,t_min=[],[]
T,T_min=[],[]
Indx_min,dT=[],[]

#%% 1 idc = 150 dA 
titulo_1 = 'M22 - EG51% + FF49% --> LN2 --> 300 kHz & 150 dA'
fig11,(ax1,ax2) = plt.subplots(2,1,figsize=(9,6),sharex=True,constrained_layout=True)

_,time_1,temp_C_1, temp_B_1 = lector_templog(temps_150[0])
delta_CB_1 = temp_C_1 - temp_B_1
indx_max_DT_1 = np.nonzero(delta_CB_1==max(delta_CB_1))[0]
ax1.plot(time_1,temp_C_1,'C0-',label='Centro')#
ax1.plot(time_1,temp_B_1,'C0-.',label='Borde')
ax1.vlines(time_1[indx_max_DT_1],temp_B_1[indx_max_DT_1],temp_C_1[indx_max_DT_1],color='C0',ls='--',label=f'ΔT max = {delta_CB_1[indx_max_DT_1][0]:.1f} °C')

_,time_2,temp_C_2, temp_B_2 = lector_templog(temps_150[1])
delta_CB_2 = temp_C_2 - temp_B_2
indx_max_DT_2 = np.nonzero(delta_CB_2==max(delta_CB_2))[0]
ax2.plot(time_2,temp_C_2,'C1-',label='Centro')#
ax2.plot(time_2,temp_B_2,'C1-.',label='Borde')
ax2.vlines(time_2[indx_max_DT_2],temp_B_2[indx_max_DT_2],temp_C_2[indx_max_DT_2],color='C1',ls='--',label=f'ΔT max = {delta_CB_2[indx_max_DT_2][0]:.1f} °C')

ax2.set_xlim(0,210)

for a in (ax1,ax2):
    a.grid()
    a.legend(title=f'f = 300 kHz   H$_0$ = {H0[0]:.0f} kA/m',
             loc='upper center',shadow=True,frameon=True,ncol=2)
    a.set_ylabel('T (°C)')
ax2.set_xlabel('t (s)')
plt.suptitle(titulo_1)

#% Comparativa e/ centro y borde, diferencia de temperatura
fig12,(ax,ax2,ax3) = plt.subplots(3,1,figsize=(9,9),sharex=True,constrained_layout=True)
ax.set_title('Centro',loc='left')
ax2.set_title('Borde',loc='left')
ax3.set_title(r'$\Delta$T',loc='left')

for i,p in enumerate(temps_150):
    _,time,temp_C, temp_B = lector_templog(p)
    delta_CB = temp_C - temp_B
    ax.plot(time,temp_C,'-',label=i+1)
    ax2.plot(time,temp_B,'-.',label=i+1)
    ax3.plot(time,delta_CB,'-',label=i+1)
    indx_min=np.nonzero(temp_C==min(temp_C))[0]
    print(f'Temp minima = {temp_C[np.nonzero(temp_C==min(temp_C))][0]:.1f} C ({temp_C[np.nonzero(temp_C==min(temp_C))][0]+273:.1f} K) alcanzada en {time[np.nonzero(temp_C==min(temp_C))][0]:.1f} s')

for a in (ax,ax2,ax3):
    a.grid()
    a.legend(title=f'f = 300 kHz   H$_0$ = {H0[0]:.0f} kA/m',
             loc='upper center',shadow=True,frameon=True,ncol=2)
    a.set_ylabel('T (°C)')
ax2.set_xlabel('t (s)')
ax2.set_xlim(0,210)
plt.suptitle(titulo_1)

#%% 2 125 dA = 53.3 kA/m
fig21,(ax1,ax2) = plt.subplots(2,1,figsize=(9,6),sharex=True,constrained_layout=True)
_,time_1,temp_B_1, temp_C_1 = lector_templog(temps_125[0])
delta_CB_1 = temp_C_1 - temp_B_1
indx_max_DT_1 = np.nonzero(delta_CB_1==max(delta_CB_1))[0]
ax1.plot(time_1,temp_C_1,'C0-',label='Centro')#
ax1.plot(time_1,temp_B_1,'C0-.',label='Borde')
ax1.vlines(time_1[indx_max_DT_1],temp_B_1[indx_max_DT_1],temp_C_1[indx_max_DT_1],color='C0',ls='--',label=f'ΔT max = {delta_CB_1[indx_max_DT_1][0]:.1f} °C')

_,time_2,temp_B_2, temp_C_2 = lector_templog(temps_125[1])
delta_CB_2 = temp_C_2 - temp_B_2
indx_max_DT_2 = np.nonzero(delta_CB_2==max(delta_CB_2))[0]
ax2.plot(time_2,temp_C_2,'C1-',label='Centro')#
ax2.plot(time_2,temp_B_2,'C1-.',label='Borde')
ax2.vlines(time_2[indx_max_DT_2],temp_B_2[indx_max_DT_2],temp_C_2[indx_max_DT_2],color='C1',ls='--',label=f'ΔT max = {delta_CB_2[indx_max_DT_2][0]:.1f} °C')

ax2.set_xlim(0,175)

for a in (ax1,ax2):
    a.grid()
    a.legend(title=f'f = 300 kHz   H$_0$ = {H0[1]:.0f} kA/m',loc='lower right',shadow=True,frameon=True,ncol=2)
    a.set_ylabel('T (°C)')

ax2.set_xlabel('t (s)')
plt.suptitle('EG51% + FF49% --> LN2 --> 300 kHz & 125 dA')

#% Comparativa e/ centro y borde, diferencia de temperatura
fig22,(ax,ax2,ax3) = plt.subplots(3,1,figsize=(9,9),sharex=True,constrained_layout=True)
ax.set_title('Centro',loc='left')
ax2.set_title('Borde',loc='left')
ax3.set_title(r'$\Delta$T',loc='left')

for i,p in enumerate(temps_125):
    _,time,temp_B, temp_C = lector_templog(p)
    delta_CB = temp_C - temp_B
    ax.plot(time,temp_C,'-',label=i+1)
    ax2.plot(time,temp_B,'-.',label=i+1)
    ax3.plot(time,delta_CB,'-',label=i+1)
    indx_min=np.nonzero(temp_C==min(temp_C))[0]
    print(f'Temp minima = {temp_C[np.nonzero(temp_C==min(temp_C))][0]:.1f} C ({temp_C[np.nonzero(temp_C==min(temp_C))][0]+273:.1f} K) alcanzada en {time[np.nonzero(temp_C==min(temp_C))][0]:.1f} s')

for a in (ax,ax2,ax3):
    a.grid()
    a.legend(title=f'f = 300 kHz   H$_0$ = {H0[1]:.0f} kA/m',loc='best',shadow=True,frameon=True,ncol=2)
    a.set_ylabel('T (°C)')
ax2.set_xlabel('t (s)')
ax2.set_xlim(0,175)
plt.suptitle('EG51% + FF49% --> LN2 --> 300 kHz & 125 dA')

#%%3 075 dA = 49.6 kA/m 
fig31,(ax1,ax2) = plt.subplots(2,1,figsize=(9,6),sharex=True,constrained_layout=True)
_,time_1,temp_B_1, temp_C_1 = lector_templog(temps_075[0])
delta_CB_1 = temp_C_1 - temp_B_1
indx_max_DT_1 = np.nonzero(delta_CB_1==max(delta_CB_1))[0]
ax1.plot(time_1,temp_C_1,'C0-',label='Centro')#
ax1.plot(time_1,temp_B_1,'C0-.',label='Borde')
ax1.vlines(time_1[indx_max_DT_1],temp_B_1[indx_max_DT_1],temp_C_1[indx_max_DT_1],color='C0',ls='--',label=f'ΔT max = {delta_CB_1[indx_max_DT_1][0]:.1f} °C')

_,time_2,temp_B_2, temp_C_2 = lector_templog(temps_075[1])
delta_CB_2 = temp_C_2 - temp_B_2
indx_max_DT_2 = np.nonzero(delta_CB_2==max(delta_CB_2))[0]
ax2.plot(time_2,temp_C_2,'C1-',label='Centro')#
ax2.plot(time_2,temp_B_2,'C1-.',label='Borde')
ax2.vlines(time_2[indx_max_DT_2],temp_B_2[indx_max_DT_2],temp_C_2[indx_max_DT_2],color='C1',ls='--',label=f'ΔT max = {delta_CB_2[indx_max_DT_2][0]:.1f} °C')

ax2.set_xlim(0,200)

for a in (ax1,ax2):
    a.grid()
    a.legend(title=f'f = 300 kHz   H$_0$ = {H0[2]:.0f} kA/m',loc='lower right',shadow=True,frameon=True,ncol=2)
    a.set_ylabel('T (°C)')
ax2.set_xlabel('t (s)')
plt.suptitle('EG51% + FF49% --> LN2 --> 300 kHz & 075 dA')

#% Comparativa e/ centro y borde, diferencia de temperatura
fig32,(ax,ax2,ax3) = plt.subplots(3,1,figsize=(9,9),sharex=True,constrained_layout=True)
ax.set_title('Centro',loc='left')
ax2.set_title('Borde',loc='left')
ax3.set_title(r'$\Delta$T',loc='left')

for i,p in enumerate(temps_075):
    _,time,temp_B, temp_C = lector_templog(p)
    delta_CB = temp_C - temp_B
    ax.plot(time,temp_C,'-',label=i+1)
    ax2.plot(time,temp_B,'-.',label=i+1)
    ax3.plot(time,delta_CB,'-',label=i+1)
    indx_min=np.nonzero(temp_C==min(temp_C))[0]
    print(f'Temp minima = {temp_C[np.nonzero(temp_C==min(temp_C))][0]:.1f} C ({temp_C[np.nonzero(temp_C==min(temp_C))][0]+273:.1f} K) alcanzada en {time[np.nonzero(temp_C==min(temp_C))][0]:.1f} s')

for a in (ax,ax2,ax3):
    a.grid()
    a.legend(title=f'f = 300 kHz   H$_0$ = {H0[2]:.0f} kA/m',loc='best',shadow=True,frameon=True,ncol=2)
    a.set_ylabel('T (°C)')
ax3.set_xlabel('t (s)')
ax3.set_xlim(0,200)
plt.suptitle('EG51% + FF49% --> LN2 --> 300 kHz & 075 dA')

#%%4 120 dA = 45.9 kA/m
fig41,(ax1,ax2) = plt.subplots(2,1,figsize=(9,6),sharex=True,constrained_layout=True)
_,time_1,temp_B_1, temp_C_1 = lector_templog(temps_120[0])
delta_CB_1 = temp_C_1 - temp_B_1
indx_max_DT_1 = np.nonzero(delta_CB_1==max(delta_CB_1))[0]
ax1.plot(time_1,temp_C_1,'C0-',label='Centro')#
ax1.plot(time_1,temp_B_1,'C0-.',label='Borde')
ax1.vlines(time_1[indx_max_DT_1],temp_B_1[indx_max_DT_1],temp_C_1[indx_max_DT_1],color='C0',ls='--',label=f'ΔT max = {delta_CB_1[indx_max_DT_1][0]:.1f} °C')

_,time_2,temp_B_2, temp_C_2 = lector_templog(temps_120[1])
delta_CB_2 = temp_C_2 - temp_B_2
indx_max_DT_2 = np.nonzero(delta_CB_2==max(delta_CB_2))[0]
ax2.plot(time_2,temp_C_2,'C1-',label='Centro')#
ax2.plot(time_2,temp_B_2,'C1-.',label='Borde')
ax2.vlines(time_2[indx_max_DT_2],temp_B_2[indx_max_DT_2],temp_C_2[indx_max_DT_2],color='C1',ls='--',label=f'ΔT max = {delta_CB_2[indx_max_DT_2][0]:.1f} °C')

ax2.set_xlim(0,200)

for a in (ax1,ax2):
    a.grid()
    a.legend(title=f'f = 300 kHz   H$_0$ = {H0[3]:.0f} kA/m',loc='lower right',shadow=True,frameon=True,ncol=2)
    a.set_ylabel('T (°C)')
ax2.set_xlabel('t (s)')
plt.suptitle('EG51% + FF49% --> LN2 --> 300 kHz & 120 dA')

#% Comparativa e/ centro y borde, diferencia de temperatura
fig42,(ax,ax2,ax3) = plt.subplots(3,1,figsize=(9,9),sharex=True,constrained_layout=True)
ax.set_title('Centro',loc='left')
ax2.set_title('Borde',loc='left')
ax3.set_title(r'$\Delta$T',loc='left')

for i,p in enumerate(temps_120):
    _,time,temp_B, temp_C = lector_templog(p)
    delta_CB = temp_C - temp_B
    ax.plot(time,temp_C,'-',label=i+1)
    ax2.plot(time,temp_B,'-',label=i+1)
    ax3.plot(time,delta_CB,'-',label=i+1)
    indx_min=np.nonzero(temp_C==min(temp_C))[0]
    print(f'Temp minima = {temp_C[np.nonzero(temp_C==min(temp_C))][0]:.1f} C ({temp_C[np.nonzero(temp_C==min(temp_C))][0]+273:.1f} K) alcanzada en {time[np.nonzero(temp_C==min(temp_C))][0]:.1f} s')

for a in (ax,ax2,ax3):
    a.grid()
    a.legend(title=f'f = 300 kHz   H$_0$ = {H0[3]:.0f} kA/m',loc='best',shadow=True,frameon=True,ncol=2)
    a.set_ylabel('T (°C)')
ax2.set_xlabel('t (s)')
ax2.set_xlim(0,200)
plt.suptitle('EG51% + FF49% --> LN2 --> 300 kHz & 120 dA')

#%%5 110 dA = 42.2 kA/m
fig51,(ax1,ax2) = plt.subplots(2,1,figsize=(9,6),sharex=True,constrained_layout=True)
_,time_1,temp_B_1, temp_C_1 = lector_templog(temps_110[0])
delta_CB_1 = temp_C_1 - temp_B_1
indx_max_DT_1 = np.nonzero(delta_CB_1==max(delta_CB_1))[0]
ax1.plot(time_1,temp_C_1,'C0-',label='Centro')#
ax1.plot(time_1,temp_B_1,'C0-.',label='Borde')
ax1.vlines(time_1[indx_max_DT_1],temp_B_1[indx_max_DT_1],temp_C_1[indx_max_DT_1],color='C0',ls='--',label=f'ΔT max = {delta_CB_1[indx_max_DT_1][0]:.1f} °C')

_,time_2,temp_B_2, temp_C_2 = lector_templog(temps_110[1])
delta_CB_2 = temp_C_2 - temp_B_2
indx_max_DT_2 = np.nonzero(delta_CB_2==max(delta_CB_2))[0]
ax2.plot(time_2,temp_C_2,'C1-',label='Centro')#
ax2.plot(time_2,temp_B_2,'C1-.',label='Borde')
ax2.vlines(time_2[indx_max_DT_2],temp_B_2[indx_max_DT_2],temp_C_2[indx_max_DT_2],color='C1',ls='--',label=f'ΔT max = {delta_CB_2[indx_max_DT_2][0]:.1f} °C')

ax2.set_xlim(0,200)

for a in (ax1,ax2):
    a.grid()
    a.legend(title=f'f = 300 kHz   H$_0$ = {H0[4]:.0f} kA/m',loc='lower right',shadow=True,frameon=True,ncol=2)
    a.set_ylabel('T (°C)')
ax2.set_xlabel('t (s)')
plt.suptitle('EG51% + FF49% --> LN2 --> 300 kHz & 110 dA')

#% Comparativa e/ centro y borde, diferencia de temperatura
fig52,(ax,ax2,ax3) = plt.subplots(3,1,figsize=(9,9),sharex=True,constrained_layout=True)
ax.set_title('Centro',loc='left')
ax2.set_title('Borde',loc='left')
ax3.set_title(r'$\Delta$T',loc='left')

for i,p in enumerate(temps_110):
    _,time,temp_B, temp_C = lector_templog(p)
    delta_CB = temp_C - temp_B
    ax.plot(time,temp_C,'-',label=i+1)
    ax2.plot(time,temp_B,'-',label=i+1)
    ax3.plot(time,delta_CB,'-',label=i+1)
    indx_min=np.nonzero(temp_C==min(temp_C))[0]
    print(f'Temp minima = {temp_C[np.nonzero(temp_C==min(temp_C))][0]:.1f} C ({temp_C[np.nonzero(temp_C==min(temp_C))][0]+273:.1f} K) alcanzada en {time[np.nonzero(temp_C==min(temp_C))][0]:.1f} s')

for a in (ax,ax2,ax3):
    a.grid()
    a.legend(title=f'f = 300 kHz   H$_0$ = {H0[4]:.0f} kA/m',loc='best',shadow=True,frameon=True,ncol=2)
    a.set_ylabel('T (°C)')
ax3.set_xlabel('t (s)')
ax3.set_xlim(0,200)
plt.suptitle('EG51% + FF49% --> LN2 --> 300 kHz & 110 dA')
#%%6 100 dA = 38.5 kA/m
fig61,(ax1,ax2,ax3,ax4) = plt.subplots(4,1,figsize=(9,12),sharex=True,constrained_layout=True)
_,time_1,temp_B_1, temp_C_1 = lector_templog(temps_100[0])
delta_CB_1 = temp_C_1 - temp_B_1
indx_max_DT_1 = np.nonzero(delta_CB_1==max(delta_CB_1))[0]
ax1.plot(time_1,temp_C_1,'C0-',label='Centro')#
ax1.plot(time_1,temp_B_1,'C0-.',label='Borde')
ax1.vlines(time_1[indx_max_DT_1],temp_B_1[indx_max_DT_1],temp_C_1[indx_max_DT_1],color='C0',ls='--',label=f'ΔT max = {delta_CB_1[indx_max_DT_1][0]:.1f} °C')

_,time_2,temp_B_2, temp_C_2 = lector_templog(temps_100[1])
delta_CB_2 = temp_C_2 - temp_B_2
indx_min_T = np.nonzero(temp_C_2==min(temp_C_2))[0][0]
delta_CB_2_aux = delta_CB_2[:indx_min_T]
indx_max_DT_2 = np.nonzero(delta_CB_2_aux==max(delta_CB_2_aux))[0]
ax2.plot(time_2,temp_C_2,'C1-',label='Centro')#
ax2.plot(time_2,temp_B_2,'C1-.',label='Borde')  
ax2.vlines(time_2[indx_max_DT_2],temp_B_2[indx_max_DT_2],temp_C_2[indx_max_DT_2],color='C1',ls='--',label=f'ΔT max = {delta_CB_2[indx_max_DT_2][0]:.1f} °C')

_,time_3,temp_B_3, temp_C_3 = lector_templog(temps_100[2])
delta_CB_3 = temp_C_3 - temp_B_3
indx_max_DT_3 = np.nonzero(delta_CB_3==max(delta_CB_3))[0]
ax3.plot(time_3,temp_C_3,'C2-',label='Centro')#
ax3.plot(time_3,temp_B_3,'C2-.',label='Borde')
ax3.vlines(time_3[indx_max_DT_3],temp_B_3[indx_max_DT_3],temp_C_3[indx_max_DT_3],color='C2',ls='--',label=f'ΔT max = {delta_CB_3[indx_max_DT_3][0]:.1f} °C')

_,time_4,temp_B_4, temp_C_4 = lector_templog(temps_100[3])
delta_CB_4 = temp_C_4 - temp_B_4
indx_max_DT_4 = np.nonzero(delta_CB_4==max(delta_CB_4))[0]
ax4.plot(time_4,temp_C_4,'C3-',label='Centro')#
ax4.plot(time_4,temp_B_4,'C3-.',label='Borde')
ax4.vlines(time_4[indx_max_DT_4],temp_B_4[indx_max_DT_4],temp_C_4[indx_max_DT_4],color='C3',ls='--',label=f'ΔT max = {delta_CB_4[indx_max_DT_4][0]:.1f} °C')

ax4.set_xlim(0,200)

for a in (ax1,ax2,ax3,ax4):
    a.grid()
    a.legend(title=f'f = 300 kHz   H$_0$ = {H0[5]:.0f} kA/m',loc='lower right',shadow=True,frameon=True,ncol=2)
    a.set_ylabel('T (°C)')
ax4.set_xlabel('t (s)')
plt.suptitle('EG51% + FF49% --> LN2 --> 300 kHz & 100 dA')

#% Comparativa e/ centro y borde, diferencia de temperatura
fig62,(ax,ax2,ax3) = plt.subplots(3,1,figsize=(9,9),sharex=True,constrained_layout=True)
ax.set_title('Centro',loc='left')
ax2.set_title('Borde',loc='left')
ax3.set_title(r'$\Delta$T',loc='left')

for i,p in enumerate(temps_100):
    _,time,temp_B, temp_C = lector_templog(p)
    delta_CB = temp_C - temp_B
    ax.plot(time,temp_C,'-',label=i+1)
    ax2.plot(time,temp_B,'-',label=i+1)
    ax3.plot(time,delta_CB,'-',label=i+1)
    indx_min=np.nonzero(temp_C==min(temp_C))[0]
    print(f'Temp minima = {temp_C[np.nonzero(temp_C==min(temp_C))][0]:.1f} C ({temp_C[np.nonzero(temp_C==min(temp_C))][0]+273:.1f} K) alcanzada en {time[np.nonzero(temp_C==min(temp_C))][0]:.1f} s')

for a in (ax,ax2,ax3):
    a.grid()
    a.legend(title=f'f = 300 kHz   H$_0$ = {H0[5]:.0f} kA/m',loc='best',shadow=True,frameon=True,ncol=2)
    a.set_ylabel('T (°C)')
ax3.set_xlabel('t (s)')
ax3.set_xlim(0,200)
plt.suptitle('EG51% + FF49% --> LN2 --> 300 kHz & 100 dA')

#%%7 090 dA = 34.7 kA/m
fig71,(ax1,ax2) = plt.subplots(2,1,figsize=(9,6),sharex=True,constrained_layout=True)
_,time_1,temp_B_1, temp_C_1 = lector_templog(temps_090[0])
delta_CB_1 = temp_C_1 - temp_B_1
indx_max_DT_1 = np.nonzero(delta_CB_1==max(delta_CB_1))[0]
ax1.plot(time_1,temp_C_1,'C0-',label='Centro')#
ax1.plot(time_1,temp_B_1,'C0-.',label='Borde')
ax1.vlines(time_1[indx_max_DT_1],temp_B_1[indx_max_DT_1],temp_C_1[indx_max_DT_1],color='C0',ls='--',label=f'ΔT max = {delta_CB_1[indx_max_DT_1][0]:.1f} °C')

_,time_2,temp_B_2, temp_C_2 = lector_templog(temps_090[1])
delta_CB_2 = temp_C_2 - temp_B_2
indx_max_DT_2 = np.nonzero(delta_CB_2==max(delta_CB_2))[0]
ax2.plot(time_2,temp_C_2,'C1-',label='Centro')#
ax2.plot(time_2,temp_B_2,'C1-.',label='Borde')
ax2.vlines(time_2[indx_max_DT_2],temp_B_2[indx_max_DT_2],temp_C_2[indx_max_DT_2],color='C1',ls='--',label=f'ΔT max = {delta_CB_2[indx_max_DT_2][0]:.1f} °C')

ax2.set_xlim(0,220)

for a in (ax1,ax2):
    a.grid()
    a.legend(title=f'f = 300 kHz   H$_0$ = {H0[6]:.0f} kA/m',loc='lower right',shadow=True,frameon=True,ncol=2)
    a.set_ylabel('T (°C)')
ax2.set_xlabel('t (s)')
plt.suptitle('EG51% + FF49% --> LN2 --> 300 kHz & 090 dA')

#% Comparativa e/ centro y borde, diferencia de temperatura
fig72,(ax,ax2,ax3) = plt.subplots(3,1,figsize=(9,9),sharex=True,constrained_layout=True)
ax.set_title('Centro',loc='left')
ax2.set_title('Borde',loc='left')
ax3.set_title(r'$\Delta$T',loc='left')

for i,p in enumerate(temps_090):
    _,time,temp_B, temp_C = lector_templog(p)
    delta_CB = temp_C - temp_B
    ax.plot(time,temp_C,'-',label=i+1)
    ax2.plot(time,temp_B,'-',label=i+1)
    ax3.plot(time,delta_CB,'-',label=i+1)
    indx_min=np.nonzero(temp_C==min(temp_C))[0]
    print(f'Temp minima = {temp_C[np.nonzero(temp_C==min(temp_C))][0]:.1f} C ({temp_C[np.nonzero(temp_C==min(temp_C))][0]+273:.1f} K) alcanzada en {time[np.nonzero(temp_C==min(temp_C))][0]:.1f} s')

for a in (ax,ax2,ax3):
    a.grid()
    a.legend(title=f'f = 300 kHz   H$_0$ = {H0[6]:.0f} kA/m',loc='best',shadow=True,frameon=True,ncol=2)
    a.set_ylabel('T (°C)')
ax3.set_xlabel('t (s)')
ax3.set_xlim(0,220)
plt.suptitle('EG51% + FF49% --> LN2 --> 300 kHz & 090 dA')

#%%8 080 dA = 31.0 kA/m
fig81,(ax1,ax2) = plt.subplots(2,1,figsize=(9,6),sharex=True,constrained_layout=True)
_,time_1,temp_B_1, temp_C_1 = lector_templog(temps_080[0])
delta_CB_1 = temp_C_1 - temp_B_1
indx_max_DT_1 = np.nonzero(delta_CB_1==max(delta_CB_1))[0]
ax1.plot(time_1,temp_C_1,'C0-',label='Centro')#
ax1.plot(time_1,temp_B_1,'C0-.',label='Borde')
ax1.vlines(time_1[indx_max_DT_1],temp_B_1[indx_max_DT_1],temp_C_1[indx_max_DT_1],color='C0',ls='--',label=f'ΔT max = {delta_CB_1[indx_max_DT_1][0]:.1f} °C')

_,time_2,temp_B_2, temp_C_2 = lector_templog(temps_080[1])
delta_CB_2 = temp_C_2 - temp_B_2
indx_max_DT_2 = np.nonzero(delta_CB_2==max(delta_CB_2))[0]
ax2.plot(time_2,temp_C_2,'C1-',label='Centro')#
ax2.plot(time_2,temp_B_2,'C1-.',label='Borde')
ax2.vlines(time_2[indx_max_DT_2],temp_B_2[indx_max_DT_2],temp_C_2[indx_max_DT_2],color='C1',ls='--',label=f'ΔT max = {delta_CB_2[indx_max_DT_2][0]:.1f} °C')

ax2.set_xlim(0,315)

for a in (ax1,ax2):
    a.grid()
    a.legend(title=f'f = 300 kHz   H$_0$ = {H0[7]:.0f} kA/m',loc='lower right',shadow=True,frameon=True,ncol=2)
    a.set_ylabel('T (°C)')
ax2.set_xlabel('t (s)')
plt.suptitle('EG51% + FF49% --> LN2 --> 300 kHz & 080 dA')

#% Comparativa e/ centro y borde, diferencia de temperatura
fig82,(ax,ax2,ax3) = plt.subplots(3,1,figsize=(9,9),sharex=True,constrained_layout=True)
ax.set_title('Centro',loc='left')
ax2.set_title('Borde',loc='left')
ax3.set_title(r'$\Delta$T',loc='left')

for i,p in enumerate(temps_080):
    _,time,temp_B, temp_C = lector_templog(p)
    delta_CB = temp_C - temp_B
    ax.plot(time,temp_C,'-',label=i+1)
    ax2.plot(time,temp_B,'-',label=i+1)
    ax3.plot(time,delta_CB,'-',label=i+1)
    indx_min=np.nonzero(temp_C==min(temp_C))[0]
    print(f'Temp minima = {temp_C[np.nonzero(temp_C==min(temp_C))][0]:.1f} C ({temp_C[np.nonzero(temp_C==min(temp_C))][0]+273:.1f} K) alcanzada en {time[np.nonzero(temp_C==min(temp_C))][0]:.1f} s')

for a in (ax,ax2,ax3):
    a.grid()
    a.legend(title=f'f = 300 kHz   H$_0$ = {H0[7]:.0f} kA/m',loc='best',shadow=True,frameon=True,ncol=2)
    a.set_ylabel('T (°C)')
ax3.set_xlabel('t (s)')
ax3.set_xlim(0,315)
plt.suptitle('EG51% + FF49% --> LN2 --> 300 kHz & 080 dA')
#%%9 060 dA = 27.3 kA/m   
fig91,(ax1,ax2) = plt.subplots(2,1,figsize=(9,6),sharex=True,constrained_layout=True)
_,time_1,temp_B_1, temp_C_1 = lector_templog(temps_060[0])
delta_CB_1 = temp_C_1 - temp_B_1
indx_max_DT_1 = np.nonzero(delta_CB_1==max(delta_CB_1))[0]
ax1.plot(time_1,temp_C_1,'C0-',label='Centro')#
ax1.plot(time_1,temp_B_1,'C0-.',label='Borde')
ax1.vlines(time_1[indx_max_DT_1],temp_B_1[indx_max_DT_1],temp_C_1[indx_max_DT_1],color='C0',ls='--',label=f'ΔT max = {delta_CB_1[indx_max_DT_1][0]:.1f} °C')

_,time_2,temp_B_2, temp_C_2 = lector_templog(temps_060[1])
delta_CB_2 = temp_C_2 - temp_B_2
indx_max_DT_2 = np.nonzero(delta_CB_2==max(delta_CB_2))[0]
ax2.plot(time_2,temp_C_2,'C1-',label='Centro')#
ax2.plot(time_2,temp_B_2,'C1-.',label='Borde')
ax2.vlines(time_2[indx_max_DT_2],temp_B_2[indx_max_DT_2],temp_C_2[indx_max_DT_2],color='C1',ls='--',label=f'ΔT max = {delta_CB_2[indx_max_DT_2][0]:.1f} °C')

ax2.set_xlim(0,315)

for a in (ax1,ax2):
    a.grid()
    a.legend(title=f'f = 300 kHz   H$_0$ = {H0[6]:.0f} kA/m',loc='lower right',shadow=True,frameon=True,ncol=2)
    a.set_ylabel('T (°C)')
ax2.set_xlabel('t (s)')
plt.suptitle('EG51% + FF49% --> LN2 --> 300 kHz & 060 dA')

#% Comparativa e/ centro y borde, diferencia de temperatura
fig92,(ax,ax2,ax3) = plt.subplots(3,1,figsize=(9,9),sharex=True,constrained_layout=True)
ax.set_title('Centro',loc='left')
ax2.set_title('Borde',loc='left')
ax3.set_title(r'$\Delta$T',loc='left')

for i,p in enumerate(temps_060):
    _,time,temp_B, temp_C = lector_templog(p)
    delta_CB = temp_C - temp_B
    ax.plot(time,temp_C,'-',label=i+1)
    ax2.plot(time,temp_B,'-',label=i+1)
    ax3.plot(time,delta_CB,'-',label=i+1)
    indx_min=np.nonzero(temp_C==min(temp_C))[0]
    print(f'Temp minima = {temp_C[np.nonzero(temp_C==min(temp_C))][0]:.1f} C ({temp_C[np.nonzero(temp_C==min(temp_C))][0]+273:.1f} K) alcanzada en {time[np.nonzero(temp_C==min(temp_C))][0]:.1f} s')

for a in (ax,ax2,ax3):
    a.grid()
    a.legend(title=f'f = 300 kHz   H$_0$ = {H0[6]:.0f} kA/m',loc='best',shadow=True,frameon=True,ncol=2)
    a.set_ylabel('T (°C)')
ax3.set_xlabel('t (s)')
ax3.set_xlim(0,315)
plt.suptitle('EG51% + FF49% --> LN2 --> 300 kHz & 060 dA')
#%% Comparo mejor enrfriamiento/calentamiento para cada campo
labels = [f'{H0[i]:.1f} kA/m' for i in range(len(H0))]

def separa_enfriamiento_calentamiento(t, T):
    i_min = np.argmin(T)
    t_min = t[i_min]

    # Enfriamiento
    t_enf = t[:i_min + 1] - t_min
    T_enf = T[:i_min + 1]

    # Calentamiento
    t_cal = t[i_min:] - t_min
    T_cal = T[i_min:]

    return (t_enf, T_enf), (t_cal, T_cal)

#%% Enfriamiento Centro y Borde
paths_300kHz = [temps_150[0],temps_125[0],temps_075[1],temps_120[1],temps_110[0],temps_100[2],temps_090[0],temps_080[0],temps_060[0]]

fig_enf, (ax,bx)= plt.subplots(2,1,figsize=(9,6),constrained_layout=True,sharex=True)
for i,p in enumerate(paths_300kHz):
    _, time, temp_B, temp_C = lector_templog(p)

    (t_enf_C, T_enf_C), _ = separa_enfriamiento_calentamiento(time, temp_C)
    (t_enf_B, T_enf_B), _ = separa_enfriamiento_calentamiento(time, temp_B)

    label = f'{H0[paths_300kHz.index(p)]:.1f}'

    ax.plot(t_enf_C, T_enf_C, '-', label=labels[paths_300kHz.index(p)])

    bx.plot(t_enf_B, T_enf_B, '-',label=labels[paths_300kHz.index(p)])

ax.set_title('Centro',loc='left')
bx.set_title('Borde',loc='left')

bx.set_xlabel('t (s)')
for a in (ax,bx):
    a.grid()
    a.legend(title=f'H$_0$ (kA/m)',loc='best',shadow=True,frameon=True,ncol=3)
    a.set_ylabel('T (°C)')
plt.suptitle('EG51% + FF49%\nEnfriamiento LN2')
#%%Calentamiento Centro y Borde
fig_cal, (ax,bx)= plt.subplots(2,1,figsize=(9,6),constrained_layout=True,sharex=True)
for i,p in enumerate(paths_300kHz):
    _, time, temp_B, temp_C = lector_templog(p)

    _,(t_cal_C, T_cal_C) = separa_enfriamiento_calentamiento(time, temp_C)
    _,(t_cal_B, T_cal_B) = separa_enfriamiento_calentamiento(time, temp_B)

    ax.plot(t_cal_C, T_cal_C, '-', label=labels[paths_300kHz.index(p)])

    bx.plot(t_cal_B, T_cal_B, '-',label=labels[paths_300kHz.index(p)])

ax.set_title('Centro',loc='left')
bx.set_title('Borde',loc='left')

bx.set_xlabel('t (s)')
bx.set_xlim(0,350)
for a in (ax,bx):
    a.grid()
    a.legend(title=f'H$_0$ (kA/m)',loc='best',shadow=True,frameon=True,ncol=3)
    a.set_ylabel('T (°C)')
plt.suptitle('EG51% + FF49%\nCalentamiento RF')


#%% Curvatura
fig1,(ax,ax2,ax3) = plt.subplots(3,1,figsize=(10,10),constrained_layout=True)
ax.set_title('Temp vs time',loc='left')
ax2.set_title('Curvatura vs Temp',loc='left')
ax3.set_title('Curvatura vs Temp',loc='left')

for i,p in enumerate(paths_300kHz):
    if i < 3:
        _,time,temp,_ = lector_templog(p)
        indx_min = np.argmin(temp)
        indx_max = indx_min + np.argmax(temp[indx_min:])
        print('\n',os.path.basename(p))
        print(f'T min = {temp[indx_min]:.1f} C ({temp[indx_min]+273:.1f} K) alcanzada en {time[indx_min]:.1f} s')
        print(f'T max = {temp[indx_max]:.1f} C ({temp[indx_max]+273:.1f} K) alcanzada en {time[indx_max]:.1f} s')

        t_curv = time[indx_min:indx_max]
        T_curv = temp[indx_min:indx_max]
        
        mask = (T_curv < 10)
        T_curv = T_curv[mask]
        t_curv = t_curv[mask]
        _,_,curv = curvatura(T_curv)

        ax.plot(t_curv-t_curv[0],T_curv,'.-',label=f'H$_0$ = {H0[i]:.1f}')
        ax2.plot(T_curv,curv,'.-',label=f'H$_0$ = {H0[i]:.1f}')
        ax3.plot(T_curv,curv,'.-',label=f'H$_0$ = {H0[i]:.1f}')

ax.axhspan(-150,-075,color='tab:red',alpha=0.2,zorder=-1)
ax.axhspan(-75,-50,color='tab:purple',alpha=0.2,zorder=-1)

ax2.axvspan(-150,-075,color='tab:red',alpha=0.2,zorder=-1)
ax3.axvspan(-75,-50,color='tab:purple',alpha=0.2,zorder=-1)
ax2.axhline(0.02,color='k',ls='--',lw=1,label='0.02')
ax3.axhline(0.02,color='k',ls='--',lw=1,label='0.02')

ax2.set_ylim(0,0.05)
ax3.set_ylim(0,0.05)
ax2.set_xlim(-180,-100)
ax3.set_xlim(-100,0)
ax.set_xlabel('t (s)')
ax.set_ylabel('T (°C)')
ax2.set_ylabel('Curvatura')
ax3.set_ylabel('Curvatura')
ax3.set_xlabel('T (°C)')

for a in [ax,ax2,ax3]:
    a.grid()
    a.legend(title='H$_0$ (kA/m)',ncol=1,shadow=True,frameon=True)
plt.suptitle('EG 51% FF 49%    LN2 --> RF\nf = 300 kHz   dA = [150, 125, 075]')
plt.savefig('EG51_FF49_LN2_300kHz_Curvatura_1',dpi=300)

#%% Ahora los otros campos
fig2,(ax,ax2,ax3) = plt.subplots(3,1,figsize=(10,10),constrained_layout=True)
ax.set_title('Temp vs time',loc='left')
ax2.set_title('Curvatura vs Temp',loc='left')
ax3.set_title('Curvatura vs Temp',loc='left')

for i,p in enumerate(paths_300kHz):
    if (i>2) & (i<6):
        _,time,temp,_ = lector_templog(p)
        indx_min = np.argmin(temp)
        indx_max = indx_min + np.argmax(temp[indx_min:])
        print('\n',os.path.basename(p))
        print(f'T min = {temp[indx_min]:.1f} C ({temp[indx_min]+273:.1f} K) alcanzada en {time[indx_min]:.1f} s')
        print(f'T max = {temp[indx_max]:.1f} C ({temp[indx_max]+273:.1f} K) alcanzada en {time[indx_max]:.1f} s')

        t_curv = time[indx_min:indx_max]
        T_curv = temp[indx_min:indx_max]
        
        mask = (T_curv < 10) & (T_curv > -170)
        T_curv = T_curv[mask]
        t_curv = t_curv[mask]
        _,_,curv = curvatura(T_curv)

        ax.plot(t_curv-t_curv[0],T_curv,'.-',label=f'H$_0$ = {H0[i]:.1f}')
        ax2.plot(T_curv,curv,'.-',label=f'H$_0$ = {H0[i]:.1f}')
        ax3.plot(T_curv,curv,'.-',label=f'H$_0$ = {H0[i]:.1f}')

ax.axhspan(-160,-075,color='tab:red',alpha=0.2,zorder=-1)
ax.axhspan(-75,-40,color='tab:purple',alpha=0.2,zorder=-1)
ax2.axvspan(-160,-075,color='tab:red',alpha=0.2,zorder=-1)
ax3.axvspan(-75,-40,color='tab:purple',alpha=0.2,zorder=-1)
ax2.axhline(0.02,color='k',ls='--',lw=1,label='0.02')
ax3.axhline(0.02,color='k',ls='--',lw=1,label='0.02')

ax.set_xlim(0,)
ax2.set_ylim(0,0.1)
ax3.set_ylim(0,0.1)
ax2.set_xlim(-170,-100)
ax3.set_xlim(-100,0)
ax.set_xlabel('t (s)')
ax.set_ylabel('T (°C)')
ax2.set_ylabel('Curvatura')
ax3.set_ylabel('Curvatura')
ax3.set_xlabel('T (°C)')

for a in [ax,ax2,ax3]:
    a.grid()
    a.legend(title='H$_0$ (kA/m)',ncol=1,shadow=True,frameon=True)
plt.suptitle('EG 51% FF 49%    LN2 --> RF\nf = 300 kHz   dA = [120, 110, 100]')
plt.savefig('EG51_FF49_LN2_300kHz_Curvatura_2',dpi=300)
#%%
fig3,(ax,ax2,ax3) = plt.subplots(3,1,figsize=(10,10),constrained_layout=True)
ax.set_title('Temp vs time',loc='left')
ax2.set_title('Curvatura vs Temp',loc='left')
ax3.set_title('Curvatura vs Temp',loc='left')

for i,p in enumerate(paths_300kHz):
    if (i>5) & (i<9):
        _,time,temp,_ = lector_templog(p)
        indx_min = np.argmin(temp)
        indx_max = indx_min + np.argmax(temp[indx_min:])
        print('\n',os.path.basename(p))
        print(f'T min = {temp[indx_min]:.1f} C ({temp[indx_min]+273:.1f} K) alcanzada en {time[indx_min]:.1f} s')
        print(f'T max = {temp[indx_max]:.1f} C ({temp[indx_max]+273:.1f} K) alcanzada en {time[indx_max]:.1f} s')

        t_curv = time[indx_min:indx_max]
        T_curv = temp[indx_min:indx_max]
        
        mask = (T_curv < 10) & (T_curv > -170)
        T_curv = T_curv[mask]
        t_curv = t_curv[mask]
        _,_,curv = curvatura(T_curv)

        ax.plot(t_curv-t_curv[0],T_curv,'.-',label=f'H$_0$ = {H0[i]:.1f}')
        ax2.plot(T_curv,curv,'.-',label=f'H$_0$ = {H0[i]:.1f}')
        ax3.plot(T_curv,curv,'.-',label=f'H$_0$ = {H0[i]:.1f}')

ax.axhspan(-160,-075,color='tab:red',alpha=0.2,zorder=-1)
ax.axhspan(-75,-40,color='tab:purple',alpha=0.2,zorder=-1)
ax2.axvspan(-160,-075,color='tab:red',alpha=0.2,zorder=-1)
ax3.axvspan(-75,-40,color='tab:purple',alpha=0.2,zorder=-1)
ax2.axhline(0.02,color='k',ls='--',lw=1,label='0.02')
ax3.axhline(0.02,color='k',ls='--',lw=1,label='0.02')

ax.set_xlim(0,)
ax2.set_ylim(0,0.1)
ax3.set_ylim(0,0.1)
ax2.set_xlim(-170,-100)
ax3.set_xlim(-100,0)
ax.set_xlabel('t (s)')
ax.set_ylabel('T (°C)')
ax2.set_ylabel('Curvatura')
ax3.set_ylabel('Curvatura')
ax3.set_xlabel('T (°C)')

for a in [ax,ax2,ax3]:
    a.grid()
    a.legend(title='H$_0$ (kA/m)',ncol=1,shadow=True,frameon=True)
plt.suptitle('EG 51% FF 49%    LN2 --> RF\nf = 300 kHz   dA = [090, 080, 060]')
plt.savefig('EG51_FF49_LN2_300kHz_Curvatura_3',dpi=300)


#%% Ajuste exponencial y bi-exponencial
# Taux=[]
# fits_exp=[]
# fits_biexp=[]
# residuos_exp=[]
# residuos_biexp=[]

# for i,(x,y) in enumerate(zip(t,T)):

#     t_aux,T_aux = x[Indx_min[i]:],y[Indx_min[i]:]

#     # Ajuste exponencial simple
#     p0_exp = [T_aux[-1],T_aux[0]-T_aux[-1],200]

#     (A,B,tau),_ = curve_fit(
#         expo,
#         t_aux,
#         T_aux,
#         p0=p0_exp
#     )

#     Tfit_exp = expo(t_aux,A,B,tau)
#     res_exp = T_aux - Tfit_exp

#     # Ajuste bi-exponencial
#     p0_bi = [
#         T_aux[-1],
#         0.7*(T_aux[0]-T_aux[-1]),50,
#         0.3*(T_aux[0]-T_aux[-1]),300
#     ]

#     (A2,B1,tau1,B2,tau2),_ = curve_fit(
#         biexpo,
#         t_aux,
#         T_aux,
#         p0=p0_bi,
#         maxfev=10000
#     )

#     Tfit_bi = biexpo(t_aux,A2,B1,tau1,B2,tau2)
#     res_bi = T_aux - Tfit_bi

#     # -------------------------------------------------------
#     # Métricas
#     # -------------------------------------------------------
#     sigma = 0.1
#     N = len(T_aux)

#     k_exp = 3
#     k_bi  = 5

#     rss_exp = np.sum(res_exp**2)
#     rss_bi  = np.sum(res_bi**2)

#     r2_exp = 1 - rss_exp/np.sum((T_aux-T_aux.mean())**2)
#     r2_bi  = 1 - rss_bi/np.sum((T_aux-T_aux.mean())**2)

#     rmse_exp = np.sqrt(np.mean(res_exp**2))
#     rmse_bi  = np.sqrt(np.mean(res_bi**2))

#     chi2_exp = np.sum((res_exp/sigma)**2)
#     chi2_bi  = np.sum((res_bi/sigma)**2)

#     chi2red_exp = chi2_exp/(N-k_exp)
#     chi2red_bi  = chi2_bi/(N-k_bi)

#     aic_exp = N*np.log(rss_exp/N) + 2*k_exp
#     aic_bi  = N*np.log(rss_bi/N) + 2*k_bi

#     delta_aic = aic_exp - aic_bi

#     print(f'H0 = {H0[i]:.1f} kA/m')
#     print(f'Exp:   tau = {tau:6.1f} s | RMSE = {rmse_exp:6.3f} °C | R² = {r2_exp:7.5f} | χ²ν = {chi2red_exp:8.2f} | AIC = {aic_exp:8.1f}')
#     print(f'BiExp: tau1 = {tau1:6.1f} s | tau2 = {tau2:6.1f} s | RMSE = {rmse_bi:6.3f} °C | R² = {r2_bi:7.5f} | χ²ν = {chi2red_bi:8.2f} | AIC = {aic_bi:8.1f}')
#     print(f'ΔAIC = {delta_aic:.1f}')
#     print('-'*100)

#     Taux.append(T_aux)
#     fits_exp.append(Tfit_exp)
#     fits_biexp.append(Tfit_bi)
#     residuos_exp.append(res_exp)
#     residuos_biexp.append(res_bi)

#     fig,(ax1,ax2)=plt.subplots(2,1,figsize=(8,5),constrained_layout=True)
#     ax1.plot(t_aux-t_aux[0],T_aux,'.',ms=3,label='Datos')

#     ax1.plot(t_aux-t_aux[0],Tfit_exp,'-',lw=2,label='Exp.')

#     ax1.plot(t_aux-t_aux[0],Tfit_bi,'--',lw=2,label='Bi-exp.')

#     texto = (rf'$\chi^2_{{\nu,\mathrm{{exp}}}}={chi2red_exp:.2f}$' '\n'
#         rf'$\chi^2_{{\nu,\mathrm{{bi}}}}={chi2red_bi:.2f}$' '\n'
#         rf'$\Delta AIC={delta_aic:.1f}$')

#     ax1.text(0.8,0.5,texto,transform=ax1.transAxes,ha='left',va='top',bbox=dict(boxstyle='round',facecolor='white',alpha=0.85))

#     ax1.set_title('Temperatura vs tiempo',loc='left')
#     ax1.set_ylabel('T (°C)')
#     ax1.set_xlabel('t (s)')

#     ax2.set_title('Residuos vs temperatura',loc='left')

#     ax2.plot(T_aux,res_exp,'.-',label='Residuo exp.')

#     ax2.plot(T_aux,res_bi,'.-',label='Residuo bi-exp.')

#     ax2.axhline(0,color='k',lw=1)

#     ax2.set_xlim(-175,0)
#     ax2.set_xlabel('T (°C)')
#     ax2.set_ylabel('Residuo (°C)')

#     for a in [ax1,ax2]:
#         a.grid()
#         a.legend()

#     plt.suptitle(f'2.5 - EG 55% FF 45% - LN2 → RF - H$_0$ = {H0[i]:.1f} kA/m')
# #%% Ploteo todos los resuiduos
# fig26,ax = plt.subplots(figsize=(8,4),constrained_layout=True)
# ax.set_title('Residuos vs Temp - EG 55% FF 45%',loc='left')

# for i in range(len(residuos_biexp)):
#     plt.plot(Taux[i], residuos_biexp[i],'.-',label=f'{H0[i]:.1f}')
# ax.axhline(0,color='k',lw=1)
# ax.axvspan(-150,-075,color='tab:red',alpha=0.2,zorder=-1)
# ax.axvspan(-60,-40,color='tab:orange',alpha=0.2,zorder=-1)
# ax.grid()
# ax.legend(title=' H$_0$ (kA/m)',loc='lower right',ncol=1)
# ax.set_xlim(-170,0)
# ax.set_ylim(-10,10)
#%% salvo figuras
# %% Salvo figuras

fig11.savefig('EG51_FF49_LN2_300kHz_150dA_1',dpi=300)
fig12.savefig('EG51_FF49_LN2_300kHz_150dA_2',dpi=300)
fig21.savefig('EG51_FF49_LN2_300kHz_125dA_1',dpi=300)
fig22.savefig('EG51_FF49_LN2_300kHz_125dA_2',dpi=300)
fig31.savefig('EG51_FF49_LN2_300kHz_075dA_1',dpi=300)
fig32.savefig('EG51_FF49_LN2_300kHz_075dA_2',dpi=300)
fig41.savefig('EG51_FF49_LN2_300kHz_120dA_1',dpi=300)
fig42.savefig('EG51_FF49_LN2_300kHz_120dA_2',dpi=300)
fig51.savefig('EG51_FF49_LN2_300kHz_110dA_1',dpi=300)
fig52.savefig('EG51_FF49_LN2_300kHz_110dA_2',dpi=300)
fig61.savefig('EG51_FF49_LN2_300kHz_100dA_1',dpi=300)
fig62.savefig('EG51_FF49_LN2_300kHz_100dA_2',dpi=300)
fig71.savefig('EG51_FF49_LN2_300kHz_090dA_1',dpi=300)
fig72.savefig('EG51_FF49_LN2_300kHz_090dA_2',dpi=300)   
fig81.savefig('EG51_FF49_LN2_300kHz_080dA_1',dpi=300)
fig82.savefig('EG51_FF49_LN2_300kHz_080dA_2',dpi=300)
fig91.savefig('EG51_FF49_LN2_300kHz_060dA_1',dpi=300)
fig92.savefig('EG51_FF49_LN2_300kHz_060dA_2',dpi=300)
fig_enf.savefig('EG51_FF49_LN2_300kHz_Enfriamiento',dpi=300)
fig_cal.savefig('EG51_FF49_LN2_300kHz_Calentamiento',dpi=300)
fig1.savefig('EG51_FF49_curvatura_150_125_075.png',dpi=300)
fig2.savefig('EG51_FF49_curvatura_120_110_100.png',dpi=300)
fig3.savefig('EG51_FF49_curvatura_090_080_060.png',dpi=300)

# %%
