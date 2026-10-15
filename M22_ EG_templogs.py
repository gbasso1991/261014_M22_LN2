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
# def lector_templog(path):
#     '''
#     Busca archivo *templog.csv en directorio especificado.
#     muestras = False plotea solo T(dt).
#     muestras = True plotea T(dt) con las muestras superpuestas
#     Retorna arrys timestamp,temperatura
#     '''
#     data = pd.read_csv(path,sep=';',header=5,
#                             names=('Timestamp','T_CH1','T_CH2'),usecols=(0,1,2),
#                             decimal=',',engine='python')
#     temp_CH1  = pd.Series(data['T_CH1']).to_numpy(dtype=float)
#     temp_CH2  = pd.Series(data['T_CH2']).to_numpy(dtype=float)
#     timestamp = np.array([datetime.strptime(date,'%Y/%m/%d %H:%M:%S') for date in data['Timestamp']])

#     time = np.array([(t-timestamp[0]).total_seconds() for t in timestamp])
#     return timestamp,time,temp_CH1, temp_CH2

import warnings

def lector_templog(path, sincronizo_CH1_t=10, delta_T_min=0.8):
    '''
    15 Oct 26
    Busca archivo *templog.csv en directorio especificado.
    Si sincronizo_CH1_t no es None, sincroniza la curva para que el punto
    anterior al primer cambio de CH1 >= delta_T_min quede en t=sincronizo_CH1_t.
    Recorta CH1 y CH2 con los mismos índices.
    Retorna timestamp, time, temp_CH1, temp_CH2.
    '''
    data = pd.read_csv(path,sep=';',header=5,
                       names=('Timestamp','T_CH1','T_CH2'),usecols=(0,1,2),
                       decimal=',',engine='python')
    temp_CH1 = pd.Series(data['T_CH1']).to_numpy(dtype=float)
    temp_CH2 = pd.Series(data['T_CH2']).to_numpy(dtype=float)
    timestamp = np.array([datetime.strptime(date,'%Y/%m/%d %H:%M:%S') for date in data['Timestamp']])
    time = np.array([(t-timestamp[0]).total_seconds() for t in timestamp])

    if sincronizo_CH1_t is not None:
        indx_cambios = np.where(np.abs(np.diff(temp_CH1)) >= delta_T_min)[0]

        if len(indx_cambios) == 0:
            warnings.warn(f'{os.path.basename(path)}: CH1 no presenta cambios >= {delta_T_min:.1f} °C. No se sincroniza.')
        else:
            indx_anterior = indx_cambios[0]
            t_anterior = time[indx_anterior]

            if np.isclose(t_anterior, sincronizo_CH1_t):
                warnings.warn(f'{os.path.basename(path)}: el punto anterior al primer cambio ya está en t={sincronizo_CH1_t:.1f} s. No hace falta recortar.')
            elif t_anterior < sincronizo_CH1_t:
                warnings.warn(f'{os.path.basename(path)}: el primer cambio ocurre demasiado pronto; el punto anterior está en t={t_anterior:.1f} s, antes de los {sincronizo_CH1_t:.1f} s solicitados. No se sincroniza mediante recorte.')
            else:
                tiempo_a_recortar = t_anterior - sincronizo_CH1_t
                indx_inicio = np.searchsorted(time, tiempo_a_recortar, side='left')
                segundos_recortados = time[indx_inicio]
                timestamp = timestamp[indx_inicio:]
                time = time[indx_inicio:] - tiempo_a_recortar
                temp_CH1 = temp_CH1[indx_inicio:]
                temp_CH2 = temp_CH2[indx_inicio:]
                print(f'{os.path.basename(path)}: recortados {segundos_recortados:.1f} s del comienzo; punto anterior al cambio en t={time[indx_anterior-indx_inicio]:.1f} s.')

    return timestamp, time, temp_CH1, temp_CH2

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
Idc = [15, 12.5, 10, 9 , 8.5, 8, 7.5, 6] dA\n'''
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
Idc_values = [15, 12.5, 10, 9 , 8.5, 8, 7.5, 6]
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
indx_max_DT_1 = np.argmax(delta_CB_1) 
t_max_delta_1 , T_max_delta_1 = time_1[indx_max_DT_1] , temp_C_1[indx_max_DT_1]

ax1.plot(time_1,temp_C_1,'C0-',label='Centro')
ax1.plot(time_1,temp_B_1,'C0-.',label='Borde')
ax1.vlines(t_max_delta_1,temp_B_1[indx_max_DT_1],temp_C_1[indx_max_DT_1],color='C0',ls='--',
           label='ΔT$_{max}$ = '+f'{delta_CB_1[indx_max_DT_1]:.1f} °C'+f'\n        t = {t_max_delta_1:.1f} s')

_,time_2,temp_C_2, temp_B_2 = lector_templog(temps_150[1])
delta_CB_2 = temp_C_2 - temp_B_2
indx_max_DT_2 = np.argmax(delta_CB_2) 
t_max_delta_2 , T_max_delta_2 = time_2[indx_max_DT_2] , temp_C_2[indx_max_DT_2]
ax2.plot(time_2,temp_C_2,'C1-',label='Centro')#
ax2.plot(time_2,temp_B_2,'C1-.',label='Borde')
ax2.vlines(time_2[indx_max_DT_2],temp_B_2[indx_max_DT_2],temp_C_2[indx_max_DT_2],color='C1',ls='--',
           label='ΔT$_{max}$ = '+f'{delta_CB_2[indx_max_DT_2]:.1f} °C'+f'\n        t = {t_max_delta_2:.1f} s')
            
ax2.set_xlim(0,200)

for a in (ax1,ax2):
    a.grid()
    a.legend(title=f'f = 300 kHz   H$_0$ = {H0[0]:.0f} kA/m',loc='upper center',shadow=True,frameon=True,ncol=2)
    a.axvline(10,color='grey',ls='--',lw=1)
    a.set_ylabel('T (°C)')
ax2.set_xlabel('t (s)')
plt.suptitle(titulo_1)

#%Comparativa e/ centro y borde, diferencia de temperatura
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
    print(f'Temp minima = {temp_C[np.nonzero(temp_C==min(temp_C))][0]:.1f} C ({temp_C[np.nonzero(temp_C==min(temp_C))][0]+273.15:.1f} K) alcanzada en {time[np.nonzero(temp_C==min(temp_C))][0]:.1f} s')

for a in (ax,ax2,ax3):
    a.grid()
    a.legend(title=f'f = 300 kHz   H$_0$ = {H0[0]:.0f} kA/m',
             loc='upper center',shadow=True,frameon=True,ncol=2)
    a.set_ylabel('T (°C)')
    a.axvline(10,color='grey',ls='--',lw=1)

ax2.set_xlabel('t (s)')
ax2.set_xlim(0,200)
plt.suptitle(titulo_1)

#%% 2 idc = 125 dA
titulo_2 = 'M22 - EG51% + FF49% --> LN2 --> 300 kHz & 125 dA'
fig21,(ax1,ax2) = plt.subplots(2,1,figsize=(9,6),sharex=True,constrained_layout=True)
_,time_1,temp_C_1,temp_B_1 = lector_templog(temps_125[0])
delta_CB_1 = temp_C_1 - temp_B_1
indx_max_DT_1 = np.argmax(delta_CB_1)
t_max_delta_1 = time_1[indx_max_DT_1]
ax1.plot(time_1,temp_C_1,'C0-',label='Centro')
ax1.plot(time_1,temp_B_1,'C0-.',label='Borde')
ax1.vlines(t_max_delta_1,temp_B_1[indx_max_DT_1],temp_C_1[indx_max_DT_1],color='C0',ls='--',
           label='ΔT$_{max}$ = '+f'{delta_CB_1[indx_max_DT_1]:.1f} °C'+f'\n        t = {t_max_delta_1:.1f} s')
_,time_2,temp_C_2,temp_B_2 = lector_templog(temps_125[1])
delta_CB_2 = temp_C_2 - temp_B_2
indx_max_DT_2 = np.argmax(delta_CB_2)
t_max_delta_2 = time_2[indx_max_DT_2]
ax2.plot(time_2,temp_C_2,'C1-',label='Centro')
ax2.plot(time_2,temp_B_2,'C1-.',label='Borde')
ax2.vlines(t_max_delta_2,temp_B_2[indx_max_DT_2],temp_C_2[indx_max_DT_2],color='C1',ls='--',
           label='ΔT$_{max}$ = '+f'{delta_CB_2[indx_max_DT_2]:.1f} °C'+f'\n        t = {t_max_delta_2:.1f} s')
ax2.set_xlim(0,200)
for a in (ax1,ax2):
    a.grid()
    a.legend(title=f'f = 300 kHz   H$_0$ = {H0[1]:.0f} kA/m',loc='upper center',shadow=True,frameon=True,ncol=2)
    a.set_ylabel('T (°C)')
    a.axvline(10,color='grey',ls='--',lw=1)
ax2.set_xlabel('t (s)')
plt.suptitle(titulo_2)

#%% Comparativa e/ centro y borde, diferencia de temperatura
fig22,(ax,ax2,ax3) = plt.subplots(3,1,figsize=(9,9),sharex=True,constrained_layout=True)
ax.set_title('Centro',loc='left')
ax2.set_title('Borde',loc='left')
ax3.set_title(r'$\Delta$T',loc='left')
for i,p in enumerate(temps_125):
    _,time,temp_C,temp_B = lector_templog(p)
    delta_CB = temp_C - temp_B
    ax.plot(time,temp_C,'-',label=i+1)
    ax2.plot(time,temp_B,'-.',label=i+1)
    ax3.plot(time,delta_CB,'-',label=i+1)
    indx_min = np.argmin(temp_C)
    print(f'Temp minima = {temp_C[indx_min]:.1f} C ({temp_C[indx_min]+273.15:.1f} K) alcanzada en {time[indx_min]:.1f} s')
for a in (ax,ax2,ax3):
    a.grid()
    a.legend(title=f'f = 300 kHz   H$_0$ = {H0[1]:.0f} kA/m',loc='upper center',shadow=True,frameon=True,ncol=2)
    a.set_ylabel('T (°C)')
    a.axvline(10,color='grey',ls='--',lw=1)
ax3.set_xlabel('t (s)')
ax3.set_xlim(0,200)
plt.suptitle(titulo_2)
#%% 3 idc = 100 dA
titulo_3 = 'M22 - EG51% + FF49% --> LN2 --> 300 kHz & 100 dA'
fig31,(ax1,ax2) = plt.subplots(2,1,figsize=(9,6),sharex=True,constrained_layout=True)
_,time_1,temp_C_1,temp_B_1 = lector_templog(temps_100[0])
delta_CB_1 = temp_C_1 - temp_B_1
indx_max_DT_1 = np.argmax(delta_CB_1)
t_max_delta_1 = time_1[indx_max_DT_1]
ax1.plot(time_1,temp_C_1,'C0-',label='Centro')
ax1.plot(time_1,temp_B_1,'C0-.',label='Borde')
ax1.vlines(t_max_delta_1,temp_B_1[indx_max_DT_1],temp_C_1[indx_max_DT_1],color='C0',ls='--',label='ΔT$_{max}$ = '+f'{delta_CB_1[indx_max_DT_1]:.1f} °C'+f'\n        t = {t_max_delta_1:.1f} s')
_,time_2,temp_C_2,temp_B_2 = lector_templog(temps_100[1])
delta_CB_2 = temp_C_2 - temp_B_2
indx_max_DT_2 = np.argmax(delta_CB_2)
t_max_delta_2 = time_2[indx_max_DT_2]
ax2.plot(time_2,temp_C_2,'C1-',label='Centro')
ax2.plot(time_2,temp_B_2,'C1-.',label='Borde')
ax2.vlines(t_max_delta_2,temp_B_2[indx_max_DT_2],temp_C_2[indx_max_DT_2],color='C1',ls='--',label='ΔT$_{max}$ = '+f'{delta_CB_2[indx_max_DT_2]:.1f} °C'+f'\n        t = {t_max_delta_2:.1f} s')
ax2.set_xlim(0,230)
for a in (ax1,ax2):
    a.grid()
    a.legend(title=f'f = 300 kHz   H$_0$ = {H0[2]:.0f} kA/m',loc='upper center',shadow=True,frameon=True,ncol=2)
    a.set_ylabel('T (°C)')
    a.axvline(10,color='grey',ls='--',lw=1)
ax2.set_xlabel('t (s)')
plt.suptitle(titulo_3)

#% Comparativa e/ centro y borde, diferencia de temperatura
fig32,(ax,ax2,ax3) = plt.subplots(3,1,figsize=(9,9),sharex=True,constrained_layout=True)
ax.set_title('Centro',loc='left')
ax2.set_title('Borde',loc='left')
ax3.set_title(r'$\Delta$T',loc='left')
for i,p in enumerate(temps_100):
    _,time,temp_C,temp_B = lector_templog(p)
    delta_CB = temp_C - temp_B
    ax.plot(time,temp_C,'-',label=i+1)
    ax2.plot(time,temp_B,'-.',label=i+1)
    ax3.plot(time,delta_CB,'-',label=i+1)
    indx_min = np.argmin(temp_C)
    print(f'Temp minima = {temp_C[indx_min]:.1f} C ({temp_C[indx_min]+273.15:.1f} K) alcanzada en {time[indx_min]:.1f} s')
for a in (ax,ax2,ax3):
    a.grid()
    a.legend(title=f'f = 300 kHz   H$_0$ = {H0[2]:.0f} kA/m',loc='upper center',shadow=True,frameon=True,ncol=2)
    a.set_ylabel('T (°C)')
    a.axvline(10,color='grey',ls='--',lw=1)
ax3.set_xlabel('t (s)')
ax3.set_xlim(0,230)
plt.suptitle(titulo_3)

#%% 4 idc = 090 dA = 34.7 kA/m
titulo_4 = 'M22 - EG51% + FF49% --> LN2 --> 300 kHz & 090 dA'
fig41,(ax1,ax2) = plt.subplots(2,1,figsize=(9,6),sharex=True,constrained_layout=True)
_,time_1,temp_C_1,temp_B_1 = lector_templog(temps_090[0])
delta_CB_1 = temp_C_1 - temp_B_1
indx_max_DT_1 = np.argmax(delta_CB_1)
t_max_delta_1 = time_1[indx_max_DT_1]
ax1.plot(time_1,temp_C_1,'C0-',label='Centro')
ax1.plot(time_1,temp_B_1,'C0-.',label='Borde')
ax1.vlines(t_max_delta_1,temp_B_1[indx_max_DT_1],temp_C_1[indx_max_DT_1],color='C0',ls='--',label='ΔT$_{max}$ = '+f'{delta_CB_1[indx_max_DT_1]:.1f} °C'+f'\n        t = {t_max_delta_1:.1f} s')
_,time_2,temp_C_2,temp_B_2 = lector_templog(temps_090[1])
delta_CB_2 = temp_C_2 - temp_B_2
indx_max_DT_2 = np.argmax(delta_CB_2)
t_max_delta_2 = time_2[indx_max_DT_2]
ax2.plot(time_2,temp_C_2,'C1-',label='Centro')
ax2.plot(time_2,temp_B_2,'C1-.',label='Borde')
ax2.vlines(t_max_delta_2,temp_B_2[indx_max_DT_2],temp_C_2[indx_max_DT_2],color='C1',ls='--',label='ΔT$_{max}$ = '+f'{delta_CB_2[indx_max_DT_2]:.1f} °C'+f'\n        t = {t_max_delta_2:.1f} s')
ax2.set_xlim(0,260)
for a in (ax1,ax2):
    a.grid()
    a.legend(title=f'f = 300 kHz   H$_0$ = {H0[3]:.0f} kA/m',loc='upper center',shadow=True,frameon=True,ncol=2)
    a.set_ylabel('T (°C)')
    a.axvline(10,color='grey',ls='--',lw=1)
ax2.set_xlabel('t (s)')
plt.suptitle(titulo_4)

#% Comparativa e/ centro y borde, diferencia de temperatura
fig42,(ax,ax2,ax3) = plt.subplots(3,1,figsize=(9,9),sharex=True,constrained_layout=True)
ax.set_title('Centro',loc='left')
ax2.set_title('Borde',loc='left')
ax3.set_title(r'$\Delta$T',loc='left')
for i,p in enumerate(temps_090):
    _,time,temp_C,temp_B = lector_templog(p)
    delta_CB = temp_C - temp_B
    ax.plot(time,temp_C,'-',label=i+1)
    ax2.plot(time,temp_B,'-.',label=i+1)
    ax3.plot(time,delta_CB,'-',label=i+1)
    indx_min = np.argmin(temp_C)
    print(f'Temp minima = {temp_C[indx_min]:.1f} C ({temp_C[indx_min]+273.15:.1f} K) alcanzada en {time[indx_min]:.1f} s')
for a in (ax,ax2,ax3):
    a.grid()
    a.legend(title=f'f = 300 kHz   H$_0$ = {H0[3]:.0f} kA/m',loc='upper center',shadow=True,frameon=True,ncol=2)
    a.set_ylabel('T (°C)')
    a.axvline(10,color='grey',ls='--',lw=1)
ax3.set_xlabel('t (s)')
ax3.set_xlim(0,260)
plt.suptitle(titulo_4)

#%% 5 idc = 085 dA
titulo_5 = 'M22 - EG51% + FF49% --> LN2 --> 300 kHz & 085 dA'
fig51,(ax1,ax2) = plt.subplots(2,1,figsize=(9,6),sharex=True,constrained_layout=True)
_,time_1,temp_C_1,temp_B_1 = lector_templog(temps_085[0])
delta_CB_1 = temp_C_1 - temp_B_1
indx_max_DT_1 = np.argmax(delta_CB_1)
t_max_delta_1 = time_1[indx_max_DT_1]
ax1.plot(time_1,temp_C_1,'C0-',label='Centro')
ax1.plot(time_1,temp_B_1,'C0-.',label='Borde')
ax1.vlines(t_max_delta_1,temp_B_1[indx_max_DT_1],temp_C_1[indx_max_DT_1],color='C0',ls='--',label='ΔT$_{max}$ = '+f'{delta_CB_1[indx_max_DT_1]:.1f} °C'+f'\n        t = {t_max_delta_1:.1f} s')
_,time_2,temp_C_2,temp_B_2 = lector_templog(temps_085[1])
delta_CB_2 = temp_C_2 - temp_B_2
indx_max_DT_2 = np.argmax(delta_CB_2)
t_max_delta_2 = time_2[indx_max_DT_2]
ax2.plot(time_2,temp_C_2,'C1-',label='Centro')
ax2.plot(time_2,temp_B_2,'C1-.',label='Borde')
ax2.vlines(t_max_delta_2,temp_B_2[indx_max_DT_2],temp_C_2[indx_max_DT_2],color='C1',ls='--',label='ΔT$_{max}$ = '+f'{delta_CB_2[indx_max_DT_2]:.1f} °C'+f'\n        t = {t_max_delta_2:.1f} s')
ax2.set_xlim(0,300)
for a in (ax1,ax2):
    a.grid()
    a.legend(title=f'f = 300 kHz   H$_0$ = {H0[4]:.0f} kA/m',loc='upper center',shadow=True,frameon=True,ncol=2)
    a.set_ylabel('T (°C)')
    a.axvline(10,color='grey',ls='--',lw=1)
ax2.set_xlabel('t (s)')
plt.suptitle(titulo_5)

#% Comparativa e/ centro y borde, diferencia de temperatura
fig52,(ax,ax2,ax3) = plt.subplots(3,1,figsize=(9,9),sharex=True,constrained_layout=True)
ax.set_title('Centro',loc='left')
ax2.set_title('Borde',loc='left')
ax3.set_title(r'$\Delta$T',loc='left')
for i,p in enumerate(temps_085):
    _,time,temp_C,temp_B = lector_templog(p)
    delta_CB = temp_C - temp_B
    ax.plot(time,temp_C,'-',label=i+1)
    ax2.plot(time,temp_B,'-.',label=i+1)
    ax3.plot(time,delta_CB,'-',label=i+1)
    indx_min = np.argmin(temp_C)
    print(f'Temp minima = {temp_C[indx_min]:.1f} C ({temp_C[indx_min]+273.15:.1f} K) alcanzada en {time[indx_min]:.1f} s')
for a in (ax,ax2,ax3):
    a.grid()
    a.legend(title=f'f = 300 kHz   H$_0$ = {H0[4]:.0f} kA/m',loc='upper center',shadow=True,frameon=True,ncol=2)
    a.set_ylabel('T (°C)')
    a.axvline(10,color='grey',ls='--',lw=1)
ax3.set_xlabel('t (s)')
ax3.set_xlim(0,300)
plt.suptitle(titulo_5)
#%% 6 idc = 080 dA
titulo_6 = 'M22 - EG51% + FF49% --> LN2 --> 300 kHz & 080 dA'
fig61,(ax1,ax2) = plt.subplots(2,1,figsize=(9,6),sharex=True,constrained_layout=True)
_,time_1,temp_C_1,temp_B_1 = lector_templog(temps_080[0])
delta_CB_1 = temp_C_1 - temp_B_1
indx_max_DT_1 = np.argmax(delta_CB_1)
t_max_delta_1 = time_1[indx_max_DT_1]
ax1.plot(time_1,temp_C_1,'C0-',label='Centro')
ax1.plot(time_1,temp_B_1,'C0-.',label='Borde')
ax1.vlines(t_max_delta_1,temp_B_1[indx_max_DT_1],temp_C_1[indx_max_DT_1],color='C0',ls='--',label='ΔT$_{max}$ = '+f'{delta_CB_1[indx_max_DT_1]:.1f} °C'+f'\n        t = {t_max_delta_1:.1f} s')
_,time_2,temp_C_2,temp_B_2 = lector_templog(temps_080[1])
delta_CB_2 = temp_C_2 - temp_B_2
indx_max_DT_2 = np.argmax(delta_CB_2)
t_max_delta_2 = time_2[indx_max_DT_2]
ax2.plot(time_2,temp_C_2,'C1-',label='Centro')
ax2.plot(time_2,temp_B_2,'C1-.',label='Borde')
ax2.vlines(t_max_delta_2,temp_B_2[indx_max_DT_2],temp_C_2[indx_max_DT_2],color='C1',ls='--',label='ΔT$_{max}$ = '+f'{delta_CB_2[indx_max_DT_2]:.1f} °C'+f'\n        t = {t_max_delta_2:.1f} s')
ax2.set_xlim(0,375)
for a in (ax1,ax2):
    a.grid()
    a.legend(title=f'f = 300 kHz   H$_0$ = {H0[5]:.0f} kA/m',loc='upper center',shadow=True,frameon=True,ncol=2)
    a.set_ylabel('T (°C)')
    a.axvline(10,color='grey',ls='--',lw=1)
ax2.set_xlabel('t (s)')
plt.suptitle(titulo_6)

#% Comparativa e/ centro y borde, diferencia de temperatura
fig62,(ax,ax2,ax3) = plt.subplots(3,1,figsize=(9,9),sharex=True,constrained_layout=True)
ax.set_title('Centro',loc='left')
ax2.set_title('Borde',loc='left')
ax3.set_title(r'$\Delta$T',loc='left')
for i,p in enumerate(temps_080):
    _,time,temp_C,temp_B = lector_templog(p)
    delta_CB = temp_C - temp_B
    ax.plot(time,temp_C,'-',label=i+1)
    ax2.plot(time,temp_B,'-.',label=i+1)
    ax3.plot(time,delta_CB,'-',label=i+1)
    indx_min = np.argmin(temp_C)
    print(f'Temp minima = {temp_C[indx_min]:.1f} C ({temp_C[indx_min]+273.15:.1f} K) alcanzada en {time[indx_min]:.1f} s')
for a in (ax,ax2,ax3):
    a.grid()
    a.legend(title=f'f = 300 kHz   H$_0$ = {H0[5]:.0f} kA/m',loc='upper center',shadow=True,frameon=True,ncol=2)
    a.set_ylabel('T (°C)')
    a.axvline(10,color='grey',ls='--',lw=1)
ax3.set_xlabel('t (s)')
ax3.set_xlim(0,375)
plt.suptitle(titulo_6)

#%% 7 idc = 075 dA
titulo_7 = 'M22 - EG51% + FF49% --> LN2 --> 300 kHz & 075 dA'
fig71,(ax1,ax2) = plt.subplots(2,1,figsize=(9,6),sharex=True,constrained_layout=True)
_,time_1,temp_C_1,temp_B_1 = lector_templog(temps_075[0])
delta_CB_1 = temp_C_1 - temp_B_1
indx_max_DT_1 = np.argmax(delta_CB_1)
t_max_delta_1 = time_1[indx_max_DT_1]
ax1.plot(time_1,temp_C_1,'C0-',label='Centro')
ax1.plot(time_1,temp_B_1,'C0-.',label='Borde')
ax1.vlines(t_max_delta_1,temp_B_1[indx_max_DT_1],temp_C_1[indx_max_DT_1],color='C0',ls='--',label='ΔT$_{max}$ = '+f'{delta_CB_1[indx_max_DT_1]:.1f} °C'+f'\n        t = {t_max_delta_1:.1f} s')
_,time_2,temp_C_2,temp_B_2 = lector_templog(temps_075[1])
delta_CB_2 = temp_C_2 - temp_B_2
indx_max_DT_2 = np.argmax(delta_CB_2)
t_max_delta_2 = time_2[indx_max_DT_2]
ax2.plot(time_2,temp_C_2,'C1-',label='Centro')
ax2.plot(time_2,temp_B_2,'C1-.',label='Borde')
ax2.vlines(t_max_delta_2,temp_B_2[indx_max_DT_2],temp_C_2[indx_max_DT_2],color='C1',ls='--',label='ΔT$_{max}$ = '+f'{delta_CB_2[indx_max_DT_2]:.1f} °C'+f'\n        t = {t_max_delta_2:.1f} s')
ax2.set_xlim(0,450)
for a in (ax1,ax2):
    a.grid()
    a.legend(title=f'f = 300 kHz   H$_0$ = {H0[6]:.0f} kA/m',loc='upper center',shadow=True,frameon=True,ncol=2)
    a.set_ylabel('T (°C)')
    a.axvline(10,color='grey',ls='--',lw=1)
ax2.set_xlabel('t (s)')
plt.suptitle(titulo_7)

#% Comparativa e/ centro y borde, diferencia de temperatura
fig72,(ax,ax2,ax3) = plt.subplots(3,1,figsize=(9,9),sharex=True,constrained_layout=True)
ax.set_title('Centro',loc='left')
ax2.set_title('Borde',loc='left')
ax3.set_title(r'$\Delta$T',loc='left')
for i,p in enumerate(temps_075):
    _,time,temp_C,temp_B = lector_templog(p)
    delta_CB = temp_C - temp_B
    ax.plot(time,temp_C,'-',label=i+1)
    ax2.plot(time,temp_B,'-.',label=i+1)
    ax3.plot(time,delta_CB,'-',label=i+1)
    indx_min = np.argmin(temp_C)
    print(f'Temp minima = {temp_C[indx_min]:.1f} C ({temp_C[indx_min]+273.15:.1f} K) alcanzada en {time[indx_min]:.1f} s')
for a in (ax,ax2,ax3):
    a.grid()
    a.legend(title=f'f = 300 kHz   H$_0$ = {H0[6]:.0f} kA/m',loc='upper center',shadow=True,frameon=True,ncol=2)
    a.set_ylabel('T (°C)')
    a.axvline(10,color='grey',ls='--',lw=1)
ax3.set_xlabel('t (s)')
ax3.set_xlim(0,450)
plt.suptitle(titulo_7)


#%% 8 idc = 060 dA
titulo_8 = 'M22 - EG51% + FF49% --> LN2 --> 300 kHz & 060 dA'
fig81,(ax1,ax2) = plt.subplots(2,1,figsize=(9,6),sharex=True,constrained_layout=True)
_,time_1,temp_C_1,temp_B_1 = lector_templog(temps_060[0])
delta_CB_1 = temp_C_1 - temp_B_1
indx_max_DT_1 = np.argmax(delta_CB_1)
t_max_delta_1 = time_1[indx_max_DT_1]
ax1.plot(time_1,temp_C_1,'C0-',label='Centro')
ax1.plot(time_1,temp_B_1,'C0-.',label='Borde')
ax1.vlines(t_max_delta_1,temp_B_1[indx_max_DT_1],temp_C_1[indx_max_DT_1],color='C0',ls='--',label='ΔT$_{max}$ = '+f'{delta_CB_1[indx_max_DT_1]:.1f} °C'+f'\n        t = {t_max_delta_1:.1f} s')
# _,time_2,temp_C_2,temp_B_2 = lector_templog(temps_060[1])
# delta_CB_2 = temp_C_2 - temp_B_2
# indx_max_DT_2 = np.argmax(delta_CB_2)
# t_max_delta_2 = time_2[indx_max_DT_2]
# ax2.plot(time_2,temp_C_2,'C1-',label='Centro')
# ax2.plot(time_2,temp_B_2,'C1-.',label='Borde')
# ax2.vlines(t_max_delta_2,temp_B_2[indx_max_DT_2],temp_C_2[indx_max_DT_2],color='C1',ls='--',label='ΔT$_{max}$ = '+f'{delta_CB_2[indx_max_DT_2]:.1f} °C'+f'\n        t = {t_max_delta_2:.1f} s')
ax2.set_xlim(0,400)
for a in (ax1,ax2):
    a.grid()
    a.legend(title=f'f = 300 kHz   H$_0$ = {H0[7]:.0f} kA/m',loc='upper center',shadow=True,frameon=True,ncol=2)
    a.set_ylabel('T (°C)')
    a.axvline(10,color='grey',ls='--',lw=1)
ax2.set_xlabel('t (s)')
plt.suptitle(titulo_8)

#% Comparativa e/ centro y borde, diferencia de temperatura
fig82,(ax,ax2,ax3) = plt.subplots(3,1,figsize=(9,9),sharex=True,constrained_layout=True)
ax.set_title('Centro',loc='left')
ax2.set_title('Borde',loc='left')
ax3.set_title(r'$\Delta$T',loc='left')
for i,p in enumerate(temps_060):
    _,time,temp_C,temp_B = lector_templog(p)
    delta_CB = temp_C - temp_B
    ax.plot(time,temp_C,'-',label=i+1)
    ax2.plot(time,temp_B,'-.',label=i+1)
    ax3.plot(time,delta_CB,'-',label=i+1)
    indx_min = np.argmin(temp_C)
    print(f'Temp minima = {temp_C[indx_min]:.1f} C ({temp_C[indx_min]+273.15:.1f} K) alcanzada en {time[indx_min]:.1f} s')
for a in (ax,ax2,ax3):
    a.grid()
    a.legend(title=f'f = 300 kHz   H$_0$ = {H0[7]:.0f} kA/m',loc='upper center',shadow=True,frameon=True,ncol=2)
    a.set_ylabel('T (°C)')
    a.axvline(10,color='grey',ls='--',lw=1)
ax3.set_xlabel('t (s)')
ax3.set_xlim(0,500)
plt.suptitle(titulo_8)
#%% Comparo enfriamiento/calentamiento para cada campo
import warnings

labels = [f'{H0[i]:.1f}' for i in range(len(H0))]
paths_300kHz = [temps_150[0],temps_125[0],temps_100[0],temps_090[0],temps_085[0],temps_080[0],temps_075[0],temps_060[0]]

def separa_enfriamiento_calentamiento(t,T_C,T_B,delta_T_salto=0.8,tiempo_previo=5):
    # Enfriamiento: desde el inicio hasta el mínimo de cada canal
    i_min_C = np.argmin(T_C)
    i_min_B = np.argmin(T_B)
    t_enf_C = t[:i_min_C+1] - t[0]
    T_enf_C = T_C[:i_min_C+1]
    t_enf_B = t[:i_min_B+1] - t[0]
    T_enf_B = T_B[:i_min_B+1]

    # Calentamiento: detectar el primer salto positivo de CH1 posterior al mínimo
    candidatos = np.where((np.diff(T_C) >= delta_T_salto) & (np.arange(len(T_C)-1) >= i_min_C))[0]
    if len(candidatos) == 0:
        warnings.warn('No se encontró un salto positivo en CH1 de al menos '
                      f'{delta_T_salto:.1f} °C después del mínimo. Se omite el calentamiento.')
        t_cal_C = T_cal_C = t_cal_B = T_cal_B = None
    else:
        i_salto = candidatos[0]
        t_salto = t[i_salto]
        i_inicio = np.searchsorted(t, t_salto-tiempo_previo, side='left')
        t_cal_C = t[i_inicio:] - t_salto
        T_cal_C = T_C[i_inicio:]
        t_cal_B = t[i_inicio:] - t_salto
        T_cal_B = T_B[i_inicio:]
        print(f'Salto CH1: ΔT = {T_C[i_salto+1]-T_C[i_salto]:.2f} °C; '
              f't salto = {t_salto:.1f} s; '
              f'inicio del calentamiento = {t[i_inicio]:.1f} s; '
              f'tiempo previo conservado = {t_salto-t[i_inicio]:.1f} s\n')

    return (t_enf_C,T_enf_C),(t_enf_B,T_enf_B),(t_cal_C,T_cal_C),(t_cal_B,T_cal_B)

#%% Enfriamiento Centro y Borde
fig_enf,(ax,bx) = plt.subplots(2,1,figsize=(9,6),constrained_layout=True,sharex=True)
for i,p in enumerate(paths_300kHz):
    _,time,temp_C,temp_B = lector_templog(p,sincronizo_CH1_t=10)
    (t_enf_C,T_enf_C),(t_enf_B,T_enf_B),_,_ = separa_enfriamiento_calentamiento(time,temp_C,temp_B)
    ax.plot(t_enf_C,T_enf_C,'-',label=labels[i])
    bx.plot(t_enf_B,T_enf_B,'-',label=labels[i])
ax.set_title('Centro (CH1)',loc='left')
bx.set_title('Borde (CH2)',loc='left')
bx.set_xlabel('t desde el inicio de la curva (s)')
bx.set_xlim(0,120)
for a in (ax,bx):
    a.grid()
    a.legend(title='H$_0$ (kA/m)',loc='best',shadow=True,frameon=True,ncol=3)
    a.set_ylabel('T (°C)')
plt.suptitle('M22 - EG51% + FF49%\nEnfriamiento LN$_2$')

#%% Calentamiento Centro y Borde
fig_cal,(ax,bx) = plt.subplots(2,1,figsize=(9,6),constrained_layout=True,sharex=True)
for i,p in enumerate(paths_300kHz):
    _,time,temp_C,temp_B = lector_templog(p,sincronizo_CH1_t=10)
    _,_,(t_cal_C,T_cal_C),(t_cal_B,T_cal_B) = separa_enfriamiento_calentamiento(time,temp_C,temp_B)
    if t_cal_C is None:
        continue
    ax.plot(t_cal_C,T_cal_C,'-',label=labels[i])
    bx.plot(t_cal_B,T_cal_B,'-',label=labels[i])
ax.set_title('Centro (CH1)',loc='left')
bx.set_title('Borde (CH2)',loc='left')
bx.set_xlabel('t (s)')
bx.set_xlim(0,350)
for a in (ax,bx):
    a.grid()
    a.legend(title='H$_0$ (kA/m)',loc='best',shadow=True,frameon=True,ncol=3)
    a.set_ylabel('T (°C)')
bx.set_xlim(-5,)
# bx.set_ylim(-200,-140)
plt.suptitle('M22 - EG51% + FF49%\nCalentamiento RF')

#%% Calentamiento Centro y Borde sin Transicion de Fase
fig_cal_sin_TF, (ax,bx,cx) = plt.subplots(3,1,figsize=(10,9),sharex=True,constrained_layout=True)
ax.set_title('Centro (CH1)',loc='left')
bx.set_title('Borde (CH2)',loc='left')
cx.set_title('Diferencia de Temperatura',loc='left')

ainset=ax.inset_axes([0.5,0.11,0.48,0.43])
ainset.grid()
for i,p in enumerate(paths_300kHz[:4]):
    _,time,temp_C,temp_B = lector_templog(p,sincronizo_CH1_t=10)
    _,_,(t_cal_C,T_cal_C),(t_cal_B,T_cal_B) = separa_enfriamiento_calentamiento(time,temp_C,temp_B)
    if t_cal_C is None:
        continue
    ax.plot(t_cal_C,T_cal_C,'-',label=labels[i])
    bx.plot(t_cal_B,T_cal_B,'-',label=labels[i])
    cx.plot(t_cal_C,T_cal_C-T_cal_B,'-',label=labels[i])
    indx_max_T_C = np.argmax(T_cal_C)
    ax.axhspan(ymin=-60,ymax=-40,color='tab:purple',alpha=0.1)
    bx.axhspan(ymin=-60,ymax=-40,color='tab:purple',alpha=0.1)
    ainset.axhspan(ymin=-60,ymax=-40,color='tab:purple',alpha=0.1)
    ainset.plot(t_cal_C,T_cal_C,'-')
    ainset.set_xlim(10,120)
    ainset.set_ylim(-70,-30)
    
    # ax.axvline(t_cal_C[indx_max_T_C],c=C[i],ls='--',lw=0.8)
    # bx.axvline(t_cal_C[indx_max_T_C],c=C[i],ls='--',lw=0.8)
    cx.axvline(t_cal_C[indx_max_T_C],c=C[i],ls='--',lw=0.8)
cx.set_xlabel('t (s)')
cx.set_xlim(-5,140)
for a in (ax,bx,cx):
    a.grid()
    a.legend(title='H$_0$ (kA/m)',loc='upper right',shadow=True,frameon=True,ncol=2)
    a.set_ylabel('T (°C)')
plt.suptitle('M22 - EG51% + FF49%\nCalentamiento RF - Sin TF')

#%% Calentamiento Centro y Borde CON Transicion de Fase
fig_cal_con_TF, (ax,bx,cx) = plt.subplots(3,1,figsize=(10,9),sharex=True,constrained_layout=True)
ax.set_title('Centro (CH1)',loc='left')
bx.set_title('Borde (CH2)',loc='left')
cx.set_title('Diferencia de Temperatura',loc='left')

ainset=ax.inset_axes([0.5,0.11,0.48,0.43])
ainset.grid()
for i,p in enumerate(paths_300kHz[4:]):
    _,time,temp_C,temp_B = lector_templog(p,sincronizo_CH1_t=10)
    _,_,(t_cal_C,T_cal_C),(t_cal_B,T_cal_B) = separa_enfriamiento_calentamiento(time,temp_C,temp_B)
    if t_cal_C is None:
        continue
    ax.plot(t_cal_C,T_cal_C,'-',label=labels[i])
    bx.plot(t_cal_B,T_cal_B,'-',label=labels[i])
    cx.plot(t_cal_C,T_cal_C-T_cal_B,'-',label=labels[i])
    indx_max_T_C = np.argmax(T_cal_C)
    ax.axhspan(ymin=-60,ymax=-40,color='tab:purple',alpha=0.1)
    bx.axhspan(ymin=-60,ymax=-40,color='tab:purple',alpha=0.1)
    ainset.axhspan(ymin=-60,ymax=-40,color='tab:purple',alpha=0.1)
    # bx.axvline(t_cal_C[indx_max_T_C],c=C[i],ls='--',lw=0.8)
    cx.axvline(t_cal_C[indx_max_T_C],c=C[i],ls='--',lw=0.8)
    ainset.plot(t_cal_C,T_cal_C,'-')
    ainset.set_xlim(110,310)
    ainset.set_ylim(-70,-30)
    
cx.set_xlabel('t (s)')
cx.set_xlim(-5,400)

for a in (ax,bx,cx):
    a.grid()
    a.legend(title='H$_0$ (kA/m)',loc='upper left',shadow=True,frameon=True,ncol=2)
    a.set_ylabel('T (°C)')
plt.suptitle('M22 - EG51% + FF49%\nCalentamiento RF - Con TF')


#%% Salvo figuras de templogs
fig11.savefig('M22_EG51_FF49_LN2_300kHz_150dA_1',dpi=300)
fig12.savefig('M22_EG51_FF49_LN2_300kHz_150dA_2',dpi=300)
fig21.savefig('M22_EG51_FF49_LN2_300kHz_125dA_1',dpi=300)
fig22.savefig('M22_EG51_FF49_LN2_300kHz_125dA_2',dpi=300)
fig31.savefig('M22_EG51_FF49_LN2_300kHz_100dA_1',dpi=300)
fig32.savefig('M22_EG51_FF49_LN2_300kHz_100dA_2',dpi=300)
fig41.savefig('M22_EG51_FF49_LN2_300kHz_090dA_1',dpi=300)
fig42.savefig('M22_EG51_FF49_LN2_300kHz_090dA_2',dpi=300)
fig51.savefig('M22_EG51_FF49_LN2_300kHz_085dA_1',dpi=300)
fig52.savefig('M22_EG51_FF49_LN2_300kHz_085dA_2',dpi=300)
fig61.savefig('M22_EG51_FF49_LN2_300kHz_080dA_1',dpi=300)
fig62.savefig('M22_EG51_FF49_LN2_300kHz_080dA_2',dpi=300)
fig71.savefig('M22_EG51_FF49_LN2_300kHz_075dA_1',dpi=300)
fig72.savefig('M22_EG51_FF49_LN2_300kHz_075dA_2',dpi=300)   
fig81.savefig('M22_EG51_FF49_LN2_300kHz_060dA_1',dpi=300)
fig82.savefig('M22_EG51_FF49_LN2_300kHz_060dA_2',dpi=300)
fig_enf.savefig('M22_EG51_FF49_LN2_300kHz_Enfriamiento',dpi=300)
fig_cal.savefig('M22_EG51_FF49_LN2_300kHz_Calentamiento',dpi=300)
fig_cal_sin_TF.savefig('M22_EG51_FF49_LN2_300kHz_Calentamiento_sin_TF',dpi=300)
fig_cal_con_TF.savefig('M22_EG51_FF49_LN2_300kHz_Calentamiento_con_TF',dpi=300)

#%% Curvatura
fig1,(ax,ax2,ax3) = plt.subplots(3,1,figsize=(10,10),constrained_layout=True)
ax.set_title('Temperatura vs tiempo',loc='left')
ax2.set_title('Curvatura vs temperatura: −180 a −100 °C',loc='left')
ax3.set_title('Curvatura vs temperatura: −100 a 0 °C',loc='left')

for i,p in enumerate(paths_300kHz):
    _,time,temp,_ = lector_templog(p)
    indx_min = np.argmin(temp)
    indx_max = indx_min + np.argmax(temp[indx_min:])
    print('\n',os.path.basename(p))
    print(f'Idc = {Idc_values[i]:.1f} A; H0 = {H0[i]:.1f} kA/m')
    print(f'T min = {temp[indx_min]:.1f} °C ({temp[indx_min]+273.15:.1f} K) alcanzada en {time[indx_min]:.1f} s')
    print(f'T max = {temp[indx_max]:.1f} °C ({temp[indx_max]+273.15:.1f} K) alcanzada en {time[indx_max]:.1f} s')

    t_curv = time[indx_min:indx_max+1]
    T_curv = temp[indx_min:indx_max+1]
    mask = T_curv < 10
    T_curv = T_curv[mask]
    t_curv = t_curv[mask]
    _,_,curv = curvatura(T_curv)

    label = f'{H0[i]:.1f}'
    ax.plot(t_curv-t_curv[0],T_curv,'.-',label=label)
    ax2.plot(T_curv,curv,'.-',label=label)
    ax3.plot(T_curv,curv,'.-',label=label)

# Zonas de interés
# ax.axhspan(-150,-75,color='tab:red',alpha=0.2,zorder=-1)
# ax.axhspan(-75,-50,color='tab:purple',alpha=0.2,zorder=-1)
# ax2.axvspan(-150,-75,color='tab:red',alpha=0.2,zorder=-1)
# ax2.axvspan(-75,-50,color='tab:purple',alpha=0.2,zorder=-1)
# ax3.axvspan(-75,-50,color='tab:purple',alpha=0.2,zorder=-1)

# Referencia de curvatura
ax2.axhline(0.02,color='k',ls='--',lw=1,label='Curvatura = 0.02')
ax3.axhline(0.02,color='k',ls='--',lw=1,label='Curvatura = 0.02')

# Límites de los ejes
ax2.set_ylim(0,0.05)
ax3.set_ylim(0,0.05)
ax2.set_xlim(-180,-100)
ax3.set_xlim(-100,0)

# Etiquetas
ax.set_xlabel('t (s)')
ax.set_ylabel('T (°C)')
ax2.set_ylabel('Curvatura')
ax3.set_ylabel('Curvatura')
ax3.set_xlabel('T (°C)')

for a in [ax,ax2,ax3]:
    a.grid()
    a.legend(title='H₀ (kA/m)',ncol=2,shadow=True,frameon=True)

plt.suptitle('EG 51% – FF 49% | LN₂ → RF\nf = 300 kHz | Idc = '+str(Idc_values)+' A')
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
        print(f'T min = {temp[indx_min]:.1f} C ({temp[indx_min]+273.15:.1f} K) alcanzada en {time[indx_min]:.1f} s')
        print(f'T max = {temp[indx_max]:.1f} C ({temp[indx_max]+273.15:.1f} K) alcanzada en {time[indx_max]:.1f} s')

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
        print(f'T min = {temp[indx_min]:.1f} C ({temp[indx_min]+273.15:.1f} K) alcanzada en {time[indx_min]:.1f} s')
        print(f'T max = {temp[indx_max]:.1f} C ({temp[indx_max]+273.15:.1f} K) alcanzada en {time[indx_max]:.1f} s')

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


fig1.savefig('EG51_FF49_curvatura_150_125_075.png',dpi=300)
fig2.savefig('EG51_FF49_curvatura_120_110_100.png',dpi=300)
fig3.savefig('EG51_FF49_curvatura_090_080_060.png',dpi=300)

# %%
