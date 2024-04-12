# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 15:40:33 2024

@author: siguerin
"""
# =============================================================================
# # importation of libraries
# =============================================================================
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter,filtfilt


def strain_rotations(e1,teta1,e2,teta2,e3,teta3):

    teta1=math.radians(teta1)
    teta2=math.radians(teta2)
    teta3=math.radians(teta3)
    M=np.array([
        [np.cos(teta1)**2,np.sin(teta1)**2,np.cos(teta1)*np.sin(teta1)],
        [np.cos(teta2)**2,np.sin(teta2)**2,np.cos(teta2)*np.sin(teta2)],
        [np.cos(teta3)**2,np.sin(teta3)**2,np.cos(teta3)*np.sin(teta3)]
    ])
    st_ar=np.array([e1,e2,e3])
    Sn=np.linalg.solve(M,st_ar)
    ex=Sn[0]
    ey=Sn[1]
    gxy=Sn[2] 
    return(ex,ey,gxy)
    
def strs_stn(ex,ey,gxy,E,mu):
    "from strains, youngs modulus,poisson ratio, return the stresses"
    Sxx=E/(1-mu**2)*(ex+mu*ey)
    Syy=E/(1-mu**2)*(ey+mu*ex)
    Sxy=E/(2*(1+mu))*gxy
    return(Sxx,Syy,Sxy)


Path = 'load_test_000.csv'

## amplification settings
volt_to_epsilon = -5000 * 1e-6
## PMMA properties
E = 2.59*1e9 #*0.82
mu = 0.35


# # =============================================================================
# #                 #########FILTER/DECIMATING PARAMETERS############
# # =============================================================================
##Choose channels than don't need high frequency_set parameters
filt_condition = False
filt_L=1  #1 if on 0 if no filter, same for decimation
dec_L=0           
CutOffFreq_L=40000
ratio_L=100 # decimation ratio


# =============================================================================
# ##correspondance strain gage-channel according to screenshot
# =============================================================================
##

gages2channels = {}
##                  1st rosette :
# gage closer to side (label cable: 11, HS6-35811)
gages2channels['R1_up45'] = 'ch1'
# gage in middle of rosette (label cable: 12, HS6-35811)
gages2channels['R1_90'] = 'ch2'
# gage closer to inside of sample (label cable: 13, HS6-35811)
gages2channels['R1_dn135'] = 'ch3'
##                  2nd rosette :
# gage closer to side (label cable: 21, HS6-35812)
gages2channels['R2_up45'] = 'ch5'
# gage in middle of rosette (label cable: 22, HS6-35812)
gages2channels['R2_90'] = 'ch6'
# gage closer to inside of sample (label cable: 23, HS6-35812)
gages2channels['R2_dn135'] = 'ch7'
##                  3rd rosette :
# gage closer to side (label cable: 11, HS6-35810)
gages2channels['R3_up45'] = 'ch9'
# gage in middle of rosette (label cable: 12, HS6-35810)
gages2channels['R3_90'] = 'ch10'
# gage closer to inside of sample (label cable: 13, HS6-35810)
gages2channels['R3_dn135'] = 'ch11'
##    135° gages with exact position along interface:
    
# gage closer to side (label cable: 11, HS6-35810)
gages2channels['R4_up45'] = 'ch13'
# gage in middle of rosette (label cable: 12, HS6-35810)
gages2channels['R4_90'] = 'ch14'
# gage closer to inside of sample (label cable: 13, HS6-35810)
gages2channels['R4_dn135'] = 'ch15'
##    135° gages with exact position along interface:
    
# gage closer to side (label cable: 11, HS6-35810)
gages2channels['R5_up45'] = 'ch17'
# gage in middle of rosette (label cable: 12, HS6-35810)
gages2channels['R5_90'] = 'ch18'
# gage closer to inside of sample (label cable: 13, HS6-35810)
gages2channels['R5_dn135'] = 'ch19'
##    135° gages with exact position along interface:

    

file = Path
header = pd.read_csv(file, skiprows=6, sep = ';',nrows = 1) 


data_temp = pd.read_csv(file, skiprows=9, sep = ';') 
data_temp = data_temp.to_numpy()
time = data_temp[:,0]
dt = (time[1]-time[0])  
Samplingrate = 1 / dt  
# time_string = pd.read_csv(file, skiprows=0, sep = ';',nrows = 1)['ASCII data file created with TiePie Multi Channel software. www.tiepie.com.'][0]
# time_absolute_event_cont = float(time_string.split(' ')[1].split(':')[0]) * 3600 + \
# float(time_string.split(' ')[1].split(':')[1]) * 60 + \
# float(time_string.split(' ')[1].split(':')[2]) + \
# float('0.' + time_string.split(' ')[2][:-1])

channels_LF = data_temp[:,:] # to filter all data but time
npts=np.size(data_temp, axis=0)
if filt_condition == True:
    Wn_L=np.float64(CutOffFreq_L)/(Samplingrate/2)
    b, a = butter(2, Wn_L)
    Ch_L_filt = filtfilt(b, a, np.transpose(channels_LF))
    Ch_L_filt=np.transpose(Ch_L_filt)
    
    if dec_L==1:
    ##undersampling if low frequencies
        time=time[0:npts:ratio_L]
        channels_LF_dec=channels_LF[0:npts:ratio_L]
        data_temp=Ch_L_filt[0:npts:ratio_L]     
        
        
        
gage_rotation = 0
ex,ey,gxy = strain_rotations(data_temp[:,1]*volt_to_epsilon,45+gage_rotation,
                              data_temp[:,2]*volt_to_epsilon,90+gage_rotation,
                              data_temp[:,3]*volt_to_epsilon,135+gage_rotation)


ex = data_temp[:,1]*volt_to_epsilon + data_temp[:,3]*volt_to_epsilon - data_temp[:,2]*volt_to_epsilon
gxy =  data_temp[:,1]*volt_to_epsilon - data_temp[:,3]*volt_to_epsilon
ey =  data_temp[:,2]*volt_to_epsilon

Sxx,Syy,Sxy = strs_stn(ex,ey,gxy,E,mu)


f1 = plt.figure(1)
ax11 = f1.add_subplot(111)
ax11.plot(time/60,Sxx, label = 'fault parallel')
ax11.plot(time/60,Syy, label = 'normal')
ax11.plot(time/60,Sxy, label = 'shear')
ax11.legend()
plt.show()
