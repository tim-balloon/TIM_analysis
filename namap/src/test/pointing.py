import pygetdata as gd
import numpy as np
import os 
import matplotlib.pyplot as plt

path = '/mnt/c/Users/gabri/Documents/GitHub/mapmaking/2012_data/'
fname = gd.dirfile(path, gd.RDONLY)

ra = fname.getdata('ra', gd.UINT32, num_frames=fname.nframes)
dec = fname.getdata('dec', gd.INT32, num_frames=fname.nframes)
lst = fname.getdata('lst', gd.UINT32, num_frames=fname.nframes)
lat = fname.getdata('lat', gd.INT32, num_frames=fname.nframes)

ra = ra*5.587935447693e-09
dec = dec*8.381903171539e-08
lat = lat*8.381903171539e-08
lst = lst*2.77777777778e-04

frame1 = 1918381
frame2 = 1922092
offset = 0

xEL_offset = np.radians(0.1163518)
EL_offset = np.radians(0.05530368)

raf = ra[frame1:frame2]
decf = np.radians(dec[frame1:frame2])
latf = np.radians(lat[frame1+offset:frame2+offset])
lstf = lst[frame1+offset:frame2+offset]

#Convert RA and DEC to AZ/EL 

hour_angle = np.radians((lstf-raf)*15)

index, = np.where(hour_angle<0)
hour_angle[index] += 2*np.pi

el_conv = np.arcsin(np.sin(decf)*np.sin(latf)+np.cos(latf)*np.cos(decf)*np.cos(hour_angle))

x_ae = -np.sin(latf)*np.cos(decf)*np.cos(hour_angle) + np.cos(latf)*np.sin(decf)
y_ae = np.cos(decf)*np.sin(hour_angle)

az_conv = np.arctan2(y_ae, x_ae)
index, = np.where(np.sin(hour_angle)>0)
az_conv[index] = 2*np.pi - az_conv[index]

#Add offset
EL_corr = el_conv+EL_offset
xEL_corr = az_conv*np.cos(EL_corr)-xEL_offset
AZ_corr = xEL_corr/np.cos(EL_corr)

#Convert AZ and EL to RA and DEC
dec_final = np.arcsin(np.sin(EL_corr)*np.sin(latf)+np.cos(latf)*np.cos(EL_corr)*np.cos(AZ_corr))

x = -np.sin(latf)*np.cos(EL_corr)*np.cos(AZ_corr)+np.cos(latf)*np.sin(EL_corr)
y = -np.cos(EL_corr)*np.sin(AZ_corr)
hour_angle_conv = np.arctan2(y, x)

index, = np.where(hour_angle_conv<0)
hour_angle_conv += 2.*np.pi


ra_final = np.radians(lstf*15)-hour_angle_conv

index, = np.where(ra_final<0)
ra_final += 2.*np.pi

#Compute Parallactic Angle
y_pa = np.cos(latf)*np.sin(hour_angle)
x_pa = np.sin(latf)*np.cos(decf) - np.cos(hour_angle)*np.cos(latf)*np.sin(decf)
pa = np.arctan2(y_pa, x_pa)

# if isinstance(pa, np.ndarray):
#     index, = np.where(pa<0)
#     pa[index] += 2*np.pi
# else:
#     if pa <= 0:
#         pa += 2*np.pi

dec_naive = decf-xEL_offset*np.sin(pa)+EL_offset*np.cos(pa)
ra_naive = np.radians(raf*15.) + (xEL_offset*np.cos(pa)+EL_offset*np.sin(pa))/np.cos(dec_naive)

plt.figure(1)
plt.plot(np.degrees(dec_naive), 'o', label = 'NaivePol')
plt.plot(np.degrees(dec_final), label = 'NaMap')
plt.plot(np.degrees(decf), label = 'Real')
plt.legend()

plt.figure(2)
plt.plot(np.degrees(ra_final), label = 'NaMap')
plt.plot(np.degrees(ra_naive), 'o', label = 'NaivePol')
plt.plot(raf*15., label = 'Real')
plt.legend()

plt.show()