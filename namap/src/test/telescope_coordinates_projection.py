from astropy.modeling import rotations
from astropy import wcs
import pygetdata as gd
import numpy as np
import os 
import matplotlib.pyplot as plt


#Load Data
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
frame2 = 1921000
offset = 0

raf = ra[frame1:frame2]
decf = (dec[frame1:frame2])
latf = (lat[frame1+offset:frame2+offset])
lstf = lst[frame1+offset:frame2+offset]

#Map parameters
crpix = np.array([50,50])
crval = np.array([132.2, -42.9])
cdelt = np.array([-0.003, 0.003])
lonpole = 180.

'''
Build a wcs for telescope coordinates rotating Celestial Sphere to native sphere
using astropy routines and then using wcs module to build it
'''

n2c = rotations.RotateCelestial2Native(lon=crval[0], lat=crval[1], lon_pole=lonpole)
new_crval = n2c(crval[0],crval[1])
new_coordinates = n2c(raf*15, decf)

w = wcs.WCS(naxis=2)
w.wcs.crpix = crpix
w.wcs.cdelt = cdelt
#w.wcs.crval = new_crval
w.wcs.crval = crval
w.wcs.ctype = ["TLON-TAN", "TLAT-TAN"]

coord = np.transpose(np.vstack((raf*15, decf)))
#coord = np.transpose(np.array([new_coordinates[0], new_coordinates[1]]))
px = w.all_world2pix(coord, 1)
px2 = w.wcs.s2p(coord, 1)

'''
Build a pixel indices based on parallactic angle rotation
'''

hour_angle = np.radians((lstf-raf)*15)

index, = np.where(hour_angle<0)
hour_angle[index] += 2*np.pi

#Compute Parallactic Angle
y_pa = np.cos(np.radians(latf))*np.sin(hour_angle)
x_pa = np.sin(np.radians(latf))*np.cos(np.radians(decf)) - np.cos(hour_angle)*np.cos(np.radians(latf))*np.sin(np.radians(decf))
pa = np.arctan2(y_pa, x_pa)

#Project coordinates on tangent plane

den = np.sin(np.radians(decf))*np.sin(np.radians(crval[1]))+np.cos(np.radians(decf))*np.cos(np.radians(crval[1]))*np.cos(np.radians(raf*15.-crval[0]))

x_proj = (np.cos(np.radians(decf))*np.sin(np.radians(raf*15.-crval[0])))/den
y_proj = (np.sin(np.radians(decf))*np.cos(np.radians(crval[1]))-np.cos(np.radians(decf))*np.sin(np.radians(crval[1]))*np.cos(np.radians(raf*15.-crval[0])))

pa = 0.
x_rot = x_proj*np.cos(pa)-y_proj*np.sin(pa)
y_rot = x_proj*np.sin(pa)+y_proj*np.cos(pa)

px_x = x_rot/np.radians(cdelt[0])+crpix[0]
px_y = y_rot/np.radians(cdelt[1])+crpix[1]