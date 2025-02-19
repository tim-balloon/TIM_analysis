from pysides.gen_fluxes import gen_Snu_arr
from astropy.io import fits
import astropy.units as u
import scipy.constants as cst
import numpy as np
from astropy import wcs
from astropy.stats import gaussian_fwhm_to_sigma
import astropy.convolution as conv
import datetime
from astropy.table import Table
import pickle
from copy import deepcopy
import os
from IPython import embed
import pickle
import matplotlib.pyplot as plt
from set_k import *
from powspec import power_spectral_density
from pysides.gen_fluxes import gen_Snu_arr
from pysides.load_params import *
#-----------------------------------------------
cube_prop_dict = pickle.load( open('cubeprop.p', 'rb'))
w = cube_prop_dict['w']
cat = Table.read('pySIDES_from_uchuu_tile_0_1deg_x_1deg.fits')
cat = cat.to_pandas()
reduction = 200
cat = cat[:50]#5468483//reduction]
cube_prop_dict['pos'][0] = cube_prop_dict['pos'][0][:50]#5468483//reduction]
cube_prop_dict['pos'][1] = cube_prop_dict['pos'][1][:50]#5468483//reduction]
params_sides = load_params('PAR_FILES/SIDES_from_original_with_fir_lines.par')
SED_dict = pickle.load(open(params_sides['SED_file'], "rb"))
spf = 10
res = (cube_prop_dict['w'].wcs.cdelt[0]*u.deg).to(u.arcsec)
pixel_sr = ((res**2)).to(u.sr).value #solid angle of the pixel in sr 
nudelt = abs(cube_prop_dict['w'].wcs.cdelt[2]) * 1e-9 #GHz
#-----------------------------------------------
profile='tophat'
z = np.arange(0,cube_prop_dict['shape'][0],1) #
channels = w.swapaxes(0, 2).sub(1).wcs_pix2world(z, 0)[0] / 1e9 #GHz
freq_list = channels
lambda_list =  ( cst.c * (u.m/u.s)  / (np.asarray(freq_list*1e9) * u.Hz)  ).to(u.um).value
Snu_arr = gen_Snu_arr(lambda_list, SED_dict, cat["redshift"], cat['mu']*cat["LIR"], cat["Umean"], cat["Dlum"], cat["issb"])
continuum_nobeam_Jypix = []
for f in range(len(z)):      
    row = Snu_arr[:,f] #Jy/pix
    histo, y_edges, x_edges = np.histogram2d(cube_prop_dict['pos'][0], cube_prop_dict['pos'][1], bins=(cube_prop_dict['y_edges'], cube_prop_dict['x_edges']), weights=row)
    continuum_nobeam_Jypix.append(histo) #Jy/pix, no beam
continuum_nobeam_Jysr = np.asarray(continuum_nobeam_Jypix) / pixel_sr

plt.imshow(continuum_nobeam_Jysr[0], origin='lower')
plt.colorbar()
#-----------------------------------------------
fwhm = w.wcs.cdelt[2]/1e9 
freq_listG = np.linspace(channels.min()-2*fwhm, channels.max()+2*fwhm, spf*len(channels)+1)
lambda_list =  ( cst.c * (u.m/u.s)  / (np.asarray(freq_listG*1e9) * u.Hz)  ).to(u.um).value
Snu_arrG = gen_Snu_arr(lambda_list, SED_dict, cat["redshift"], cat['mu']*cat["LIR"], cat["Umean"], cat["Dlum"], cat["issb"])
dnu = np.diff(freq_listG).mean()
sigma = fwhm * gaussian_fwhm_to_sigma # Convert FWHM to sigma 
transmission = np.exp(-((freq_listG[:, None] - channels) ** 2) / (2 * (sigma)**2))
Snu_arr_transmitted = Snu_arrG[:,:,np.newaxis] * transmission *(dnu/nudelt)
Snu_arrG = np.sum(Snu_arr_transmitted , axis=1) #* dnu 
continuum_nobeam_gaussian = []
for f in range(0, len(z)):      
    row = Snu_arrG[:,f] #Jy/pix
    histo, y_edges, x_edges = np.histogram2d(cube_prop_dict['pos'][0], cube_prop_dict['pos'][1], bins=(cube_prop_dict['y_edges'], cube_prop_dict['x_edges']), weights=row)
    continuum_nobeam_gaussian.append(histo) #Jy/pix, no beam

continuum_nobeam_gaussian = np.asarray(continuum_nobeam_gaussian) / pixel_sr
plt.figure()
plt.imshow(continuum_nobeam_gaussian[0], origin='lower')
plt.colorbar()
#-----------------------------------------------
j=3
fig, (ax1, ax2) = plt.subplots(2, figsize=(3,6))
#----
ax1.vlines(channels, ymin=0, ymax=1,color='k')
ax1.vlines(channels-nudelt/2, ymin=0, ymax=1,color='k', ls=':')
ax1.vlines(channels+nudelt/2, ymin=0, ymax=1,color='k', ls=':')
ax1.set_ylim(0,1)
for i in range(len(z)): ax1.plot(freq_listG, transmission[:,i], '-or')
#----
#for i in range(100): ax2.plot(freq_listG, Snu_arr_transmitted[i, :,0])
for i in range(len(z)): ax2.plot(freq_listG, Snu_arr_transmitted[j, :,i] , ':og', markersize=1)
ax2.plot(channels,   Snu_arrG[j,:]               , 'or')
ax2.plot(channels, Snu_arr[j,:], 'xk')
print('the ratio is:',Snu_arrG[j,:] / Snu_arr[j,:], 'and squared it is:', (Snu_arrG[j,:] / Snu_arr[j,:])**2)
ax2.set_yscale('log') 
#-----------------------------------------------
k_nyquist, k_min, k_bin_width, k_bin_tab, k_out, k_map = set_k_infos(cube_prop_dict['shape'][1], cube_prop_dict['shape'][0], res.to(u.rad), delta_k_over_k = 0.05)
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(6,3))
for f in range(len(z)):      
    pk, e  =  power_spectral_density(continuum_nobeam_Jysr[f,:,:], res.to(u.rad), bins = k_bin_tab)
    ax1.loglog(k_out.to(1/u.arcmin), pk)
    pkg, e =  power_spectral_density(continuum_nobeam_gaussian[f,:,:], res.to(u.rad), bins = k_bin_tab)
    ax2.loglog(k_out.to(1/u.arcmin), pkg,)    
print('The pk ratio is:',pkg/pk)
plt.show()