import numpy as np
import pickle
from astropy.cosmology import Planck15 as cosmo
import astropy.units as u
import scipy.constants as cst
from astropy.io import fits
import datetime
from astropy.table import Table
import os
from IPython import embed
from astropy.stats import gaussian_fwhm_to_sigma
from astropy import wcs
import astropy.convolution as conv
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import pandas as pd
from pysides.gen_fluxes import *
from pysides.load_params import *
from pysides.gen_lines import *
#---
import os
import time
import vaex as vx
import matplotlib
#---
from pysides.make_cube import *

freq_CII = 1900.53690000 * u.GHz
freq_CI10 = 492.16 *u.GHz
freq_CI21 = 809.34 * u.GHz
rest_freq_list = [115.27120180  *u.GHz* J_up for J_up in range(1, 9)]
rest_freq_list.append(freq_CI10); rest_freq_list.append(freq_CI21); rest_freq_list.append(freq_CII); 
line_list = ["CO{}{}".format(J_up, J_up - 1) for J_up in range(1, 9)]
line_list.append('CI10'); line_list.append('CI21'); line_list.append('CII_de_Looze')
fir_lines_list = ['NeII13', 'NeIII16', 'H2_17', 'SIII19', 'OIV26', 'SIII33', 'SiII35', 'OIII52', 'NIII57', 'OI63', 'OIII88', 'NII122', 'OI145','NII205'] #Do not add CII which is dealt with differently!                     
fir_lines_nu =   [23403.00218579, 19279.2577492 , 17603.7849677 , 16023.11373597, 11579.46921591,  8954.37449223,  8609.77765652,  5786.382127  , 5230.15453594,  4746.55569981,  3392.85262562,  2459.33107465, 2060.4292646, 1462.4022342]
for l, nu in zip(fir_lines_list, fir_lines_nu):
    line_list.append(l)
    rest_freq_list.append(nu*u.GHz)


def gen_spt_sed(cat, file):

    params = load_params('PAR_FILES/SIDES_from_original_with_fir_lines.par')

    cat = gen_fluxes(cat, params)
    cat = gen_CO(cat, params)
    cat = gen_CII(cat, params)
    cat = gen_CI(cat, params)
    cat = gen_fir_lines(cat, params)

    res = file['pixel_size']
    pixel_sr = (res.value * np.pi/180/3600)**2 #solid angle of the pixel in sr 
    params_cube = {'freq_min':file["freq_min"].value, 'freq_max':file['freq_max'].value, 'freq_resol':file["freq_resol"].value, "pixel_size":res.value}
    cube_prop_dict = set_wcs(cat, params_cube)
    z = np.arange(0,cube_prop_dict['shape'][0],1)
    w = cube_prop_dict['w']
    freq_obs = w.swapaxes(0, 2).sub(1).wcs_pix2world(z, 0)[0]
    #Continuum  
    lambda_list =  ( cst.c * (u.m/u.s)  / (np.asarray(freq_obs) * u.Hz)  ).to(u.um)
    SED_dict = pickle.load(open('pysides/SEDfiles/SED_finegrid_dict.p', "rb"))
    print("Generate monochromatic fluxes...")
    Snu_arr = gen_Snu_arr(lambda_list.value, SED_dict, cat["redshift"], cat['mu']*cat["LIR"], cat["Umean"], cat["Dlum"], cat["issb"])# cat_line['mu']*
    Slines = np.zeros(Snu_arr.shape)
    Slines_index = np.zeros((len(cat), len(line_list)))

    for iJ, (J, rest_freq) in enumerate(zip(line_list, rest_freq_list)):
        nu_obs   = rest_freq / (1+cat['redshift'])
        S, channel = line_channel_flux_densities(J, rest_freq, cat, cube_prop_dict) 
        indices = np.round(channel).astype(int)
        valid_indices_mask = np.logical_and(indices > 0, indices < len(freq_obs)-1)
        if( (valid_indices_mask!=False).any() ):
            filtered_indices = indices[valid_indices_mask]
            filtered_addresses = np.where(valid_indices_mask)[0]
            print(f"Generate {J} fluxes...",)
            Slines[filtered_addresses, filtered_indices] += np.asarray(S)[filtered_addresses]
            Slines_index[filtered_addresses, iJ ]= filtered_indices
            print(Slines[filtered_addresses, filtered_indices].min()*1e3, Slines[filtered_addresses, filtered_indices].max()*1e3)
    return cat, freq_obs, lambda_list, Snu_arr, Slines, Slines_index
    
    
if __name__ == "__main__":

    from astropy.table import Table
    data = Table.read('analysis_fcts/smg_dndz_tables.fits', hdu=1)
    names = [name for name in data.colnames if len(data[name].shape) <= 1]
    cat = data[names].to_pandas()
    TIM = {'pixel_size':45*u.arcsec, 'freq_min':715e9*u.Hz, 'freq_max':1250e9*u.Hz, 'freq_resol':1e9*u.Hz}
    cat['issb'] = False
    cat['redshift'] = cat['z_spec']
    cat['qflag'] = True
    cat['SFR'] = cat['int_SFR']*1e3

    cat, channels, l_channels, continuum, lines = gen_spt_sed(cat, TIM) 

