import numpy as np
import pandas
from astropy.cosmology import Planck15 as cosmo
import pickle
import astropy.units as u
import matplotlib.pyplot as plt
import astropy as ap
import os 
from astropy.table import Table
import argparse
from astropy.io import fits
import astropy.constants as cst
from IPython import embed
import scipy.constants as cst

def make_log_bin(min_logbin_value, max_logbin_value, nbins):

    log_bins = np.linspace(min_logbin_value, max_logbin_value, nbins)
    Deltabin = (max_logbin_value - min_logbin_value) * 1. / nbins #in dex
    log_mean = log_bins[:-1] + Deltabin / 2

    return log_bins, Deltabin, log_mean

def B_from_rhoL(cat, line, nu_rest, z, dz, Vslice): 

    nu_obs = nu_rest.value /(1+cat['redshift'])
    L = cat[f'I{line}'] * (1.04e-3) * nu_obs * cosmo.luminosity_distance(cat['redshift'])**2
    rhoL = np.sum(L) / Vslice 
    B_to_rhoL_conv = ((4*np.pi*nu_rest.to(u.Hz)*cosmo.H(z))/(4e7 *cst.c*1e-3)).value #Lsolar/Mpc3

    return (rhoL/B_to_rhoL_conv).value
     
def B_and_sn(cat, line, nu_rest, z, dz, field_size, log_bins):
    
    nu_obs = nu_rest /(1+cat['redshift'])
    dnu=dz*nu_obs/(1+z)
    vdelt = (cst.c * 1e-3) * dnu / nu_obs #km/s
    S = cat['I'+line] / vdelt  #Jy <-- Jy.km/s
    B = np.sum(S) / field_size #background intensity 
    shot_noise = np.sum(S**2) / field_size

    S = S.loc[S>0]
    histo, edges = np.histogram(np.log10(S), bins = log_bins, weights=S**2) 
    dndS_Ssquare = histo/field_size #Contribution to shot noise
    histo, edges = np.histogram(np.log10(S), bins = log_bins, weights=S) 
    dndS_S = histo/field_size #Contribution to background intensity

    return B.value, dndS_S.value, shot_noise.value, dndS_Ssquare.value
    
log_S_bins, S_Deltabin, log_S_mean = make_log_bin(-10, -3, 28)

cat = Table.read("pySIDES_from_uchuu_tile_119_0.2deg_x_1deg.fits")
cat = cat.to_pandas()

freq_CII = 1900.53690000 * u.GHz
zmean = 2
dz = 0.05

zbins = (zmean-dz/2,zmean+dz/2 )
Dc_bins = cosmo.comoving_distance(zbins)
tile_size = (0.2 * 1*u.deg**2).to(u.sr)
cat_bin = cat.loc[ (cat['redshift'] > zbins[0]) & (cat['redshift'] < zbins[0+1])]
Vslice = tile_size.value / 3 * (Dc_bins[0+1]**3-Dc_bins[0]**3)

B, B_contrib, shot_noise, shot_noise_contrib = B_and_sn(cat_bin, 'CII_de_Looze', freq_CII, zmean, dz, tile_size, log_S_bins) 
Bprim = B_from_rhoL(cat_bin, 'CII_de_Looze', freq_CII, zmean, dz, Vslice)

BS = 10; plt.rc('font', size=BS); plt.rc('axes', titlesize=BS); plt.rc('axes', labelsize=BS); mk = 3; lw=1
fig, ax = plt.subplots(figsize=(4,4), dpi=200) 

ax.set_xlim(-13,-6); ax.set_ylim(-15,4)
ax.set_xlabel("ln($\\rm S_{\\nu} $) [mJy]")
ax.set_ylabel('log(dn/dS $\\rm S^3$) [$\\rm Jy^2$/sr]', color='b')
ax.axhline(np.log10(shot_noise), -15,5, c='b', ls=':' )
x = log_S_mean - 3
y = np.log10(shot_noise_contrib)
w = np.where(~np.isinf(y))[0]
ax.plot(x[w], y[w], c='b') 
ax.tick_params('y', colors='blue')
ax.set_title(f'[CII]@z={np.round(zmean,1)}'+'$\\rm \\pm$'+f'{dz}')

ax2 = ax.twinx()
ax2.set_ylabel('log(dn/dS $\\rm S^2$) [$\\rm Jy$/sr]', color='r')
ax2.axhline(np.log10(Bprim), -15,5, c='grey', ls='solid' )
ax2.axhline(np.log10(B), -15,5, c='r', ls=':' )
y = np.log10(B_contrib)
w = np.where(~np.isinf(y))[0]
ax2.plot(x[w], y[w], c='r') 
ax2.tick_params('y', colors='r')

fig.tight_layout()
fig.savefig(f"figures/contrib_shotnoise_and_background.png", transparent=True)

plt.show()
