import pandas as pd
import numpy as np
import datetime, sys, os, pickle
import scipy.constants as cst
from astropy.stats import gaussian_fwhm_to_sigma
import matplotlib.pyplot as plt
from specutils import Spectrum1D
from specutils.manipulation import FluxConservingResampler
from astropy.cosmology import Planck18 as cosmo
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.markers as mmarkers
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.gen_gaussian_realizations import *
from src.map_power_spectrum import *
import numpy.core.numeric
sys.modules['numpy._core.numeric'] = numpy.core.numeric
from astropy.visualization import ZScaleInterval

nu0 = 1900.53690000
telescope_diameter = 2
Omega_survey = 1 * (np.pi/180)**2 #sr
t_survey = 200 * 3600 #s
k = np.logspace(np.log10(1e-4),np.log10(1e2))
nrealizations = 10
R = 250
dnu = 4.0 #GHz
nu_min = 705
nu_max = 833
kwargs = dict( delta_k_over_k_perp=0.4,delta_k_over_k_par=0.4)

dlognu = np.log(1 + 1/R)
N = int(np.floor((np.log(nu_max) - np.log(nu_min)) / dlognu)) + 1
freqs_array = nu_min * (1 + 1/R) ** np.arange(N)
lambda_um_resampled = cst.c / (freqs_array*1e9) / 1e-6
if(R is not None): dnu = freqs_array * dlognu
else: dnu = np.ones(len(freqs_array))*dnu
zcii = nu0 / freqs_array - 1
fwhm = 1.22  * cst.c / (freqs_array * 1e9 * telescope_diameter) * u.rad
Omega_beam_resampled  = (2*np.pi*(fwhm * gaussian_fwhm_to_sigma)**2).to(u.sr)
_, model = load_TIM_noise_model(freqs_array, path='../')
redshifts = nu0 / freqs_array -1 
dz = redshifts * dnu / freqs_array
redshifts_min = redshifts - dz/2
redshifts_max = redshifts + dz/2
Dc_max = cosmo.comoving_distance(redshifts_max)
Dc_min = cosmo.comoving_distance(redshifts_min)
chi = cosmo.comoving_distance(redshifts)
dchi_over_dnu =  (cst.c*1e-3) / cosmo.H(redshifts) * (1+redshifts)**2 / nu0 * dnu
dV = Omega_survey * chi**2 * (cst.c*1e-3) / cosmo.H(redshifts) * (1+redshifts)**2 * dnu / nu0
Delta_Dc_list = cst.c*1e-3*(1+redshifts) / cosmo.H(redshifts).value * dnu / freqs_array

#----------------------------------------------------------------------------------------------------------
channel_idx = 33
ratio_survey_time_over_hitmap_time = 20
frequency_channel = 804.13 #GHz
Ndets_per_channel = 42
acquisition_frequency = 100 #Hz
dt = 1/acquisition_frequency #seconds 
ifreq = np.abs(freqs_array[:, None] - frequency_channel).argmin(axis=0).item()  # scalar
idx = np.array([ifreq,])
ind_bins = (idx,) 
hit_map = np.load("hitmap_T10.00hrs_dither31.92arcsec_steps7_altstep79.80arcsec_real_offsets.npz")['hit_map']
n0 = hit_map.shape[0]//2
fig, axs = plt.subplots(figsize=(5,5), dpi=200)
zscale = ZScaleInterval()
vmin, vmax = zscale.get_limits(np.asarray(hit_map))
im = axs.imshow(hit_map, origin='lower', cmap="RdBu_r", vmin=vmin, vmax=vmax)
cbar = fig.colorbar(im, ax=axs, orientation='vertical',)
cbar.set_label('hitcounts') 
fig.tight_layout()
hit_map_real = hit_map
#------------------------------------------------------------------------------------

#---------------------------------------------------------------------------
Fig, (axpk, axspec) = plt.subplots(1,2, figsize=(10,5), dpi=200)
axpk.set_xscale('log');axpk.set_yscale('log');axspec.set_yscale('log')
axpk.set_ylabel("$\\rm P_{noise}\, [Jy^2.Mpc^3]$")
axpk.set_xlabel('k $\\rm [Mpc^{-1}]$')
axspec.set_xlabel('frequency [GHz]')
axspec.set_ylabel('$\\rm I_{\\nu}$ [Jy/sr]')
axpk.set_title('Power spectrum')
axspec.set_title('E.M spectrum')

for ib, ibin in enumerate(ind_bins):

    #Ideal case to compare with
    #---------------------------------------------------------------------------
    res = fwhm[idx].min() / 2 #<--------------------- !!! carefull if we set FWHM / 2 or FHWM / 3
    npix_size = int(np.sqrt((Omega_survey/res.value**2) ).max())
    len_pix_bin  = cosmo.comoving_transverse_distance(redshifts[ibin]).value * res.value
    t_int = t_survey * Ndets_per_channel / npix_size**2
    NEIint_direct_sensitivity = model.value / np.sqrt(t_int)
    NEI_int = model.value/np.sqrt(t_survey) #Jy/sr
    P_mine = NEI_int**2 / Ndets_per_channel * dV 

    P_n_per_bin = P_mine[ibin]                    
    NEI = model.value[ibin]
    nus = freqs_array[ibin]
    dnus = dnu[ibin]
    Delta_Dc_bin = Delta_Dc_list[idx]
    NEIintbin = NEI_int[idx]

    cx = hit_map_real.shape[1] // 2
    cy = hit_map_real.shape[0] // 2
    n = 100#int((1 /np.degrees(res.value))) #153
    half = n // 2
    sub_hitmap_real = hit_map_real[cx - half : cx - half + n,cy - half : cy - half + n]         

    
    pk3d = threedim_power_spectrum_for_comoving_cubes(  np.zeros((len(ibin), hit_map_real.shape[0], hit_map_real.shape[1])), 
                                                      Delta_Dc_bin, len_pix_bin, len_pix_bin, **kwargs)
        
    pk3d_sub = threedim_power_spectrum_for_comoving_cubes( np.zeros((len(ibin), n, n)),
                                                       Delta_Dc_bin, len_pix_bin, len_pix_bin, **kwargs)
    
    pk3d_angspec = threedim_power_spectrum_for_angular_cubes( np.zeros((len(ibin), hit_map_real.shape[0], hit_map_real.shape[1])), res.value,nus, dnus,nu0,**kwargs) 

    for number in range(nrealizations):

        noise_cube_hitmap = noise_map(hit_map_real, NEI, dt, ratio_survey_time_over_hitmap_time=ratio_survey_time_over_hitmap_time)
        sub_noise_cube = noise_map(sub_hitmap_real, NEI, dt, ratio_survey_time_over_hitmap_time=ratio_survey_time_over_hitmap_time)
        axspec.plot(nus, np.nanstd(noise_cube_hitmap, axis=(1,2)),'*:k', alpha=0.1 )
        axspec.plot(nus, np.nanstd(sub_noise_cube, axis=(1,2)),'*:g', alpha=0.1 )

        #--------------------------------------------------------------------------
        #if you only know about the angular-spectral properties of your cube
        '''
        pk3d_angspec.cube = np.asarray(noise_cube_hitmap)
        _, _, _, _,  pk_outsphere2, _, _, k_out_sphere, _, _ = pk3d_angspec.ap3()
        axpk.loglog(k_out_sphere, pk_outsphere2, '-r', alpha=0.1, drawstyle='steps-mid')
        '''
        #--------------------------------------------------------------------------


        #--------------------------------------------------------------------------
        pk3d.cube = np.asarray(noise_cube_hitmap)
        _, _, _, _,  pk_out_sphere, _, _, k_out_sphere, _, _ = pk3d.p3()
        axpk.loglog(k_out_sphere, pk_out_sphere, '-k',  alpha=0.1, label=f'Total hitmap',drawstyle='steps-mid')
 
        pk3d_sub.cube = np.asarray(sub_noise_cube)
        _, _, _, _,  pk_out_sphere, _, _, k_out_sphere, _, _ = pk3d_sub.p3()
        axpk.loglog(k_out_sphere, pk_out_sphere, '-g', alpha=0.1, label=f'1deg2 hitmap',drawstyle='steps-mid')
        #--------------------------------------------------------------------------
        

        #----------------------------------------------------------------------------------------
        #Ideal case
        pk = P_n_per_bin.value*np.ones(len(k))
        cube, _ = gaussian_random_field_3d(k, pk, n, n, 1, len_pix_bin, len_pix_bin, Delta_Dc_bin )
        pk3d_sub.cube = np.asarray(cube)
        _, _, _, _,  pk_out_sphere, _, _, k_out_sphere, _, _ = pk3d_sub.p3()
        axspec.plot(nus, np.std(cube, axis=(1,2)), 'd:b', alpha=0.1)
        axpk.loglog(k_out_sphere, pk_out_sphere, '-b',alpha=0.1, drawstyle='steps-mid')
        #----------------------------------------------------------------------------------------

        if(False):

            fig, axs = plt.subplots(3,2, figsize=(8,12), dpi=150)               
            zscale = ZScaleInterval()
            vmin, vmax = zscale.get_limits(np.asarray(noise_cube_hitmap)/1e6)
            im = axs[0,0].imshow(noise_cube_hitmap[0,:,:]/1e6, origin='lower', cmap="RdBu_r", vmin=vmin, vmax=vmax)
            cbar = fig.colorbar(im, ax=axs[0,0], orientation='vertical',)
            cbar.set_label('noise [MJy/sr]') 
            im = axs[0,1].imshow(hit_map_real*dt*ratio_survey_time_over_hitmap_time/60, origin='lower', cmap="RdBu_r", vmin=0,vmax=3*60)
            cbar = fig.colorbar(im, ax=axs[0,1], orientation='vertical',)
            cbar.set_label('$\\rm t_{pix}$ [minutes]') 
            p4, p50, p96 = np.nanpercentile(hit_map_real*dt*ratio_survey_time_over_hitmap_time/60, [4,50,96])
            
            im = axs[1,0].imshow(sub_noise_cube[0,:,:]/1e6, origin='lower', cmap="RdBu_r", vmin=vmin, vmax=vmax)
            cbar = fig.colorbar(im, ax=axs[1,0], orientation='vertical',)
            cbar.set_label('noise [MJy/sr]') 
            im = axs[1,1].imshow(sub_hitmap_real*dt*ratio_survey_time_over_hitmap_time/60, origin='lower', cmap="RdBu_r", vmin=0,vmax=3*60)
            cbar = fig.colorbar(im, ax=axs[1,1], orientation='vertical',)
            cbar.set_label('$\\rm t_{pix}$ [minutes]') 
            p4, p50, p96 = np.nanpercentile(sub_hitmap_real*dt*ratio_survey_time_over_hitmap_time/60, [4,50,96])

            im = axs[2,0].imshow(np.asarray(cube)[0,:,:]/1e6, origin='lower', cmap="RdBu_r", vmin=vmin, vmax=vmax)
            cbar = fig.colorbar(im, ax=axs[2,0], orientation='vertical',)
            cbar.set_label('Ideal noise (Fourier) [MJy/sr]')  # Adjust the label if needed
            
patchs = []
patch = mlines.Line2D([], [], ls='solid', color='k', label='hitmap-based, 200h, full map'); patchs.append(patch)
patch = mlines.Line2D([], [], ls='solid', color='g', label='hitmap-based, 200h, 100x100 map'); patchs.append(patch)
patch = mlines.Line2D([], [], ls='solid', color='b', label='Ideal map, 200h'); patchs.append(patch)
axspec.legend(handles =patchs)
Fig.tight_layout()
plt.show()