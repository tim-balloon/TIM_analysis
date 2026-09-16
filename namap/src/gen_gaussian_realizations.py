import numpy as np
from scipy import interpolate
import scipy.constants as cst
from astropy.cosmology import Planck18 as cosmo
import pandas as pd
from astropy.stats import gaussian_fwhm_to_sigma
from specutils import Spectrum1D
from specutils.manipulation import FluxConservingResampler
import astropy.units as u

def load_TIM_noise_model(freq_output = None, telescope_diameter=2, path='./'):

    noise_model_HF = pd.read_csv(path+'TIM_SW_loading.tsv', sep='\t')
    noise_model_LF = pd.read_csv(path+'TIM_LW_loading.tsv', sep='\t')
    lambda_HF = noise_model_HF["# Wavelength[um]"]*1e3 #nm
    nu_HF = cst.c/(lambda_HF*1e-9)/1e9 #GHz
    nHF = noise_model_HF["NEI[Jy/sr s^1/2]"]
    lambda_LF = noise_model_LF["# Wavelength[um]"]*1e3 #nm
    nu_LF = cst.c/(lambda_LF*1e-9)/1e9 #GHz
    nLF = noise_model_LF["NEI[Jy/sr s^1/2]"]
    freqs = np.concatenate((nu_LF[::-1], nu_HF[::-1]))
    noise = (np.concatenate((nLF[::-1], nHF[::-1]))*u.Jy/u.sr)
    #------------------------------------------------------------
    fwhm =  1.22  * cst.c / (freqs * 1e9 * telescope_diameter) * u.rad
    Omega_beam  = (2*np.pi*(fwhm * gaussian_fwhm_to_sigma)**2).to(u.sr)

    if(freq_output is None): return freqs, noise
    else:
        fwhm_output = 1.22  * cst.c / (freq_output * 1e9 * telescope_diameter) * u.rad
        Omega_beam_output  = (2*np.pi*(fwhm_output * gaussian_fwhm_to_sigma)**2).to(u.sr)
        NEFD = noise * Omega_beam
        spec = Spectrum1D(spectral_axis=freqs*u.GHz, flux=NEFD)
        fluxc_resample = FluxConservingResampler()
        sed_in_tim = fluxc_resample(spec, freq_output*u.GHz) 
        model = sed_in_tim.flux/Omega_beam_output

        if(False):

            lambdaw = np.concatenate((lambda_HF, lambda_LF)) /1e3 #um
            noisew = (np.concatenate((nHF, nLF))*u.Jy/u.sr)
            fwhmw = 1.22 * lambdaw * 1e-6 / telescope_diameter * u.rad
            Omega_beamw = (2*np.pi*(fwhmw* gaussian_fwhm_to_sigma)**2).to(u.sr)
            NEFDw = noisew * Omega_beamw

            fig, (ax, ax2) = plt.subplots(2,figsize=(3,3), dpi=200, sharex=True )  
            ax.loglog(lambdaw, noisew,  c='k', label = '$\\rm \\delta \\nu = 2GHz$' )
            ax2.loglog(lambdaw, NEFDw,  c='k' )
            ax2.set_xlabel("Wavelength [$\\rm \\mu$m]")
            ax.set_ylabel("NEI$\\rm _{\\nu}$ $\\rm [Jy/sr.s^{1/2}]$")
            ax2.set_ylabel("NEFD$\\rm _{\\nu}$ [$\\rm Jy.s^{1/2}$]")

            ax.set_xscale("linear")
            ax.set_xlim(240,420)
            def w_to_f(x): return cst.c/(x*1e-6)/1e9
            secax = ax.secondary_xaxis("top", functions=(w_to_f,w_to_f))
            secax.set_xlabel('Frequency [GHz]')

            def jy_to_njy(x): return x/1e6
            secax = ax.secondary_yaxis("right", functions=(jy_to_njy,jy_to_njy))
            secax.set_ylabel("NEI$\\rm _{\\nu}$ $\\rm [MJy/sr.s^{1/2}]$")
            fig.subplots_adjust(wspace=0, hspace=0)

            ax2.plot( cst.c / (freq_output*1e9) / 1e-6, model*Omega_beam_output, ls = '--', c='orange', )
            ax.plot( cst.c / (freq_output*1e9) / 1e-6,  model, ls = '--', c='orange', label = 'output')
            ax.legend()

            plt.show()

        return freq_output, model
    
def gaussian_random_field_3d(k, pk, nx, ny, nz, dx, dy, dz, pk_map = None, k_cutoff= None,force = True):

    """
    Generate a 3D Gaussian random field from a 3D power spectrum.

    Parameters
    ----------
    k : array
        Wavenumbers [Mpc^-1]
    pk : array
        Power spectrum P(k) [Jy^2 / Mpc^3]

    nx, ny, nz : int
        Cube size

    dx, dy, dz : float
        Pixel size [Mpc]

    Returns
    -------
    cube : ndarray
        Gaussian random field [Jy]
    pk_map : ndarray
        3D power spectrum sampled on FFT grid
    """

    # --- white noise cube ---
    noise = np.random.normal(size=(nz, ny, nx))
    noise_fft = np.fft.fftn(noise)

    #Interpolate input power spectrum
    if(pk_map is None):

        # --- Fourier frequencies ---
        kx = np.fft.fftfreq(nx, dx) * 2*np.pi
        ky = np.fft.fftfreq(ny, dy) * 2*np.pi

        if(nz>1):
                kz = np.fft.fftfreq(nz, dz) * 2*np.pi
                kz, ky, kx = np.meshgrid(kz, ky, kx, indexing='ij')
                k_map = np.sqrt(kx**2 + ky**2 + kz**2)
        else: 
                ky, kx = np.meshgrid(ky, kx, indexing='ij')
                k_map = np.sqrt(kx**2 + ky**2)

        if(k_cutoff is not None): kmax = np.minimum(k_cutoff, k.max())
        else: kmax = np.minimum(k.max(), k_map.max()) 
        pk_map = np.zeros(k_map.shape) 
        w = np.where((k_map>k.min()) & (k_map<=kmax))
        if(not w[0].any()): print("wrong k range")
        else:
            f = interpolate.interp1d( k, pk,  kind='linear')
            pk_map[w] = f(k_map[w])
            w1 = np.where( pk_map <= 0)
            if(w1[0].shape[0] != 0 and force): pk_map[w1] = 0
            
        f = interpolate.interp1d(k, pk, bounds_error=False, fill_value=0)
        pk_map = f(k_map)

    # --- voxel volume ---
    Vvox = dx * dy * dz

    # --- apply power spectrum ---
    field_fft = noise_fft * np.sqrt(pk_map / Vvox)
    #print(f'in gen gaussian realizations, Vvox = {Vvox:.2e} Mpc3')

    # --- inverse transform ---
    cube = np.real(np.fft.ifftn(field_fft))

    return cube, pk_map

def noise_map(hitmap, nei, dt, ratio_survey_time_over_hitmap_time = 1):

    t_pix = hitmap * dt * ratio_survey_time_over_hitmap_time
    ny, nx = hitmap.shape
    n_chan = len(nei)
    # Initialize cube
    noise_cube = np.zeros((n_chan, ny, nx, ))
    for i, I in enumerate(nei):
        sigma = np.zeros_like(hitmap, dtype=float)
        sigma[hitmap > 0] = I / np.sqrt(t_pix[hitmap > 0])
        sigma[hitmap == 0] = np.nan  # optional: mark empty pixels
        noise = np.random.normal(loc=0.0, scale=sigma)
        if(False):
            fig, axs = plt.subplots(1,2, figsize=(8,4), dpi=150)
            zscale = ZScaleInterval()
            vmin, vmax = zscale.get_limits(noise)
            im = axs[0].imshow(noise, origin='lower', cmap="RdBu_r", vmin=vmin, vmax=vmax)
            cbar = fig.colorbar(im, ax=axs[0], orientation='vertical',)
            cbar.set_label('noise [MJy/sr]')  # Adjust the label if needed
            im = axs[1].imshow(hitmap, origin='lower')
            cbar = fig.colorbar(im, ax=axs[1], orientation='vertical',)
            cbar.set_label('hit counts')  # Adjust the label if needed
            fig.tight_layout()
        noise_cube[i,:, :] = noise
    return noise_cube