import numpy as np
import scipy.constants as cst
from astropy import units as u
from IPython import embed
from scipy.optimize import curve_fit
from scipy import interpolate
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

class angular_power_spectrum:
    """
    Class to measure an angular power spectrum out of a 2D map

    Parameters
    ----------

    Returns
    -------
    """

    def __init__(self, map, res, delta_k_over_k=0, map2=None):

        """
        Create an instance of the class.

        Parameters
        ----------

        map : 2D numpy array
            The map to analysed, in Jy/sr
        res : float
            pixel size of the map in rad
        delta_k_over_k : float
            relative bin width (0 = linear bins)
        map2: None or 2D array
            if a another map of the same shape as the 1st map is provided, compute their cross-correlation.

        Returns
        -------
        """
        self.map = map
        self.map2 = map2
        self.res = res
        self.delta_k_over_k = delta_k_over_k
        self.ny, self.nx = map.shape

    # ------------------------------------------------------------
    # Correct spatial frequency map 
    # ------------------------------------------------------------
    def give_map_spatial_freq(self):
        """
        Return the 2D Discrete Fourier Transform sample frequencies.

        Parameters
        ----------

        Returns
        -------
        k_map: 2D array
            the Fourier Transform frequency map
        """

        ny, nx = self.ny, self.nx
        res = self.res

        # FFT frequencies in cycles per radian
        ky = np.fft.fftfreq(ny, d=res)
        kx = np.fft.fftfreq(nx, d=res)

        KX, KY = np.meshgrid(kx, ky)

        k_map = np.sqrt(KX**2 + KY**2)
        return k_map

    
    # ------------------------------------------------------------
    # Make k bins
    # ------------------------------------------------------------
    def make_bintab(self, kmin, kmax, dk_min):
        """
        Logarithmic or linear bins.
        if delta_k_over_k is 0, the returned bins are linearly spaced. Else they are log spaced. 

        Parameters
        ----------

        Returns
        -------
        bintab: array
            the wavenumber k bins
        
        """
        dkk = self.delta_k_over_k

        if dkk == 0:
            # linear bins
            bintab = np.arange(kmin, kmax + dk_min, dk_min)
        else:
            k = kmin
            bintab = [kmin]
            while k < kmax:
                dk = max(k * dkk, dk_min)
                dk = min(dk, kmax - k)
                k += dk
                bintab.append(k)

        return np.array(bintab)

    # ------------------------------------------------------------
    # Compute the k-binning and maps
    # ------------------------------------------------------------
    def set_k_infos(self):
        """
        Compute all wavenumber related quantities based on the resolution and shape of the map to be analysed.
        The function computes the wavenumber k bins, the center of the bins, the Nyquist wavenumber, and the 2D k map. 

        Parameters
        ----------

        Returns
        -------        
        """

        ny, nx = self.ny, self.nx
        res = self.res

        k_map = self.give_map_spatial_freq()

        kmin = 1.0 / (min(ny, nx) * res)
        kmax = np.max(k_map)

        dk_min = kmin  # natural Fourier bin width

        k_bin_tab = self.make_bintab(kmin, kmax, dk_min)

        # Bin centers
        k_out = 0.5 * (k_bin_tab[1:] + k_bin_tab[:-1])

        self.k_map = k_map
        self.k_bin_tab = k_bin_tab
        self.k_out = k_out

    # ------------------------------------------------------------
    # Main P(k) estimator
    # ------------------------------------------------------------
    def p2(self):
        """
        Estimates the angular power spectrum in Jy**2/sr of an 2D angular map in Jy/sr

        Parameters
        ----------

        Returns
        -------
        pk: array
            the Fourier amplitudes in Jy**2/sr
        k_out: array
            the center of the wavenumber k bins in rad-1        
        """

        ny, nx = self.ny, self.nx
        res = self.res

        self.set_k_infos()

        k_map = self.k_map

        # FFTs
        ft = np.fft.fft2(self.map)

        if self.map2 is None:
            ft2 = ft
        else:
            ft2 = np.fft.fft2(self.map2)

        norm = (res**2) / (nx * ny)
        p2map = (ft * np.conj(ft2)).real * norm

        # Compute radial average
        hist_w, _ = np.histogram(k_map, bins=self.k_bin_tab, weights=p2map)
        hist_n, _ = np.histogram(k_map, bins=self.k_bin_tab)

        pk = np.zeros_like(hist_w)
        mask = hist_n > 0
        pk[mask] = hist_w[mask] / hist_n[mask]
        pk[~mask] = np.nan

        return pk, self.k_out

if __name__ == '__main__':

    import numpy as np
    from astropy.io import fits
    import matplotlib.pyplot as plt
    from astropy.visualization import ZScaleInterval
    from astropy.wcs import WCS

    fig, axs = plt.subplots(1,3,figsize=(8, 6) )#, subplot_kw={'projection': wcs})

    for extension,c in zip((0,1),('r', 'b')):

        map_value = fits.getdata('../fits_and_hdf5/scanned_map_TOD_on_2_sources_separated_by_150.8_with_1xbigger_sigma_PSF_LW.fits', ext=0)[0] 
        hdr = fits.getheader('../fits_and_hdf5/scanned_map_TOD_on_2_sources_separated_by_150.8_with_1xbigger_sigma_PSF_LW.fits', ext=0)
        wcs3d = WCS(hdr) 
        res = (hdr['CDELT1'] * u.Unit(hdr['CUNIT1'])).to(u.rad).value
        wcs = wcs3d.slice((extension, slice(None), slice(None)))

        if(extension==1):
            map_value = map_value[50:150, 50:150]
            map_value[np.isnan(map_value)] = 0

        zscale = ZScaleInterval()
        if(extension==0): vmin, vmax = zscale.get_limits(map_value)

        im = axs[extension].imshow(map_value, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        divider = make_axes_locatable(axs[extension])
        cax = divider.append_axes("right", size="5%", pad=0.05)  # size can be a percentage or absolute
        fig.colorbar(im, cax=cax, label='Amplitude')
        axs[extension].set_xlabel('X pixel')
        axs[extension].set_ylabel('Y pixel')
        axs[extension].set_aspect('equal')

        apk = angular_power_spectrum(map_value, res, delta_k_over_k=0.3 )
        pk, k = apk.p2()
        axs[-1].loglog(k,pk,f'-o{c}', markersize=1)
        axs[-1].set_aspect('equal')
        plt.tight_layout()

        import powspec 
        pk_powspec, k_edges = powspec.power_spectral_density(map_value, res=res)
        axs[-1].loglog(k_edges[1:],pk_powspec,'-ok', markersize=1)

    plt.show()
