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
    Class to measure angular power spectra out of a list of 2D maps

    Parameters
    ----------

    Returns
    -------
    """

    def __init__(self, maps, res, delta_k_over_k=0, map2=None):

        """
        Create an instance of the class.

        Parameters
        ----------

        maps : list
            The list of 2d maps to analysed, in Jy/sr
        res : float
            pixel size of the map in rad
        delta_k_over_k : float
            relative bin width (0 = linear bins)
        map2: list
            if a another list of maps of the same shape as the 1st list is provided, compute their cross-correlation.

        Returns
        -------
        """
        self.maps = maps                      #List of 2d maps in units Jy/sr 
        self.map2 = map2                      #if provided, list of 2d maps to cross-correlate with. 
        self.res = res                        #pixel size of maps in sr. 
        self.delta_k_over_k = delta_k_over_k  #relative k bin width
        self.ny, self.nx = maps[0].shape      #Number of pixels in the y and x directions

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
    def p2(self,mask_correction=False):
            
        """
        Estimates the angular power spectrum in Jy**2/sr of an 2D angular map in Jy/sr

        Parameters
        ----------

        Returns
        -------
        pk: array
            the Fourier amplitudes in Jy**2/sr
        k_bin_tab: array
            the k bins in rad-1        
        """

        ny, nx = self.ny, self.nx
        res = self.res

        norm = (res**2) / (nx * ny)

        self.set_k_infos()

        k_map = self.k_map

        # FFTs
        pk_list = []

        for i, map in enumerate(self.maps):

            # Create a mask (1 where valid, 0 where NaN)
            mask = np.isfinite(map).astype(float)

            # Fill NaNs with 0 (or the mean, depending on your normalization)
            map_filled = np.nan_to_num(map, nan=0.0)

            ft = np.fft.fft2(map_filled)

            if self.map2 is None:
                ft2 = ft
            else:
                map_filled_2 = np.nan_to_num(self.map2[i], nan=0.0)
                ft2 = np.fft.fft2(map_filled_2)
            
            p2map = (ft * np.conj(ft2)).real * norm

            if(mask_correction): 
            # Corrected power spectrum
                ft_mask = np.fft.fft2(mask)
                pmask = np.abs(ft_mask)**2 
                w = np.where(np.abs(pmask)>0)
                p2map[w] /=  np.abs(pmask)[w]

            # Compute radial average
            hist_w, _ = np.histogram(k_map, bins=self.k_bin_tab, weights=p2map)
            hist_n, _ = np.histogram(k_map, bins=self.k_bin_tab)

            pk = np.zeros_like(hist_w)
            mask = hist_n > 0
            pk[mask] = hist_w[mask] / hist_n[mask]
            pk[~mask] = np.nan
            pk_list.append(pk)

        return pk_list, self.k_out


class tod_psd:
    """
    Class to measure a power spectrum density of time-ordered data (TODs). 

    Parameters
    ----------

    Returns
    -------
    """

    def __init__(self, det_data, freq_res, delta_f_over_f=0):

        """
        Create an instance of the class.

        Parameters
        ----------

        det_data : list
            list of TODs
        freq_res : float
            acquisition frequency of the TODs
        delta_k_over_k : float
            relative bin width (0 = linear bins)
        Returns
        -------
        """
        self.det_data = det_data              #List of ime-ordered data
        self.freq_res = freq_res              #frequency of the ime-ordered data
        self.delta_f_over_f = delta_f_over_f  #relative bin width
        self.n = len(det_data[0])             # lenght of the time ordered data

    # ------------------------------------------------------------
    # Correct spatial frequency map 
    # ------------------------------------------------------------
    def give_fourier_freq(self):
        """
        Return the 2D Discrete Fourier Transform sample frequencies.

        Parameters
        ----------

        Returns
        -------
        f: 1D array
            the Fourier Transform frequencies
        """

        n = self.n
        freq_res = self.freq_res
        # FFT frequencies in cycles per radian
        f = np.fft.fftfreq(n, d=1/freq_res)

        return f

    # ------------------------------------------------------------
    # Make k bins
    # ------------------------------------------------------------
    def make_bintab(self, fmin, fmax, df_min):
        """
        Logarithmic or linear bins.
        if delta_k_over_k is 0, the returned bins are linearly spaced. Else they are log spaced. 

        Parameters
        ----------

        Returns
        -------
        bintab: array
            the Fourier frequency bins
        
        """
        dff = self.delta_f_over_f

        if dff == 0:
            # linear bins
            bintab = np.arange(fmin, fmax + df_min, df_min)
        else:
            f = fmin
            bintab = [fmin]
            while f < fmax:
                df = max(f * dff, df_min)
                df = min(df, fmax - f)
                f += df
                bintab.append(f)

        return np.array(bintab)

    # ------------------------------------------------------------
    # Compute the k-binning and maps
    # ------------------------------------------------------------
    def set_f_infos(self):
        """
        Compute all wavenumber related quantities based on the frequency and lenght of the TODs to be analysed.

        Parameters
        ----------

        Returns
        -------        
        """

        n = self.n
        freq_res = self.freq_res

        f = self.give_fourier_freq()

        fmin = 1.0 / (n * (1/freq_res))
        fmax = np.max(f)

        df_min = fmin  # natural Fourier bin width

        f_bin_tab = self.make_bintab(fmin, fmax, df_min)

        # Bin centers
        f_out = 0.5 * (f_bin_tab[1:] + f_bin_tab[:-1])

        self.f = f
        self.f_bin_tab = f_bin_tab
        self.f_out = f_out

    # ------------------------------------------------------------
    # Main P(k) estimator
    # ------------------------------------------------------------
    def p2(self,mask_correction=False):
            
        """
        Estimates the power spectral densities

        Parameters
        ----------

        Returns
        -------
        psd_list: array
            the Fourier amplitudes
        k_bin_tab: array
            the k bins in rad-1        
        """

        n  = self.n
        freq_res = self.freq_res
        
        norm = (1/freq_res)**2 / n # is the square really there ?

        self.set_k_infos()

        f = self.f

        # FFTs
        psd_list = []
        
        for i, tod in enumerate(self.det_data):

            # Create a mask (1 where valid, 0 where NaN)
            mask = np.isfinite(tod).astype(float)

            # Fill NaNs with 0 (or the mean, depending on your normalization)
            tod_filled = np.nan_to_num(tod, nan=0.0)

            ft = np.fft.fft2(tod)
            p2 = (ft * np.conj(ft)).real * norm

            # Compute radial average
            hist_w, _ = np.histogram(f, bins=self.f_bin_tab, weights=p2)
            hist_n, _ = np.histogram(f, bins=self.f_bin_tab)

            psd_list.append(hist_w / hist_n )

        return psd_list, self.f_out
