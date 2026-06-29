import numpy as np
import scipy.constants as cst
from astropy import units as u
from IPython import embed
from scipy.optimize import curve_fit
from scipy import interpolate
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

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
