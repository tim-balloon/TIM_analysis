import numpy as np
import scipy.constants as cst
from astropy import units as u
from IPython import embed
from scipy.optimize import curve_fit
from scipy import interpolate
from mpl_toolkits.axes_grid1 import make_axes_locatable

class tod_psd:
    """
    Class to measure the power spectral density (PSD) of time-ordered data (TODs).

    Parameters
    ----------

    Returns
    -------
    """

    def __init__(self, det_data, freq_res, delta_f_over_f=0.5):

        """
        Create an instance of the class.

        Parameters
        ----------
        det_data : list
            List of time-ordered data arrays, with one array per detector.
        freq_res : float
            Sampling frequency of the time-ordered data.
        delta_f_over_f : float, optional
            Relative frequency-bin width. A value of 0 gives linear bins,
            while a non-zero value gives logarithmically spaced bins.

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
        Return the discrete Fourier-transform sample frequencies.

        Parameters
        ----------

        Returns
        -------
        f : numpy.ndarray
            One-dimensional array containing the Fourier frequencies
            associated with the TOD sampling.
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
        Construct linear or logarithmic frequency bins.

        If delta_f_over_f is 0, the bins are linearly spaced with a
        minimum width of df_min. Otherwise, the bin width scales with
        frequency while remaining larger than df_min.

        Parameters
        ----------
        fmin : float
            Minimum Fourier frequency of the binning range.
        fmax : float
            Maximum Fourier frequency of the binning range.
        df_min : float
            Minimum allowed frequency-bin width.

        Returns
        -------
        bintab : numpy.ndarray
            Array containing the frequency-bin edges.
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
        Compute the Fourier frequencies and frequency-bin information.

        The natural Fourier frequency resolution is determined from the
        TOD length and sampling frequency. The resulting frequency array,
        bin edges, and bin centers are stored as class attributes.

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
        Estimate the power spectral density of the TODs.

        The PSD is computed independently for each detector by taking
        the squared modulus of the Fourier transform and averaging the
        resulting power within the defined frequency bins.

        Parameters
        ----------
        mask_correction : bool, optional
            Whether to apply a correction for masked or invalid samples.
            Currently not used in the calculation.

        Returns
        -------
        psd_list : numpy.ndarray
            Power spectral density values for each detector, averaged
            within the frequency bins.
        f_out : numpy.ndarray
            Centers of the Fourier-frequency bins.      
        """

        n  = self.n
        freq_res = self.freq_res
        
        norm = (1/freq_res)**2 / n # is the square really there ?

        self.set_f_infos()

        f = self.f

        # FFTs
        psd_list = []
        
        for i, tod in enumerate(self.det_data):

            # Create a mask (1 where valid, 0 where NaN)
            mask = np.isfinite(tod).astype(float)

            # Fill NaNs with 0 (or the mean, depending on your normalization)
            tod_filled = np.nan_to_num(tod, nan=0.0)

            ft = np.fft.fft(tod)
            p2 = (ft * np.conj(ft)).real * norm

            # Compute radial average
            hist_w, _ = np.histogram(f, bins=self.f_bin_tab, weights=p2)
            hist_n, _ = np.histogram(f, bins=self.f_bin_tab)

            psd_list.append(hist_w / hist_n )

        return psd_list, self.f_out
