import numpy as np
import scipy.signal as sgn
from scipy.ndimage import uniform_filter1d
#import pygetdata as gd
import src.loaddata as ld
import h5py
from IPython import embed 

class data_cleaned():

    '''
    Class to clean the detector TOD using the functions in 
    the next classes. Check them for more explanations

    Parameters
    ----------

    Returns
    -------
    '''

    def __init__(self, data, detlist, fs, 
                 cutoff, 
                 polynomialorder, 
                 despike, sigma, prominence, 
                 sigma_clipping, low_thresh, high_thresh,
                 DT):
        """
        creates an instance of the class to clean the detector TODs.

        Parameters
        ----------
        data : list
            detector TODs
        fs : float
            frequency sampling of the detectors
        cutoff : float
            cutoff frequency of the highpass filter     
        polynomialorder : int
            polynomial order for fitting
        despike : bool
            if True despikes the data using scipy.signal
        sigma : float
            height in std value of peaks to remove 
        prominence : float
            prominence of peaks w.r.t neighboring peaks in std value
        DT : type
            Float precision required
        sigma_clipping : bool
            if True, remove TODs whose variance is below low_thresh or above high_thresh
        low_thresh : float
            lower variance treshold in std units below which remove a TOd from further analysis
        high_thresh : float
            higher variance treshold in std units above which remove a TOd from further analysis
        Returns
        -------
        """

        self.data = data                       #detector TODs
        self.detlist = detlist                 #detector name list
        self.fs = float(fs)                    #frequency sampling of the detector
        self.cutoff = float(cutoff)            #cutoff frequency of the highpass filter     
        self.polynomialorder = polynomialorder #polynomial order for fitting
        self.sigma = sigma                     #height in std value to look for spikes
        self.prominence = prominence           #prominence in std value to look for spikes
        self.despike = despike                 #if True despikes the data 
        self.sigma_clipping = sigma_clipping   #if True disgard TODs based on their variance.
        self.low_thresh = low_thresh           #Lower variance threshold in sigma on which to disgard a TOD from further analysis.
        self.high_thresh  = high_thresh        #Higher variance threshold in sigma on which to disgard a TOD from further analysis.
        self.DT = DT                           #Float precision

    def data_clean(self):

        '''
        Function to clean the TODs
        
        Parameters
        ----------

        Returns
        -------
        cleaned_data : list
            list of cleaned data timestreams
        accepted_detectors_list : list 
            List of detectors name who passed the sigma clipping
        rejected_detetectors_list : list
            List of detectors name who did not passed the sigma clipping and been removed from further analysis
        '''
        
        
        cleaned_data = [] #[np.zeros_like(slice) for slice in self.data]

        rejected_detetectors_list = []
        if(not self.sigma_clipping): accepted_detectors_list = self.detlist
        else:                        accepted_detectors_list = []

        for i, (data, det_name) in enumerate(zip(self.data, self.detlist)):

            if self.sigma_clipping:
                clip = sigma_clipping(data)
                reject = clip.clipping(low_thresh = self.low_thresh, high_thresh = self.high_thresh) 
                if(reject):
                    rejected_detetectors_list.append(det_name)
                    continue
                else: 
                    accepted_detectors_list.append(det_name)
                    data_clipped = data
            else: data_clipped = data.copy()

            det_data = detector_trend(data, self.DT)

            if self.polynomialorder != 0: 
                residual_data = det_data.fit_residual(order=self.polynomialorder)
            else: residual_data = data.copy()

            if self.despike:
                desp = despike(residual_data)
                data_despiked = desp.replace_peak(hthres=self.sigma, pthres=self.prominence)
            else: data_despiked = residual_data.copy()
            
            if self.cutoff != 0:
                filterdat = filterdata(data_despiked, self.cutoff, self.fs, self.DT)
                cleaned_data.append( filterdat.ifft_filter(window=True) )
            else: cleaned_data.append( data_despiked )
        
        return cleaned_data, accepted_detectors_list, rejected_detetectors_list
    
class despike():

    '''
    Class for detecting and replacing spikes in a time-ordered data (TOD) signal.

    The class identifies peaks in the TOD based on their height and/or
    prominence relative to the standard deviation of the signal. Detected
    spikes can then be characterized by their width and replaced with a
    noise realization.

    Parameters
    ----------

    Returns
    -------
    '''
    
    def __init__(self, data):

        '''
        Create an instance of the despike class.

        Parameters
        ----------
        data : numpy.ndarray
            One-dimensional time-ordered data (TOD) to be despiked.

        Returns
        -------
        '''

        self.data = data

    def findpeak(self, hthres=5, pthres=0):
        '''
        Find peaks in the TOD that are likely to be spikes.

        Peaks are identified using the absolute value of the signal
        after subtracting its mean when the signal is strictly positive.
        The peak height and prominence thresholds are expressed in units
        of the standard deviation of the signal.

        Parameters
        ----------
        hthres : float, optional
            Threshold on the peak height, expressed in units of the
            standard deviation of the signal. A value of 0 disables
            the height criterion. Default is 5.
        pthres : float, optional
            Threshold on the peak prominence, expressed in units of the
            standard deviation of the signal. A value of 0 disables
            the prominence criterion. Default is 0.

        Returns
        -------
        index : numpy.ndarray
            Indices of the detected peaks in the TOD.
        '''

        index = np.ones(1)
        # ledge = np.array([], dtype = 'int')
        # redge = np.array([], dtype = 'int')

        y_std = np.std(self.data)
        y_mean = np.mean(self.data)

        if np.amin(self.data) > 0:
            data_to_despike = self.data-y_mean
        else:
            data_to_despike = self.data.copy()

        # plt.plot(np.abs(data_to_despike))
        # plt.show()

        if hthres != 0 and pthres == 0:
            index, param = sgn.find_peaks(np.abs(data_to_despike), height = hthres*y_std, distance=100)
        elif pthres != 0 and hthres == 0:
            index, param = sgn.find_peaks(np.abs(data_to_despike), prominence = pthres*y_std)
        elif hthres != 0 and pthres != 0:
            index, param = sgn.find_peaks(np.abs(data_to_despike), height = hthres*y_std, \
                                          prominence = pthres*y_std)

        # ledget = sgn.peak_widths(np.abs(data_to_despike),index)[2]
        # redget = sgn.peak_widths(np.abs(data_to_despike),index)[3]

        # ledge = np.append(ledge, np.floor(ledget).astype(int))
        # redge = np.append(redge, np.ceil(redget).astype(int))

        #print('INDEX', index)

        return index

    def peak_width(self, peaks, hthres=5, pthres=0, window = 100):
        '''
        Estimate the left and right edges of detected peaks.

        For each detected peak, the method searches for the minimum
        absolute signal value within a specified window on either side
        of the peak. These minima are used as the left and right edges
        of the peak.

        Parameters
        ----------
        peaks : numpy.ndarray
            Indices of the peaks for which the widths should be estimated.
        hthres : float, optional
            Peak-height threshold used when identifying peaks. This
            parameter is only relevant if peaks are determined within
            this method. Default is 5.
        pthres : float, optional
            Peak-prominence threshold used when identifying peaks. This
            parameter is only relevant if peaks are determined within
            this method. Default is 0.
        window : int, optional
            Number of samples on each side of a peak over which to search
            for the left and right edges. Default is 100.

        Returns
        -------
        param: numpy.ndarray
            Peak widths estimated using ``scipy.signal.peak_widths``.
        ledge : numpy.ndarray
            Indices of the left edges of the peaks.
        redge : numpy.ndarray
            Indices of the right edges of the peaks.
        '''
        
        #peaks = self.findpeak(hthres=hthres, pthres=pthres)
        y_mean = np.mean(self.data)
        if np.amin(self.data) > 0:
            data_to_despike = self.data-y_mean
        else:
            data_to_despike = self.data.copy()
        param = sgn.peak_widths(np.abs(data_to_despike),peaks, rel_height = 1.0)

        ledge = np.array([], dtype='int')
        redge = np.array([], dtype='int')

        for i in range(len(peaks)):
            left_edge, = np.where(np.abs(data_to_despike[peaks[i]-window:peaks[i]]) == \
                                  np.amin(np.abs(data_to_despike[peaks[i]-window:peaks[i]])))
            right_edge, = np.where(np.abs(data_to_despike[peaks[i]:peaks[i]+window]) == \
                                   np.amin(np.abs(data_to_despike[peaks[i]:peaks[i]+window])))

            left_edge += (peaks[i]-window)
            right_edge += peaks[i]

            ledge = np.append(ledge, left_edge[-1])
            redge = np.append(redge, right_edge[-1])
            #print('INDEX', i, peaks[i], left_edge, right_edge)
            #print('PEAKS', left_edge, right_edge, peaks[i])
        #print(len(peaks), len(ledge), len(redge))
        return param[0].copy(), ledge, redge

    def replace_peak(self, hthres=5, pthres = 5, peaks = np.array([]), widths = np.array([])):

        """
        Replace detected spikes with a noise realization.

        Spikes are identified and their corresponding samples are replaced
        with a noise realization. The noise distribution is selected based
        on the relationship between the mean and variance of the despiked
        signal. A Poisson distribution is used when the mean and variance
        are approximately equal; otherwise, a Gaussian distribution is used.

        Parameters
        ----------
        hthres : float, optional
            Threshold on the peak height, expressed in units of the
            standard deviation of the signal. Default is 5.
        pthres : float, optional
            Threshold on the peak prominence, expressed in units of the
            standard deviation of the signal. Default is 5.
        peaks : numpy.ndarray, optional
            Indices of previously detected peaks. If not provided, the
            peaks are identified using ``findpeak``.
        widths : numpy.ndarray, optional
            Peak edge information previously computed by ``peak_width``.
            If not provided, the peak widths and edges are computed
            automatically.

        Returns
        -------
        replaced : numpy.ndarray
            Copy of the input TOD with the detected spike samples
            replaced by a noise realization.
        """


        x_inter = np.array([], dtype = 'int')

        ledge = np.array([], 'int')
        redge = np.array([], 'int')
        replaced = self.data.copy()

        if np.size(peaks) == 0:
            peaks = self.findpeak(hthres=hthres, pthres=pthres)
        if np.size(widths) == 0:
            widths = self.peak_width(peaks=peaks, hthres=hthres, pthres=pthres)

        for i in range(0, len(peaks)):
            # width = int(np.ceil(widths[0][i]))
            # # if width <= 13:
            # #     interval = 25
            # # elif width > 13 and width < 40:
            # #     interval = width*2
            # # else:
            # #     interval = width*3

            left_edge = int(np.floor(widths[1][i]))
            right_edge = int(np.ceil(widths[2][i]))
            ledge = np.append(ledge, left_edge)
            redge = np.append(redge, right_edge)

            x_inter = np.append(x_inter, np.arange(left_edge, right_edge))
            replaced[left_edge:right_edge] = (replaced[left_edge]+\
                                              replaced[right_edge])/2.

        final_mean = np.mean(replaced)
        final_std = np.std(replaced)
        final_var = np.var(replaced)

        p_stat = np.abs(final_mean/final_var-1.)
        #print('CHAR', p_stat, final_mean, final_std, final_var)
        if p_stat <=1e-2:
            '''
            This means that the variance and the mean are approximately the 
            same, so the distribution is Poissonian.
            '''
            mu = (final_mean+final_var)/2.
            y_sub = np.random.poisson(mu, len(x_inter))
        else:
            y_sub = np.random.normal(final_mean, final_std, len(x_inter))

        if np.size(y_sub) > 0:
            replaced[x_inter] = y_sub
        #print(left_edge, right_edge)
        #print('TEST', x_inter, ledge, redge)

        return replaced

class filterdata():
    '''
    Class for filtering detector time-ordered data (TOD).

    The class provides Butterworth and cosine high-pass filters that can
    be applied either directly in the time domain or in Fourier space.

    Parameters
    ----------

    Returns
    -------
    '''

    def __init__(self, data, cutoff, fs, DT):
        """
        Create an instance of the filterdata class.

        Parameters
        ----------
        data : numpy.ndarray
            Detector time-ordered data (TOD) to be filtered.
        cutoff : float
            High-pass filter cutoff frequency in Hz.
        fs : float
            Sampling frequency of the detector TOD in Hz.
        DT : type
            float precision required
        
        Returns
        -------
        """
        self.data = data
        self.cutoff = cutoff
        self.fs = fs
        self.DT = DT

    def highpass(self, order):
        '''
        Compute the coefficients of a Butterworth high-pass filter.

        Parameters
        ----------
        order : int
            Order of the Butterworth filter.

        Returns
        -------
        b : numpy.ndarray
            Numerator coefficients of the filter.
        a : numpy.ndarray
            Denominator coefficients of the filter.
        '''
        
        nyq = 0.5*self.fs
        normal_cutoff = self.cutoff / nyq
        b, a = sgn.butter(order, normal_cutoff, btype='highpass', analog=False)
        return b, a

    def butter_highpass_filter(self, order=5):
        """
        Apply a Butterworth high-pass filter to the detector TOD.

        Parameters
        ----------
        order : int, optional
            Order of the Butterworth filter. Default is 5.

        Returns
        -------
        filterdata : numpy.ndarray
            Detector TOD after applying the Butterworth high-pass filter.
        """

        b, a = self.highpass(order)
        filterdata = sgn.lfilter(b, a, self.data)

        return filterdata

    def cosine_filter(self, f):
        '''
        Compute the response of a cosine high-pass filter.

        The filter response is zero below half the cutoff frequency,
        smoothly increases from zero to one between half the cutoff
        frequency and the cutoff frequency, and is one above the
        cutoff frequency.

        Parameters
        ----------
        f : float
            Frequency at which to evaluate the filter response, in Hz.

        Returns
        -------
        resp : float
            Filter response at the input frequency. The value ranges
            from 0 to 1.
        '''

        if f < .5*self.cutoff:
            resp = 0
        elif 0.5*self.cutoff <= f  and f <= self.cutoff:
            resp = 0.5-0.5*np.cos(np.pi*(f-0.5*self.cutoff)*(self.cutoff-0.5*self.cutoff)**-1)
        elif f > self.cutoff:
            resp = 1
        return resp
    
    def fft_filter(self, window):
        '''
        Compute the Fourier transform of the filtered detector TOD.

        A cosine high-pass filter is applied to the Fourier transform of
        the input data. Optionally, a Hann window can be applied to the
        data before computing the Fourier transform.

        Parameters
        ----------
        window : bool
            If True, apply a Hann window to the input data before computing
            the Fourier transform. If False, no window is applied.

        Returns
        -------
        filtereddata: numpy.ndarray
            Filtered Fourier transform of the detector TOD.
        '''

        if window is True:
            window_data = np.hanning(len(self.data))

            fft_data = np.fft.rfft(self.data*self.DT(window_data))
        else:
            fft_data = np.fft.rfft(self.data)

        fft_frequency = np.fft.rfftfreq(np.size(self.data), 1/self.fs)

        vect = np.vectorize(self.cosine_filter)

        filtereddata = vect(fft_frequency)*fft_data

        return filtereddata

    def ifft_filter(self, window):
        """
        Transform the filtered Fourier-domain data back to the time domain.

        The Fourier-domain data are obtained using ``fft_filter`` and
        transformed back using an inverse real Fourier transform.

        Parameters
        ----------
        window : bool
            If True, apply a Hann window before computing the Fourier
            transform. If False, no window is applied.

        Returns
        -------
        numpy.ndarray
            Filtered detector TOD in the time domain.
        """
        

        ifft_data = np.fft.irfft(self.fft_filter(window=window), len(self.data))

        return self.DT(ifft_data)
    
class detector_trend():

    '''
    Class for fitting and removing a polynomial trend from detector
    time-ordered data (TOD).

    Parameters
    ----------

    Returns
    -------
    '''

    def __init__(self, data, DT):
        '''

        Create an instance of the detector_trend class.

        Parameters
        ----------
        data : numpy.ndarray
            Detector time-ordered data (TOD) to be detrended.
        DT : type
            float precision required

        Returns
        -------
        '''

        self.data = data
        self.DT = DT

    def polyfit(self, edge = 0, order=6):
        '''
        Fit a polynomial trend to the detector TOD.

        The polynomial is fitted to the TOD as a function of the sample
        index. The fitted polynomial is then evaluated over the full
        length of the TOD.

        Parameters
        ----------
        edge : int, optional
            Number of samples at the edges of the TOD to exclude from
            the fit. Currently not used in the polynomial fitting.
            Default is 0.
        order : int, optional
            Order of the polynomial used to fit the TOD. Default is 6.

        Returns
        -------
        y_fin : numpy.ndarray
            Polynomial fit evaluated over the full TOD.
        index_exclude : numpy.ndarray
            Indices of samples excluded from the fit.
        '''

        x = np.arange(len(self.data))

        index_exclude = np.array([], dtype=int)

        if np.size(edge) == 1:
            if(self.DT == np.float16): p = np.polyfit(x, np.float32(self.data), order)
            else: p = np.polyfit(x, self.data, order)
            poly = np.poly1d(p)
            y_fin = poly(x).astype(self.DT)

        return y_fin, index_exclude.astype(int)
    
    def fit_residual(self, edge = 0, order=6):
        """
        Remove the fitted polynomial trend from the detector TOD.

        The polynomial trend is computed using ``polyfit`` and subtracted
        from the input TOD. Samples identified as excluded indices are
        set to zero before subtracting the fitted trend.

        Parameters
        ----------
        edge : int, optional
            Number of samples at the edges of the TOD to exclude from
            the fit. Default is 0.
        order : int, optional
            Order of the polynomial used to fit the TOD. Default is 6.

        Returns
        -------
        final_tod : numpy.ndarray
            Detrended detector TOD obtained by subtracting the fitted
            polynomial trend from the input data.
        """
        polyres = self.polyfit(edge=edge, order=order)
        fitteddata = polyres[0]
        index = polyres[1]

        zero_data = self.data.copy()
        if(len(index)>0): zero_data[index] = 0.

        final_tod = -fitteddata+zero_data

        return final_tod

class sigma_clipping():

    '''

    Class for evaluating the standard deviation of a time-ordered data
    (TOD) stream and identifying timestreams with unusually low or high
    variance.

    Parameters
    ----------

    Returns
    -------
    '''

    def __init__(self, data):
        '''
        Create an instance of the sigma_clipping class.

        Parameters
        ----------
        data : numpy.ndarray
            Time-ordered data (TOD) to be evaluated.

        Returns
        -------
        '''

        self.data = np.float32(data)
    
    def clipping(self, low_thresh, high_thresh):
        '''
        Determine whether the TOD should be rejected based on its variance.

        The mean and mean square of the TOD are computed using a sliding
        window spanning the full length of the timestream. The standard
        deviation is then computed from these quantities. The timestream
        is flagged for rejection if any value of the standard deviation
        falls below the lower threshold or exceeds the upper threshold.

        Parameters
        ----------
        low_thresh : float
            Lower threshold for the standard deviation of the TOD.
            The timestream is rejected if its standard deviation falls
            below this value.
        high_thresh : float
            Upper threshold for the standard deviation of the TOD.
            The timestream is rejected if its standard deviation exceeds
            this value.

        Returns
        -------
        reject : bool
            ``True`` if the timestream should be rejected because its
            standard deviation is outside the specified thresholds;
            ``False`` otherwise.
        '''        

        # mean in sliding window
        
        mean = uniform_filter1d(self.data, size=len(self.data), mode='nearest')

        # mean of squared signal
        mean_sq = uniform_filter1d(self.data**2, size=len(self.data), mode='nearest')

        # variance = E[x^2] - (E[x])^2
        var = mean_sq - mean**2

        # and standard deviation
        sigma = np.sqrt(var)
        #print(f'sigma mean={sigma.mean():.3f}, min={sigma.min():.3f}, max={sigma.max():.3f},, median={np.median(sigma):.3f}')
        reject = np.any((sigma > high_thresh) | (sigma < low_thresh))

        return reject

class kidsutils():
    '''
    Class containing useful functions for KIDs

    Parameters
    ----------

    Returns
    -------
    '''

    def rotatePhase(self, I, Q):

        '''
        Rotate phase for a KID

        Parameters
        ----------
        I : numpy.ndarray
            I Time-ordered data 
        Q : numpy.ndarray
            Q Time-ordered data 

        Returns
        -------
        I : numpy.ndarray
            Rotated I Time-ordered data 
        Q : numpy.ndarray
            Rotated Q Time-ordered data 
        '''

        X = I+1j*Q
        phi_avg = np.arctan2(np.mean(Q),np.mean(I))
        E = X*np.exp(-1j*phi_avg)
        I = E.real
        Q = E.imag

        return I, Q

    def KIDphase(self, I, Q):

        '''
        Compute the phase of a KID. This is proportional to power, in particular
        Power = Phase/Responsivity

        Parameters
        I: numpy.ndarray
            I phase of the kid 
        Q: numpy.ndarray
            Q phase of the kid
        ----------
        Returns
        -------
        phi: numpy.ndarray
            the phase of the kid
        '''

        phibar = np.arctan2(np.mean(Q),np.mean(I))
        #I_rot, Q_rot = self.rotatePhase(I, Q)
        phi = np.arctan2(Q,I)

        return phi-phibar

    def KIDmag(self, I, Q):

        ''' 
        Compute the magnitude response of a KID

        Parameters
        I: numpy.ndarray
            I phase of the kid 
        Q: numpy.ndarray
            Q phase of the kid
        ----------
        Returns
        -------
        mag: numpy.ndarray
            the magnitude
        '''

        return np.sqrt(I**2+Q**2)
    
class AntiAliasingFilter():
    """

    Anti-aliasing filter for downsampling time-ordered data.

    The filter is designed as a linear-phase low-pass filter with
    configurable cutoff frequency, number of taps, and window function.
    The filtered data can then be downsampled from ``fs_in`` to ``fs_out``.
    
    Parameters
    ----------

    Returns
    -------
    """

    def __init__(self, fs_in, fs_out, DT, fc=None, numtaps=257, window='hann'):
        """
        Create an anti-aliasing filter for downsampling time-ordered data.

        Parameters
        ----------
        fs_in : float
            Input sampling frequency in Hz.
        fs_out : float
            Output sampling frequency in Hz.
        DT : type
            Data type used for the filtered output.
        fc : float, optional
            Low-pass filter cutoff frequency in Hz. If not provided, the
            cutoff frequency is set to ``0.45 * fs_out``.
        numtaps : int, optional
            Number of coefficients in the filter. Defaults to 257.
        window : str, optional
            Window function applied to the filter coefficients. Must be
            'hann' or 'hamming'. Defaults to 'hann'.

        Returns
        -------
        """
        self.fs_in = fs_in
        self.fs_out = fs_out
        self.fc = fc if fc is not None else 0.45 * fs_out
        self.numtaps = numtaps
        self.window = window
        self.DT=DT

        self.h = self._design_filter()

    def _design_filter(self):
        """
        Design linear-phase low-pass filter
        
        Parameters
        ----------

        Returns
        -------
        h : numpy.ndarray
            Normalized filter coefficients with unity DC gain.
        """
        
        n = np.arange(self.numtaps) - (self.numtaps - 1) / 2

        h = 2 * self.fc / self.fs_in * np.sinc(2 * self.fc * n / self.fs_in)

        if self.window == 'hann':
            h *= np.hanning(self.numtaps)
        elif self.window == 'hamming':
            h *= np.hamming(self.numtaps)
        else:
            raise ValueError("window must be 'hann' or 'hamming'")

        # Unity DC gain
        h /= np.sum(h)

        return h

    def filter(self, x):
        """
        Apply the anti-aliasing filter to the input data.

        Parameters
        ----------
        x : numpy.ndarray
            Input time-ordered data sampled at fs_in

        Returns
        -------
        filtered : numpy.ndarray
            Low-pass filtered data with the same length as the input

        """
        filtered = np.convolve(x, self.h, mode='same').astype(self.DT)
        return filtered

    def downsample(self, x):
        """

        Downsample the filtered data to fs_out.

        Samples are selected at regular intervals corresponding to the
        ratio between the input and output sampling frequencies. The
        input data should be low-pass filtered before downsampling to
        prevent aliasing.

        Parameters
        ----------
        x : numpy.ndarray
            Input data sampled at fs_in, typically after applying filter()

        Returns
        -------
        decimated_data : numpy.ndarray
            Downsampled data sampled at approximately fs_out.
        """
        ratio = self.fs_in / self.fs_out
        n_out = int(len(x) / ratio)
        idx = (np.arange(n_out) * ratio).astype(int)
        decimated_data = x[idx]
        return decimated_data

    def process(self, x):
        """
        Filter and downsample the input data.

        Parameters
        ----------
        x : numpy.ndarray
            Input time-ordered data sampled at fs_in.

        Returns
        -------
        decimated_x : numpy.ndarray
            Low-pass filtered and downsampled data sampled at approximately fs_out.
        """
        
        x_filt = self.filter(x)
        decimated_x = self.downsample(x_filt)
        return decimated_x
