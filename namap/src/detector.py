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
        create an instance of the class to clean the detector TODs.
        Parameters
        ----------
        data: list
            detector TODs
        fs: float
            frequency sampling of the detectors
        cutoff: float
            cutoff frequency of the highpass filter     
        polynomialorder: int
            polynomial order for fitting
        despike: bool
            if True despikes the data using scipy.signal
        sigma: float
            height in std value to look for spikes 
        prominence: float
            prominence in std value to look for spikes
        DT: type
        Float precision required
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
        Function to return the cleaned TOD as numpy array
        Parameters
        ----------
        Returns
        -------
        cleaned_data: list
            list of cleaned data timestreams
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
    Class to despike the TOD
    Parameters
    ----------
    Returns
    -------
    '''
    
    def __init__(self, data):

        '''
        Create an instance of the class to despike the TOD
        Parameters
        ----------
        data: 1d array
            the TOD to be despiked
        Returns
        -------
        '''

        self.data = data

    def findpeak(self, hthres=5, pthres=0):

        '''
        This function finds the peak in the TOD.
        hthresh and pthres are measured in how many std the height (or the prominence) 
        of the peak is computed. The height of the peak is computed with respect to 
        the mean of the signal    
        Parameters
        ----------
        hthres: int
            height in sigma of the peak
        pthres: int 
            prominence of the peak in sigma. 
        
        Returns
        -------
        index: list
            list of index of the peaks in the timestreams. 
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
        Function to estimate the width of the peaks.
        Window is the parameter used by the algorith to find the minimum 
        left and right of the peak. The minimum at left and right is used
        to compute the width of the peak
        Parameters
        ----------
        peaks: list
            list of index of the peaks found in the timestream. 
        hthres: int
            height in sigma of the peak. 
        pthres: int 
            prominence of the peak in sigma. 
        window: float
            how far in samples to look to the left and right of each peak to find the local minimum
        Returns
        -------
        param[0].copy(): list
            list of peak widths
        ledge: list
            list of peaks' left edge index 
        redge: list
            list of peaks' right edge index 
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

        '''
        This function replaces the spikes data with noise realization. Noise can be gaussian
        or poissonian based on the statistic of the data
                
        Function to estimate the width of the peaks.
        Window is the parameter used by the algorith to find the minimum 
        left and right of the peak. The minimum at left and right is used
        to compute the width of the peak
        Parameters
        ----------
        hthres: int
            height in sigma of the peak. 
        pthres: int 
            prominence of the peak in sigma. 
        peaks: list
            list of index of the peaks found in the timestream. 
        widths: list
            list of peak widths
        Returns
        -------
        replaced: array
            timestream array with replaced peaks. 
        '''

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
    class for filter the detector TOD
    Parameters
    ----------
    Returns
    -------
    '''

    def __init__(self, data, cutoff, fs, DT):
        
        '''
        See data_cleaned for parameters explanantion
        Parameters
        ----------
        data: 1d array
            the data to be filtered
        cutoff: float
            the frequency of the filter
        fs: float
            the sampling frequency of array
        DT: type
            float precision required

        Returns
        -------
        '''

        self.data = data
        self.cutoff = cutoff
        self.fs = fs
        self.DT = DT

    def highpass(self, order):

        '''
        Highpass Butterworth filter.
        Parameters
        order: int
            the order of the butterworth filter
        ----------
        Returns
        b: ndarray
            Numerator polynomials of the IIR filter
        a: ndarray
            Denominator polynomials of the IIR filter
        -------
        '''
        
        nyq = 0.5*self.fs
        normal_cutoff = self.cutoff / nyq
        b, a = sgn.butter(order, normal_cutoff, btype='highpass', analog=False)
        return b, a

    def butter_highpass_filter(self, order=5):

        '''
        Data filtered with a butterworth filter 
        Parameters
        order: int
            the order of the butterworth filter
        ----------
        Returns
        filterdata: array
            filtered timestream
        -------        
        '''
        b, a = self.highpass(order)
        filterdata = sgn.lfilter(b, a, self.data)
        return filterdata

    def cosine_filter(self, f):

        '''
        Highpass cosine filter
        Parameters
        f: float
            frequency at which to evaluate the filter
        ----------
        Returns
        cosline_filter: float
            the transmission of the cosine filter at f. 
        -------
        '''

        if f < .5*self.cutoff:
            return 0
        elif 0.5*self.cutoff <= f  and f <= self.cutoff:
            return 0.5-0.5*np.cos(np.pi*(f-0.5*self.cutoff)*(self.cutoff-0.5*self.cutoff)**-1)
        elif f > self.cutoff:
            return 1
    
    def fft_filter(self, window):

        '''
        Return an fft of the despiked data using the cosine filter.
        
        Parameters
        Window: bool
            A parameter that can be true if the FFT is computed using a Hanning window. 
        ----------
        Returns
        filtereddata: array
            the filtered timestream. 
        -------
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

        '''
        Inverse FFT of cleaned FFT data calculated in the previous function.

        Parameters
        Window: bool
            A parameter that can be true if the FFT is computed using a Hanning window. 
        ----------
        Returns
        filtereddata: array
            the inverse FFT of cleaned FFT data. 
        -------
        '''

        ifft_data = np.fft.irfft(self.fft_filter(window=window), len(self.data))

        return self.DT(ifft_data)
    
class detector_trend():

    '''
    Class to detrend a TOD
    Parameters
    ----------
    Returns
    -------
    '''

    def __init__(self, data, DT):
        '''
        create an instance of the class to detrend a TOD
        Parameters
        ----------
        data: 1 array
            the timestream to remove the trend of. 
        Returns
        -------
        '''

        self.data = data
        self.DT = DT

    def polyfit(self, edge = 0, order=6):

        '''
        Function to fit a trend line to a TOD
        Parameters
        edge: int
            dimension of the list of timestreams passed to the function. 
        order: int
            order of the polynome to be fit to the timsetream.
        ----------
        Returns
        y_fin: array
            fitted data
        index_exclude: array
            index of elements to set to 0. 
        -------
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

        '''
        Function to remove the trend polynomial from the TOD
        Parameters
        edge: int
            dimension of the list of timestreams passed to the function. 
        order: int
            order of the polynome to be fit to the timsetream.
        ----------
        Returns
        fit_residual: array
            the residual between the timestream and the fit of the timestream. 
        -------
        '''
        polyres = self.polyfit(edge=edge, order=order)
        fitteddata = polyres[0]
        index = polyres[1]

        zero_data = self.data.copy()
        if(len(index)>0): zero_data[index] = 0.

        return -fitteddata+zero_data

class sigma_clipping():

    '''
    Class to measure the variance in a timestream, and remove it from further analysis if its variance is too low or too high. 
    Parameters
    ----------
    Returns
    -------
    '''

    def __init__(self, data):
        '''
        Create an instance of the class to measure the variance in a timestream
        Parameters
        ----------
        data: 1d array
            the detector timestream
        Returns
        -------
        '''

        self.data = np.float32(data)
    
    def clipping(self, low_thresh, high_thresh):
        '''
        Measure the variance of a timestream and return True if its variance is within the thresholds.
        Parameters
        ----------
        low_trhesh: float
            the lower threshold in sigma for the timestream variance
        
        high_trhesh: float
            the higher threshold in sigma for the timestream variance
        Returns
        -------
        reject: bool
            If True, the timestream variance is inside the thresholds
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
        Returns
        -------
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
        I: array
            I phase of the kid 
        Q: array: 
            Q phase of the kid
        ----------
        Returns
        -------
        phi: array
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
        I: array
            I phase of the kid 
        Q: array: 
            Q phase of the kid
        ----------
        Returns
        -------
        mag: array
            the magnitude
        '''

        return np.sqrt(I**2+Q**2 )

    """
    def interpolation_roach(self, data, bins, sampling, DT, IT):

        '''
        data: values that need to be interpolated
        bins: bins of data coming from the pps signal
        sampling: frequency sampling of the detectors 
        Parameters
        ----------
        Returns
        -------
        '''
        '''
        start = np.append(0, np.cumsum(bins[:-1]))
        end = np.cumsum(bins)
        ln = np.linspace(start, end-1, int(sampling))
        idx = np.reshape(np.transpose(ln), np.size(ln))
        idx_plus = np.append(idx[:-1]+1, idx[-1])
        return (data[idx_plus.astype(int)]-data[idx.astype(int)])*(idx-idx.astype(int))+data[idx.astype(int)]
        '''
        # Ensure bins are in DT
        bins = np.array(bins, dtype=DT)
        
        # Start and end positions
        start = np.append(DT(0), np.cumsum(bins[:-1], dtype=DT))
        end = np.cumsum(bins, dtype=DT)
        
        # Linear spacing within each bin
        ln = np.linspace(start, end - DT(1), int(sampling), dtype=DT)
        
        # Flatten index array
        idx = np.reshape(np.transpose(ln), np.size(ln)).astype(DT)
        
        # idx_plus as integer indices for data access
        idx_plus = np.append(idx[:-1] + DT(1), idx[-1]).astype(DT)
        
        # Integer indices for data
        idx_int = idx.astype(DT)
        
        # Interpolation in chosen DT
        
        result = (data[idx_plus.astype(IT)] - data[idx_int.astype(IT)]) * (idx - idx_int) + data[idx_int.astype(IT)]

        return result.astype(DT)
    """

    """
    def det_time(self, path, roach_number, frames, ctime_start, ctime_end, sampling):
        '''
        get the time timestreams. 
        Need implementation for TIM
        Parameters
        ----------
        Returns
        -------
        '''

        roach_string_ctime = ['ctime_packet_roach' + f'{roach}' for roach in roach_number] #roach_string_ctime = 'ctime_packet_roach'+str(int(roach_number))
        pps_roach_string = ['pps_count_roach' + f'{roach}' for roach in roach_number] #pps_roach_string = 'pps_count_roach'+str(int(roach_number))

        num_frames= int(frames[1]-frames[0])
        
        ctime_roach_list, spf_ctime_roach = ld.data_value.load(path,roach_string_ctime, first_frame=frames[0], num_frames=num_frames) #d.getdata(roach_string_ctime, first_frame=frames[0], num_frames=num_frames)
        pps_list, spf_pps = ld.data_value.load(path, pps_roach_string, first_frame=frames[0], num_frames=num_frames)#d.getdata(pps_roach_string, first_frame=frames[0], num_frames=num_frames)
        
        ctime_roach_renormed = []
        bins_list = [] 
        for i in range(len(roach_number)):
            if pps_list.ndim == 1:  # If it's a 1D array
                pps = pps_list  # Assign the entire array
                ctime_roach = ctime_roach_list
            else:  # If it's a 2D array
                pps = pps_list[i, :]
                ctime_roach = ctime_roach_list[i,:]

            bn = np.bincount(pps)
            bins = bn[bn>0]
            bins_list.append(bins)
            
            if bins[0] < 350:
                pps = pps[bins[0]:]
                ctime_roach = ctime_roach[bins[0]:]
            if bins[-1] < 350:
                pps = pps[:-bins[-1]]
                ctime_roach = ctime_roach[:-bins[-1]]
                
            ctime_roach =ctime_roach* 1e-2
            ctime_roach += 1570000000
            pps_duration =  pps[-1]-pps[0]+1
            pps_final =  pps[0]+np.arange(0, pps_duration, 1/sampling) 
            ctime_roach = self.interpolation_roach(ctime_roach, bins[bins>350], sampling)
            ctime_roach += pps_final

            ctime_roach_renormed.append(ctime_roach)

        return np.asarray(ctime_roach_renormed), np.asarray(bins_list)
    """

class AntiAliasingFilter():
    """
    Anti-aliasing filter for downsampling TODs.
    """

    def __init__(self, fs_in, fs_out, DT, fc=None, numtaps=257, window='hann'):
        """
        Parameters
        ----------
        fs_in : float
            Input sampling frequency [Hz]
        fs_out : float
            Output sampling frequency [Hz]
        fc : float or None
            Cutoff frequency [Hz]. If None, uses 0.45 * fs_out
        numtaps : int
            Length of FIR filter (odd recommended)
        window : str
            'hann' or 'hamming'
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
        Design linear-phase FIR low-pass filter
        Parameters
        ----------
        Returns
        -------
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
        Apply anti-aliasing filter
        Parameters
        ----------
        x: 1D array
            the detector data to be filtered
        Returns
        -------
        filtered: 1D array
            the low-pass filtered data
        """
        filtered = np.convolve(x, self.h, mode='same').astype(self.DT)
        return filtered

    def downsample(self, x):
        """
        Downsample filtered signal.
        Uses nearest-neighbor time picking (safe after LPF).
        Parameters
        ----------
        x: 1D array
            the data to be decimated
        Returns
        -------
        decimated_data: 1D array
            the downsampled data
        """
        ratio = self.fs_in / self.fs_out
        n_out = int(len(x) / ratio)
        idx = (np.arange(n_out) * ratio).astype(int)
        decimated_data = x[idx]
        return decimated_data

    def process(self, x):
        """
        Filter + downsample
        Parameters
        ----------
        x: 1D array
            the detector data to be filtered
        Returns
        -------
        filtered: 1D array
            the low-pass filtered data
        """
        x_filt = self.filter(x)
        return self.downsample(x_filt)
