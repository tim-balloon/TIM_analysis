import numpy as np
from scipy.linalg import svd
from scipy.optimize import least_squares
from photutils import find_peaks
from scipy.signal import find_peaks as fp

from astropy.stats import sigma_clipped_stats
from IPython import embed

class beam(object):
    """
    Class to fit the beam with a gaussian model

    Parameters
    ----------

    Returns
    -------
    """

    def __init__(self, data, param = None, fact=20):
        """
        Create an instance of the beam class. 

        Parameters
        ----------
        data: 2D array
            the map in which to find and fit the Gaussian beam
        params: array, optional
            Initial guess on the [amp, xo, yo, sigma_x, sigma_y, theta] paramters of the Gaussian.

        Returns
        -------

        """

        self.data = data
        self.param = param
        self.fact = fact

        shape = self.data.shape
    
        self.xgrid = np.arange(shape[1])
        self.ygrid = np.arange(shape[0])
        
        self.xy_mesh = np.meshgrid(self.xgrid,self.ygrid)

    def multivariate_gaussian_2d(self, params):
        """
        Compute the sum of one or more 2D rotated Gaussian functions on the grid
        defined by self.xy_mesh, and return the result as a flattened array.

        Each Gaussian is parameterized by 6 consecutive values in `params`:
            params = [amp, xo, yo, sigma_x, sigma_y, theta]
        
        Parameters
        ----------
        params : array
            Current model parameter values

        Returns
        -------
        multivariate_gaussian: array
            the final result as a 1D array 
        
        """

        # Unpack the 2D (x, y) coordinate grids from the class (meshgrid arrays)
        (x, y) = self.xy_mesh

        # Number of Gaussians: params contains 6 parameters for each Gaussian
        n_gaussians = int(np.size(params) / 6)

        # Loop over each Gaussian
        for i in range(n_gaussians):

            # Index of the first parameter for the i-th Gaussian
            j = i * 6

            # Extract Gaussian parameters
            amp = params[j]         # amplitude
            xo = float(params[j+1]) # center in x
            yo = float(params[j+2]) # center in y
            sigma_x = params[j+3]   # standard deviation along x
            sigma_y = params[j+4]   # standard deviation along y
            theta = params[j+5]     # rotation angle in radians

            # Coefficients of the exponent for a rotated 2D Gaussian
            # These come from the standard form of an elliptic rotated Gaussian.
            a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
            b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
            c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)

            # Compute the Gaussian on the full x/y grid
            gaussian_2d = amp * np.exp(
                -(a * (x - xo)**2 + 2*b*(x - xo)*(y - yo) + c * (y - yo)**2)
            )

            # If this is the first Gaussian, initialize the model
            if i == 0:
                multivariate_gaussian = gaussian_2d
            # Otherwise, add this Gaussian to the sum
            else:
                multivariate_gaussian += gaussian_2d

        # Return the final result as a 1D array (useful for least-squares fitting)
        return np.ravel(multivariate_gaussian)

    def residuals(self, params, x, y, err, maxv):
        """
        Compute the residuals between the Gaussian model and the data.

        Parameters
        ----------
        params : array
            Current model parameter values (passed by least_squares).
        x : array
            Not actually used — kept for API compatibility.
        y : array
            The data values (flattened).
        err : array
            The error values for each data point.
        maxv : float
            Maximum value of the data (used to set a threshold).

        Returns
        -------
        residuals : array
            (data - model) / error, evaluated only for pixels above threshold.
        """

        # Compute the model on the grid (flattened)
        dat = self.multivariate_gaussian_2d(params)

        # Select only pixels with values >= 20% of the maximum
        # This masks out noisy/low-signal regions from the fit.
        index, = np.where(y >= 0.2 * maxv)

        # Compute normalized residuals for selected pixels
        return (y[index] - dat[index]) / err[index]

    def peak_finder(self, map_data, mask_pf = False, fact=10, sigma_clip=3.0):

        """
        Find peaks in a 2D map, build Gaussian initial guesses,
        and update an exclusion mask to avoid double detections.

        Parameters
        ----------
        map_data : 2D array
            the map in which to find the peaks

        mask_pf : bool or array-like, optional
            Initial mask to exclude regions during peak finding. Default is False (no mask).

        fact : int
            Factor to determine the peak-finding box size
        
        sigma_clip: float
            The number of standard deviations to use for both the lower and upper clipping limit.

        Returns
        -------

        """

        # Get number of pixels along each grid axis
        x_lim = np.size(self.xgrid)
        y_lim = np.size(self.ygrid)

        # Peak-finding box size (height, width),
        # roughly map-size/20 in each direction
        bs = np.array([ np.max((int(np.floor(y_lim / fact)),12)),
                        np.max((int(np.floor(x_lim / fact)),12)) ])
        
        # Compute sigma-clipped statistics of the full dataset
        mean, median, std = sigma_clipped_stats(self.data, sigma=sigma_clip)

        # Detection threshold = median + 5σ
        threshold = median + (5. * std)
        print(threshold)

        # --- Build a mask for NaN values ---
        if hasattr(self, 'nanmask'):
            nanmask = self.nanmask
        else:
            nanmask = np.isnan(map_data)

        # --- Peak detection ---
        if mask_pf is False:
            # No mask provided → create an empty mask
            mask_pf = np.zeros_like(map_data, dtype=bool)

        # Combine user mask and NaN mask
        combined_mask = mask_pf | nanmask

        # Peak finding with NaN masking
        tbl = find_peaks(map_data, threshold, box_size=bs, mask=combined_mask)

        # Formatting for printing the peak values
        # Only keep peaks with amplitude above a threshold
        if tbl is None or len(tbl) == 0: return 0
        tbl = tbl[tbl['peak_value'] > threshold]  
        tbl['peak_value'].info.format = '%.8g'

        # Arrays to collect initial Gaussian guesses
        guess = np.array([])

        # Arrays that store the x,y positions of detected peaks
        x_i = np.array([])
        y_i = np.array([])

        # Loop over detected peaks
        for i in range(len(tbl['peak_value'])):

            # Construct initial guess parameters for a 2D Gaussian:
            #   amplitude, x0, y0, sigma_x, sigma_y, correlation
            guess_temp = np.array([
                tbl['peak_value'][i],
                self.xgrid[tbl['x_peak'][i]],
                self.ygrid[tbl['y_peak'][i]],
                1., 1., 0.
            ])

            # Append these parameters to the global guess array
            guess = np.append(guess, guess_temp)

            # Extract x,y index positions of the peak
            index_x = self.xgrid[tbl['x_peak'][i]]
            index_y = self.ygrid[tbl['y_peak'][i]]

            # Store peak positions
            x_i = np.append(x_i, index_x)
            y_i = np.append(y_i, index_y)

            # Mark a rectangular region around the peak as "used"
            # to prevent re-identifying peaks in the same area
            mask_pf[index_y - bs[1] : index_y + bs[1],
                    index_x - bs[0] : index_x + bs[0]] = True

            # Initialize or append to self.param and self.mask
            if self.param is None:
                # First peak detected → initialize parameter array
                self.param = guess_temp
                self.mask = mask_pf.copy()
            else:
                # Additional peaks → append parameters and update mask
                self.param = np.append(self.param, guess_temp)
                self.mask = np.logical_or(self.mask, mask_pf)

    def fit(self):

        """
        Performs a Levenberg–Marquardt least-squares fit of the model defined in
        self.residuals() to the data stored inside the class.

        Parameters
        ----------

        Returns
        -------
        p : OptimizeResult
            Result object from scipy.optimize.least_squares containing the fitted parameters.
        var : ndarray
            Estimated covariance matrix of the fitted parameters, derived from the Jacobian.
        """

        try:
            # Print the initial guess parameters
            #print('PARAM', self.param)

            # Perform the least-squares optimization.
            # - self.residuals: computes (model - data)
            # - x0=self.param: initial guess for all parameters
            # - args: additional arguments passed to the residual function
            # - method='lm': use Levenberg–Marquardt (requires dense Jacobian)

            # Flatten data
            # Mask of finite pixels
            
            # Masked data
            #data_flat = flat_data[mask]

            flat_data = np.ravel(self.data)
            weights = np.isfinite(flat_data)
            flat_data = np.nan_to_num(flat_data, nan=0.0)
            
            p = least_squares(
                self.residuals,
                x0=self.param,
                args=(
                    self.xy_mesh,                 # meshgrid of (x, y)
                    flat_data,          # flattened data array
                    weights,  
                    np.amax(flat_data),            # maximum of the data (often used for normalizing)
                ),
                method='lm'
            )
                
            # ----------------------------------------------------------------------
            # Compute covariance matrix using the Jacobian from the optimized fit.
            # ----------------------------------------------------------------------

            # Perform the SVD of the Jacobian: J = U * diag(s) * VT
            # We do not need U, only singular values s and VT.
            _, s, VT = svd(p.jac, full_matrices=False)

            # Define a threshold to filter out tiny singular values,
            # which avoids numerical instabilities when inverting.
            threshold = np.finfo(float).eps * max(p.jac.shape) * s[0]

            # Keep only singular values larger than the threshold.
            s = s[s > threshold]

            # Keep the corresponding rows of VT.
            # (Number of retained rows = number of retained singular values)
            VT = VT[:s.size]

            # Compute covariance matrix of fitted parameters:
            # Cov(θ) = (J^T J)^(-1)
            # Using SVD: (V diag(s^2) V^T)^(-1) = V diag(1/s^2) V^T
            var = np.dot(VT.T / s**2, VT)

            return p, var

        # --------------------------------------------------------------------------
        # Error handling
        # --------------------------------------------------------------------------

        except np.linalg.LinAlgError:
            # Raised when the SVD fails or the Jacobian is singular → fit diverged
            msg = 'Fit not converged'
            return msg, 0

        except ValueError:
            # Typically raised when LM receives too many parameters
            # or when the residual function shape is inconsistent.
            msg = 'Too Many parameters'
            return msg, 0

    def beam_fit(self, mask_pf=False):
        """
        Main function to fit one or more 2D Gaussian beams to a 2D map.

        Parameters
        ----------
        mask_pf : bool or array-like, optional
            Initial mask to exclude regions during peak finding. Default is False (no mask).

        Returns
        -------
        fit_data : 2D array
            The fitted Gaussian map (sum of all fitted Gaussians).
        fit_param : array
            Fitted parameters for all Gaussians [amp, xo, yo, sigma_x, sigma_y, theta,...].
        var : 2D array
            Covariance matrix of the fitted parameters.
            If the fit did not converge, returns a message and zeros.
        """

        # -----------------------------
        # Check if initial Gaussian parameters already exist
        # -----------------------------
        if self.param is not None:
            # Already have initial parameters → count number of peaks
            peak_found = np.size(self.param) / 6
            force_fit = True   # We already have parameters, so we force a single fit
        else:
            # No initial parameters → find peaks in the map and generate guesses
            
            self.peak_finder(map_data=self.data, mask_pf=mask_pf, fact=self.fact)

            peak_number_ini = np.size(self.param) / 6  # initial number of peaks found
            peak_found = peak_number_ini
            force_fit = False  # May need to iterate to find additional peaks

        # -----------------------------
        # Iteratively fit Gaussians until no new peaks are found
        # -----------------------------
        while peak_found > 0:
            # Perform the least-squares Gaussian fit
            
            
            fit_param, var = self.fit()
            
            
            # Check if the fit converged
            if isinstance(fit_param, str):
                # Fit failed, exit loop
                msg = 'fit not converged'
                break
            else:
                # Compute the fitted Gaussian map from the fitted parameters
                
                fit_data = self.multivariate_gaussian_2d(fit_param.x).reshape(np.outer(self.ygrid, self.xgrid).shape)
                

                if force_fit is False:
                    # Subtract the fitted map from the original to find residual peaks
                    res = self.data - fit_data
                    
                    # Look for additional peaks in the residual
                    self.peak_finder(map_data=res, fact=self.fact)
                    
                    # Update the number of new peaks found
                    peak_number = np.size(self.param) / 6
                    
                    peak_found = peak_number - peak_number_ini
                    peak_number_ini = peak_number
                else:
                    # If parameters were provided initially, stop after one fit
                    peak_found = -1

        # -----------------------------
        # Return results
        # -----------------------------
        if isinstance(fit_param, str):
            # Fit did not converge → return message and zeros
            return msg, 0, 0
        else:
            # Successful fit → return fitted map, parameters, and covariance
            print('PARAM_FIT', fit_param.x)
            return fit_data, fit_param.x, var


class Beam1D(object):
    """
    Fit one or more 1D Gaussians to a collapsed map.
    """

    def __init__(self, data, param=None, threshold_frac=0.2, fact=20):
        """
        Parameters
        ----------
        data : 1D array
            The collapsed map
        param : list or array, optional
            Initial guess for multiple Gaussians:
            [amp1, x01, sigma1, amp2, x02, sigma2, ...]
        n_peaks : int, optional
            Number of peaks to detect automatically
        threshold_frac : float
            Fraction of maximum to threshold peaks
        """
        self.data = np.array(data)
        self.param = param
        self.fact = fact
        self.xgrid = np.arange(len(self.data))
        self.threshold_frac = threshold_frac

    # ------------------------------------------------------------------
    # Estimate initial guesses from peaks
    # ------------------------------------------------------------------

    def peak_finder(self, map_data, mask_pf = False, fact=10, sigma_clip=3.0):

        # Get number of pixels along each grid axis
        x_lim = np.size(self.xgrid)

        # Peak-finding box size (height, width),
        # roughly map-size/20 in each direction
        bs = 5 #np.max((int(np.floor(x_lim / fact)),12))
        
        # Compute sigma-clipped statistics of the full dataset
        mean, median, std = sigma_clipped_stats(self.data, sigma=sigma_clip)
        # Detection threshold = median + 5σ
        threshold = np.max(( median + (5. * std), 0))

        data_for_fit = np.nan_to_num(self.data, nan=-np.inf)
        # --- Peak detection ---
        if mask_pf is False: mask_pf = np.zeros_like(map_data, dtype=bool)

        data_for_fit[mask_pf] = -np.inf # or np.nan if you prefer

        # Peak finding with NaN masking
        peaks, properties = fp(map_data, threshold=threshold, distance=bs, height=0)

        # Formatting for printing the peak values
        # Only keep peaks with amplitude above a threshold
        if len(peaks) == 0: return 0
        #tbl = tbl[tbl['peak_value'] > threshold]  

        # Arrays to collect initial Gaussian guesses
        guess = np.array([])

        # Arrays that store the x,y positions of detected peaks
        x_i = np.array([])

        # Loop over detected peaks
        for i in range(len(peaks)):

            # Construct initial guess parameters for a 2D Gaussian:
            #   amplitude, x0, sigma_x, correlation
            guess_temp = np.array([
                properties['peak_heights'][i],
                self.xgrid[peaks[i]],
                1., 0.
            ])

            # Append these parameters to the global guess array
            guess = np.append(guess, guess_temp)

            # Extract x,y index positions of the peak
            index_x = self.xgrid[peaks[i]]

            # Store peak positions
            x_i = np.append(x_i, index_x)

            #----
            for peak, left_th, right_th in zip(peaks, properties['left_thresholds'], properties['right_thresholds']):
            # Left threshold index
                left_idx = peak
                while left_idx > 0 and map_data[left_idx] > left_th:
                    left_idx -= 1

                # Right threshold index
                right_idx = peak
                while right_idx < len(map_data)-1 and map_data[right_idx] > right_th:
                    right_idx += 1
                
                #----
                # Mark a rectangular region around the peak as "used"
                # to prevent re-identifying peaks in the same area
                mask_pf[left_idx:right_idx+1] = True

            # Initialize or append to self.param and self.mask
            if self.param is None:
                # First peak detected → initialize parameter array
                self.param = guess_temp
                self.mask = mask_pf.copy()
            else:
                # Additional peaks → append parameters and update mask
                self.param = np.append(self.param, guess_temp)
                self.mask = np.logical_or(self.mask, mask_pf)

    # ------------------------------------------------------------------
    # Sum of 1D Gaussians
    # ------------------------------------------------------------------
    def gaussian_1d_sum(self, params):
        n_gaussians = len(params) // 4
        y_model = np.zeros_like(self.xgrid, dtype=float)
        for i in range(n_gaussians):
            amp, x0, sigma, _ = params[i*4:(i+1)*4]
            y_model += amp * np.exp(-0.5 * ((self.xgrid - x0)/sigma)**2)
        return y_model

    # ------------------------------------------------------------------
    # Residuals
    # ------------------------------------------------------------------
    def residuals(self, params, y, err, maxv): #x ? 

        # Compute the model on the grid (flattened)
        dat = self.gaussian_1d_sum(params)

        # Select only pixels with values >= 20% of the maximum
        # This masks out noisy/low-signal regions from the fit.
        index, = np.where(y >= 0.1 * maxv)

        # Compute normalized residuals for selected pixels
        return (y[index] - dat[index]) / err[index] #return self.data - self.gaussian_1d_sum(params)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self):

        data_for_fit = np.nan_to_num(self.data, nan=-np.inf)
        weights = np.isfinite(self.data)

        p = least_squares(
            self.residuals,
            x0=self.param,
            args = (data_for_fit, weights, np.amax(data_for_fit)), 
            method='lm'
        )

        try:
            _, s, VT = svd(p.jac, full_matrices=False)
            threshold = np.finfo(float).eps * max(p.jac.shape) * s[0]
            s = s[s > threshold]
            VT = VT[:s.size]
            var = np.dot(VT.T / s**2, VT)
        except Exception:
            var = np.zeros((len(p.x), len(p.x)))

        return p, var

    # ------------------------------------------------------------------
    # Main beam fit
    # ------------------------------------------------------------------
    def beam_fit(self, mask_pf = False):

        if self.param is not None: 
            #self.param = self.estimate_initial_guess()
            peak_number_ini = np.size(self.param) / 4
            force_fit = True

        else: 

            self.peak_finder(map_data=self.data, mask_pf = mask_pf)
            peak_number_ini = np.size(self.param) / 4
            peak_found = peak_number_ini
            force_fit = False

        while peak_found > 0: 
            
            fit_param, var = self.fit()
                      
            # Check if the fit converged
            if isinstance(fit_param, str):
                # Fit failed, exit loop
                msg = 'fit not converged'
                break
            else: 

                fit_data = self.gaussian_1d_sum(fit_param.x)
                
                if force_fit is False:
                    # Subtract the fitted map from the original to find residual peaks
                    res = self.data - fit_data
                    
                    # Look for additional peaks in the residual
                    self.peak_finder(map_data=res, fact=self.fact)
                    
                    # Update the number of new peaks found
                    peak_number = np.size(self.param) / 4
                    
                    peak_found = peak_number - peak_number_ini
                    peak_number_ini = peak_number

                else: 
                    peak_found = -1

        if isinstance(fit_param, str):
            # Fit did not converge → return message and zeros
            return msg, 0, 0
        else:
            # Successful fit → return fitted map, parameters, and covariance
            print('PARAM_FIT', fit_param.x)
            return fit_data, fit_param.x, var
        
if __name__ == "__main__":

    import numpy as np
    from astropy.io import fits
    import matplotlib.pyplot as plt
    from astropy.visualization import ZScaleInterval
    from astropy.wcs import WCS

    for extension in (0,1):

        #map_value = fits.getdata('/home/mvancuyck/Desktop/TIM_analysis/timestream_maker/fits_and_hdf5/cube_2sources_separated_by_150.8arcsecs_with_1xbigger_sigma_PSF.fits', )[0]#ext=0)[0]
        map_value = fits.getdata('../fits_and_hdf5/scanned_map_TOD_on_2_sources_separated_by_150.8_with_1xbigger_sigma_PSF_LW.fits', ext=extension)[0] 
        hdr = fits.getheader('../fits_and_hdf5/scanned_map_TOD_on_2_sources_separated_by_150.8_with_1xbigger_sigma_PSF_LW.fits', ext=extension)
        wcs3d = WCS(hdr) 
        wcs = wcs3d.slice((extension, slice(None), slice(None)))
        valid = ~np.isnan(map_value)
        # find rows & columns containing at least one valid pixel
        rows = np.where(valid.any(axis=1))[0]
        cols = np.where(valid.any(axis=0))[0]
        # crop
        map_value = map_value[rows.min():rows.max(), cols.min():cols.max()]

        zscale = ZScaleInterval()
        vmin, vmax = zscale.get_limits(map_value)

        fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': wcs})
        ax.set_title('extension')
        im = ax.imshow(map_value, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, orientation='vertical',)
        ax.set_xlabel('X pixel')
        ax.set_ylabel('Y pixel')
                         
        beam_value = beam(map_value, )#param = self.beamparam
        beam_map = beam_value.beam_fit()
        param = beam_map[1]
        if isinstance(beam_map[0], str): print(beam_map[0])
        else: 
            plt.figure(figsize=(8, 6))
            plt.contour(beam_map[0], levels=10, colors='red')
            plt.imshow(beam_map[0], origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
            plt.colorbar(label='Amplitude')
            plt.title('2D Gaussian Fit Contours')
            plt.xlabel('X pixel')
            plt.ylabel('Y pixel')
        print('')
        
        collapsed_map = np.mean(np.nan_to_num(map_value, nan=0.0), axis=0)
        #collapsed_map = np.nan_to_num(map_value[map_value.shape[0]//2, :], nan=0.0)
        plt.figure()
        plt.plot(collapsed_map, label='data')
        b = Beam1D(collapsed_map)
        fit_profile, params, cov = b.beam_fit()
        print("Amplitude =", params[0])
        print("Center x0 =", params[1])
        print("Sigma =", params[2])
        plt.plot(fit_profile, ':',label='fit')

        #diag_mean = np.mean(np.nan_to_num(np.diag(map_value), nan=0.0))       
        plt.legend()
        

        
        
    plt.show()
