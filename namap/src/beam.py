import numpy as np
from scipy.linalg import svd
from scipy.optimize import least_squares
from photutils import find_peaks
from scipy.signal import find_peaks as fp
from astropy.stats import sigma_clipped_stats
from IPython import embed

class beam(object):
    """
    Fit one or more 2D Gaussian models to a map.

    The class identifies peaks in the input map, uses them to initialize
    Gaussian components, and iteratively fits the resulting model to the
    data. Multiple rotated 2D Gaussian components can be fitted.

    Parameters
    ----------

    Returns
    -------
    """

    def __init__(self, data, param = None, fact=20, mask=False):
        """
        Create an instance of the class for fitting 2D Gaussian beams.

        Parameters
        ----------
        data : numpy.ndarray
            2D map in which to identify and fit Gaussian beams.
        param : numpy.ndarray, optional
            Initial Gaussian parameters. Each Gaussian is described by six
            parameters in the order
            [amp, xo, yo, sigma_x, sigma_y, theta].
        fact : int, optional
            Factor used to determine the peak-finding box size.
        mask : bool or numpy.ndarray, optional
            Initial mask used to exclude pixels during peak finding.
        
        Returns
        -------
        """

        self.data = data
        self.param = param
        self.fact = fact
        self.mask = mask
        shape = self.data.shape
    
        self.xgrid = np.arange(shape[1])
        self.ygrid = np.arange(shape[0])
        
        self.xy_mesh = np.meshgrid(self.xgrid,self.ygrid)

    def multivariate_gaussian_2d(self, params):
        """
        Compute the sum of one or more rotated 2D Gaussian functions.

        Each Gaussian is described by six consecutive parameters in
        params:

        [amp, xo, yo, sigma_x, sigma_y, theta]

        where theta is the rotation angle in radians.

        Parameters
        ----------
        params : numpy.ndarray
            Gaussian model parameters. Six consecutive values are used
            for each Gaussian component.

        Returns
        -------
        multivariate_gaussian : numpy.ndarray
            Flattened 2D map containing the sum of all Gaussian components.
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
        Compute normalized residuals between the Gaussian model and data.

        Only pixels with values above 10% of the maximum data value are
        included in the residual calculation.

        Parameters
        ----------
        params : numpy.ndarray
            Current Gaussian model parameters.
        x : numpy.ndarray
            Coordinate grid. This argument is retained for compatibility
            with the least-squares fitting interface and is not used directly.
        y : numpy.ndarray
            Flattened data values.
        err : numpy.ndarray
            Error or weight values associated with each data point.
        maxv : float
            Maximum value of the data, used to define the fitting threshold.

        Returns
        -------
        residuals : numpy.ndarray
            Normalized residuals (data - model) / error for pixels
            satisfying the fitting threshold.
        """

        # Compute the model on the grid (flattened)
        dat = self.multivariate_gaussian_2d(params)

        # Select only pixels with values >= 20% of the maximum
        # This masks out noisy/low-signal regions from the fit.
        index, = np.where(y >= 0.1 * maxv)

        # Compute normalized residuals for selected pixels
        return (y[index] - dat[index]) / err[index]

    def peak_finder(self, map_data, fact=10, sigma_clip=3.0):
        """
        Identify peaks in a 2D map and generate Gaussian initial guesses.

        The peak-finding threshold is defined as the sigma-clipped median
        plus five times the sigma-clipped standard deviation. Detected peaks
        are converted into initial Gaussian parameters and surrounding
        regions are added to the exclusion mask to avoid duplicate
        detections.

        Parameters
        ----------
        map_data : numpy.ndarray
            2D map in which to identify peaks.
        fact : int, optional
            Factor used to determine the size of the peak-finding box.
            The box size is approximately the map dimensions divided by
            fact, with a minimum size of 12 pixels.
        sigma_clip : float, optional
            Number of standard deviations used for sigma-clipped statistics.
            
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

        # --- Build a mask for NaN values ---
        if hasattr(self, 'nanmask'):
            nanmask = self.nanmask
        else:
            nanmask = np.isnan(map_data)

        # --- Peak detection ---
        if self.mask is False:
            # No mask provided → create an empty mask
            mask_pf = np.zeros_like(map_data, dtype=bool)
        else: mask_pf = self.mask.copy()

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
            ymin = max(0, index_y - bs[1])
            xmin = max(0, index_x - bs[0])

            ymax = min(mask_pf.shape[0], index_y + bs[1])
            xmax = min(mask_pf.shape[1], index_x + bs[0])

            mask_pf[ymin:ymax, xmin:xmax] = True

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
        Fit the Gaussian model to the input map using least squares.

        The fit uses the Levenberg-Marquardt algorithm and the initial
        parameters stored in ``self.param``. The covariance matrix of the
        fitted parameters is estimated from the Jacobian using its
        singular-value decomposition.

        Parameters
        ----------

        Returns
        -------
        p : scipy.optimize.OptimizeResult or str
            Result returned by ``scipy.optimize.least_squares`` containing
            the fitted parameters. A string is returned if the fit fails.
        var : numpy.ndarray or int
            Estimated covariance matrix of the fitted parameters, or ``0``
            if the fit fails.
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
            weights = np.isfinite(flat_data).astype(int)
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

    def beam_fit(self):

        """
        Fit one or more 2D Gaussian beams to the input map.

        If initial Gaussian parameters are provided through ``self.param``,
        a single fit is performed. Otherwise, peaks are first identified
        and fitted iteratively. After each fit, the fitted model is
        subtracted from the map and additional peaks are searched for in
        the residual.

        Returns
        -------
        fit_data : numpy.ndarray or str
            2D map containing the sum of the fitted Gaussian components.
            If the fit does not converge, a status message is returned.
        fit_param : numpy.ndarray or int
            Fitted Gaussian parameters in the order
            [amp, xo, yo, sigma_x, sigma_y, theta, ...].
        var : numpy.ndarray or int
            Covariance matrix of the fitted parameters. Returns 0 if
            the fit does not converge.
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
            
            self.peak_finder(map_data=self.data, fact=self.fact)

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
            #print('PARAM_FIT', fit_param.x)
            return fit_data, fit_param.x, var

class Beam1D(object):
    """
    Fit one or more 1D Gaussian functions to a collapsed map.

    Parameters
    ----------
    data : numpy.ndarray
        1D collapsed map to fit.
    param : numpy.ndarray, optional
        Initial Gaussian parameters. Each Gaussian is described by four
        consecutive values in the order
        ``[amp, x0, sigma, extra]``.
    threshold_frac : float, optional
        Fraction of the maximum value used to define the fitting threshold.
    fact : int, optional
        Factor used to determine the peak-finding box size.
    """

    def __init__(self, data, param=None, threshold_frac=0.2, fact=20):
        """
        Create an instance of the Beam1D class.

        Parameters
        ----------
        data : numpy.ndarray
            1D collapsed map to fit.
        param : numpy.ndarray, optional
            Initial Gaussian parameters. Each Gaussian is described by four
            consecutive values in the order
            ``[amp, x0, sigma, extra]``.
        threshold_frac : float, optional
            Fraction of the maximum value used to define the fitting
            threshold.
        fact : int, optional
            Factor used to determine the peak-finding box size.
        """

        self.data = np.array(data)
        self.param = param
        self.fact = fact
        self.xgrid = np.arange(len(self.data))
        self.threshold_frac = threshold_frac

    def peak_finder(self, map_data, mask_pf=False, fact=10, sigma_clip=3.0):
        """
        Find peaks in a 1D vector and generate initial Gaussian parameters.

        Peaks are identified using a threshold based on sigma-clipped
        statistics. For each detected peak, an initial Gaussian parameter
        set is generated and the region around the peak is added to the
        exclusion mask.

        Parameters
        ----------
        map_data : numpy.ndarray
            1D vector in which to identify peaks.
        mask_pf : bool or numpy.ndarray, optional
            Initial mask used to exclude regions during peak finding.
            Default is ``False``.
        fact : int, optional
            Factor used to determine the peak-finding box size.
        sigma_clip : float, optional
            Number of standard deviations used for the sigma-clipped
            statistics. Default is 3.0.

        """

        # Get the number of pixels along the 1D grid.
        x_lim = np.size(self.xgrid)

        # Define the minimum separation between detected peaks.
        bs = 5 #np.max((int(np.floor(x_lim / fact)),12))
        
        # Compute sigma-clipped statistics of the full dataset.
        mean, median, std = sigma_clipped_stats(
            self.data,
            sigma=sigma_clip
        )

        # Define the detection threshold as the larger of
        # median + 5 sigma and zero.
        threshold = np.max((median + (5. * std), 0))

        # Replace NaN values with -inf so they cannot contribute to the fit.
        data_for_fit = np.nan_to_num(self.data, nan=-np.inf)

        # Initialize or copy the peak-finding mask.
        if self.mask is False:
            mask_pf = np.zeros_like(map_data, dtype=bool)
        else:
            mask_pf = self.mask.copy()
        
        data_for_fit[mask_pf] = -np.inf

        # Find peaks above the detection threshold.
        peaks, properties = fp(
            map_data,
            threshold=threshold,
            distance=bs,
            height=0
        )

        # Stop if no peaks are detected.
        if len(peaks) == 0:
            return 0

        # Initialize the array containing the Gaussian parameter guesses.
        guess = np.array([])

        # Initialize the array containing the detected peak positions.
        x_i = np.array([])

        # Build an initial Gaussian parameter set for each detected peak.
        for i in range(len(peaks)):

            # Initial Gaussian parameters:
            # amplitude, x0, sigma, and an additional parameter.
            guess_temp = np.array([
                properties['peak_heights'][i],
                self.xgrid[peaks[i]],
                1.,
                0.
            ])

            # Append the parameters to the global initial-guess array.
            guess = np.append(guess, guess_temp)

            # Get the position of the detected peak.
            index_x = self.xgrid[peaks[i]]

            # Store the peak position.
            x_i = np.append(x_i, index_x)

            # Determine the region around each peak using the left and
            # right threshold positions.
            for peak, left_th, right_th in zip(
                peaks,
                properties['left_thresholds'],
                properties['right_thresholds']
            ):

                # Find the left boundary of the peak region.
                left_idx = peak
                while left_idx > 0 and map_data[left_idx] > left_th:
                    left_idx -= 1

                # Find the right boundary of the peak region.
                right_idx = peak
                while right_idx < len(map_data)-1 and map_data[right_idx] > right_th:
                    right_idx += 1
                
                # Mark the peak region as used to prevent it from being
                # identified again.
                mask_pf[left_idx:right_idx+1] = True

            # Initialize or update the Gaussian parameter array and mask.
            if self.param is None:
                # First detected peak.
                self.param = guess_temp
                self.mask = mask_pf.copy()
            else:
                # Additional detected peak.
                self.param = np.append(self.param, guess_temp)
                self.mask = np.logical_or(self.mask, mask_pf)

    def gaussian_1d_sum(self, params):
        """
        Compute the sum of one or more 1D Gaussian functions.

        Each Gaussian is parameterized by four consecutive values in
        ``params``. The first three values are used to define the Gaussian:

        ``[amp, x0, sigma, extra]``

        The fourth parameter is retained in the parameter array but is not
        used in the Gaussian calculation.

        Parameters
        ----------
        params : numpy.ndarray
            Current model parameter values.

        Returns
        -------
        y_model : numpy.ndarray
            1D model containing the sum of all Gaussian components.
        """

        n_gaussians = len(params) // 4

        y_model = np.zeros_like(self.xgrid, dtype=float)

        for i in range(n_gaussians):
            amp, x0, sigma, _ = params[i*4:(i+1)*4]

            y_model += amp * np.exp(
                -0.5 * ((self.xgrid - x0) / sigma)**2
            )

        return y_model

    def residuals(self, params, y, err, maxv):
        """
        Compute the residuals between the Gaussian model and the data.

        Only data points above a fraction of the maximum data value are
        included in the residual calculation.

        Parameters
        ----------
        params : numpy.ndarray
            Current model parameter values.
        y : numpy.ndarray
            Data values.
        err : numpy.ndarray
            Error or weighting values for each data point.
        maxv : float
            Maximum value of the data, used to define the fitting threshold.

        Returns
        -------
        residuals : numpy.ndarray
            Residuals between the data and Gaussian model for the selected
            data points.
        """

        # Compute the Gaussian model on the 1D grid.
        dat = self.gaussian_1d_sum(params)

        # Select data points above 10% of the maximum value.
        index, = np.where(y >= 0.1 * maxv)

        # Compute the weighted residuals.
        return (y[index] - dat[index]) * err[index]

    def fit(self):
        """
        Perform a Levenberg-Marquardt least-squares fit of the Gaussian model.

        The fitted parameter covariance matrix is estimated from the
        Jacobian using its singular-value decomposition.

        Returns
        -------
        p : scipy.optimize.OptimizeResult
            Result object returned by ``scipy.optimize.least_squares``,
            containing the fitted Gaussian parameters.
        var : numpy.ndarray
            Estimated covariance matrix of the fitted parameters.
        """

        # Replace NaN values with -inf before fitting.
        data_for_fit = np.nan_to_num(self.data, nan=-np.inf)

        # Identify finite data points.
        weights = np.isfinite(self.data)

        # Perform the least-squares Gaussian fit.
        p = least_squares(
            self.residuals,
            x0=self.param,
            args=(data_for_fit, weights, np.amax(data_for_fit)),
            method='lm'
        )

        try:
            # Compute the singular-value decomposition of the Jacobian.
            _, s, VT = svd(p.jac, full_matrices=False)

            # Define a threshold for numerically insignificant singular values.
            threshold = (
                np.finfo(float).eps
                * max(p.jac.shape)
                * s[0]
            )

            # Keep only singular values above the numerical threshold.
            s = s[s > threshold]
            VT = VT[:s.size]

            # Estimate the covariance matrix from the retained singular values.
            var = np.dot(VT.T / s**2, VT)

        except Exception:
            # Return a zero covariance matrix if the covariance calculation
            # fails.
            var = np.zeros((len(p.x), len(p.x)))

        return p, var

    def beam_fit(self, mask_pf=False):
        """
        Fit one or more 1D Gaussian functions to the input vector.

        If initial Gaussian parameters are provided, a single fit is
        performed. Otherwise, peaks are identified automatically and the
        Gaussian model is iteratively fitted. After each fit, the fitted
        model is subtracted from the data and additional peaks are searched
        for in the residual.

        Parameters
        ----------
        mask_pf : bool or numpy.ndarray, optional
            Initial mask used to exclude regions during peak finding.
            Default is ``False``.

        Returns
        -------
        fit_data : numpy.ndarray or str
            1D map containing the sum of the fitted Gaussian components.
            If the fit does not converge, a status message is returned.
        fit_param : numpy.ndarray or int
            Fitted Gaussian parameters in the order
            ``[amp, x0, sigma, extra, ...]``.
        var : numpy.ndarray or int
            Covariance matrix of the fitted parameters. Returns ``0`` if
            the fit does not converge.
        """

        # If initial parameters are provided, perform a single fit.
        if self.param is not None:
            #self.param = self.estimate_initial_guess()
            peak_number_ini = np.size(self.param) / 4
            force_fit = True

        else:

            # Find initial peaks and generate Gaussian parameter guesses.
            self.peak_finder(
                map_data=self.data,
                mask_pf=mask_pf
            )

            peak_number_ini = np.size(self.param) / 4
            peak_found = peak_number_ini
            force_fit = False

        # Iteratively fit the Gaussian components and search for additional
        # peaks in the residual.
        while peak_found > 0:

            fit_param, var = self.fit()

            # Check whether the fit converged.
            if isinstance(fit_param, str):
                msg = 'fit not converged'
                break

            else:

                # Generate the fitted 1D Gaussian model.
                fit_data = self.gaussian_1d_sum(fit_param.x)

                if force_fit is False:

                    # Subtract the fitted model to identify additional peaks.
                    res = self.data - fit_data

                    # Search for additional peaks in the residual.
                    self.peak_finder(
                        map_data=res,
                        fact=self.fact
                    )

                    # Determine the number of newly detected peaks.
                    peak_number = np.size(self.param) / 4

                    peak_found = peak_number - peak_number_ini
                    peak_number_ini = peak_number

                else:

                    # Initial parameters were provided, so stop after one fit.
                    peak_found = -1

        if isinstance(fit_param, str):
            # Fit did not converge.
            return msg, 0, 0

        else:
            # Return the fitted model, parameters, and covariance matrix.
            return fit_data, fit_param.x, var
