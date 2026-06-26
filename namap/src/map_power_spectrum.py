import numpy as np
import scipy.constants as cst
from astropy import units as u
from IPython import embed
from scipy.optimize import curve_fit
from scipy import interpolate
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from astropy.cosmology import Planck18 as cosmo

class angular_power_spectrum:

    """
    Class to measure an angular power spectrum out of a 2D map

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
        self.maps = maps
        self.map2 = map2
        self.resx = res
        self.delta_k_over_k = delta_k_over_k
        self.ny, self.nx = maps[0].shape

    # ------------------------------------------------------------
    # Make k bins
    # ------------------------------------------------------------
    def make_bintab(self, kmin, kmax, dk_min, dkk=None):
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
        if(dkk is None):
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
    def set_k_infos_2d(self):
        """
        Compute all wavenumber related quantities based on the resolution and shape of the map to be analysed.
        The function computes the wavenumber k bins, the center of the bins, the Nyquist wavenumber, and the 2D k map. 

        Parameters
        ----------

        Returns
        -------        
        """

        ny, nx = self.ny, self.nx
        res = self.resx

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
        nx, ny, res = self.nx, self.ny, self.resx
        Ys, Xs = np.unravel_index(np.arange(nx*ny),(ny,nx))
        N = np.zeros((ny, nx))
        M = np.zeros((ny, nx))
        for  xs, ys, in zip(Xs, Ys):
            N[ys,xs] = int(xs)
            M[ys,xs] = int(ys)
        w_sup = np.where(M>ny/2)
        M[w_sup] = ny - M[w_sup]
        w_sup = np.where(N>nx/2)
        N[w_sup] = nx - N[w_sup]
        Kx = M/nx/res
        Ky = N/ny/res
        k_map = (Kx**2 + Ky**2)**(1/2)
        #l_map = k_map * 2 * np.pi
        
        return k_map

        '''
        ny, nx = self.ny, self.nx
        res = self.res

        # FFT frequencies in cycles per radian
        ky = np.fft.fftfreq(ny, d=res)
        kx = np.fft.fftfreq(nx, d=res)

        KX, KY = np.meshgrid(kx, ky)

        k_map = np.sqrt(KX**2 + KY**2)
        return k_map
        '''


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
        res = self.resx

        norm = (res**2) / (nx * ny)

        self.set_k_infos_2d()

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

        return pk_list, self.k_bin_tab, self.k_out

class threedim_power_spectrum_for_comoving_cubes(angular_power_spectrum):
    """
    Class to measure an spherically-averaged power spectrum out of a 3D cube

    Parameters
    ----------

    Returns
    -------
    """

    def __init__(self, cube, resz, resx, resy, delta_k_over_k_perp=0, delta_k_over_k_par=0, correct_spectral_window=False):

        """
        Create an instance of the class

        Parameters
        ----------
        cube: np.ndarray
            the cube to measure the power spectrum from in Jy/sr
        resz: float
            the resolution of the cube in the radial direction, in Mpc
        resx: float
            the resoltion of the cube in the x transverse direction, in Mpc
        resy: float
            the resoltion of the cube in the y transverse direction, in Mpc
        delta_k_over_k_perp: float
            the log binning of the k space along the transverse direction. If set to 0, the binning is linear with k_min as bin size.
        delta_k_over_k_par: float
            the log binning of the k space along the radial direction. If set to 0, the binning is linear with k_min as bin size.
        
        Returns
        -------
        """

        self.cube = cube
        self.nz, self.ny, self.nx = cube.shape
        self.resz = resz
        self.resx = resx
        self.resy = resy
        self.cube2 = None
        self.delta_k_over_k_perp = delta_k_over_k_perp
        self.delta_k_over_k_par = delta_k_over_k_par
        self.correct_spectral_window = correct_spectral_window

    # ------------------------------------------------------------
    # Compute the k-binning and maps
    # ------------------------------------------------------------
    def set_k_infos(self):
        """
        Compute all wavenumber related quantities based on the resolution and shape of the cube to be analysed.
        The function computes the wavenumber k bins, the center of the bins, the Nyquist wavenumber, and the 3D k map. 

        Parameters
        ----------

        Returns
        -------        
        """


        """
        k_transv_m1 = np.sqrt(v_freq[:,None]**2 + w_freq[None,:]**2)
        Nmodes1d_m1, _ = np.histogram( k_transv_m1.flatten(), bins=k_edges)
        print('Nmodes of method 1:',Nmodes1d_m1)
        import matplotlib.pyplot as plt
        plt.loglog(self.k_out, Nmodes1d_m1, 'k')
        k_transv_m2 = self.give_map_spatial_freq() * 2*np.pi
        Nmodes1d_m2, _ = np.histogram( k_transv_m2.flatten(), bins=k_edges)
        print('Nmodes of method 2:'Nmodes1d_m2)
        plt.loglog(self.k_out, Nmodes1d_m2, 'r')
        plt.show()
        """

        w_freq = 2*np.pi*np.fft.fftfreq(self.nx, d=self.resx)
        v_freq = 2*np.pi*np.fft.fftfreq(self.ny, d=self.resy)
        u_freq = 2*np.pi*np.fft.fftfreq(self.nz, d=self.resz)
        k_transv =  2*np.pi*self.give_map_spatial_freq() 
        #a = np.sqrt(v_freq[:,None]**2 + w_freq[None,:]**2)
        
        k_transv_3d = np.zeros((self.nz, self.ny, self.nx))
        k_transv_3d[:,:,:] = k_transv[None,:,:]      
        k_z = np.sqrt( u_freq**2 )     
        k_z_3d = np.zeros((self.nz, self.ny, self.nx))
        k_z_3d[:,:,:] = k_z[:,None, None]  
        k_sphere = np.sqrt(k_z_3d**2+k_transv_3d**2) #np.sqrt( u_freq[:,None,None]**2+ v_freq[None,:,None]**2 + w_freq[None,None,:]**2)

        self.k_map_3d = k_sphere

        kmin_perp, kmax_perp = k_transv[k_transv>0].min(), k_transv.max()

        k_z_positive = k_z[k_z > 0]
        if k_z_positive.size > 0:
            kmin_parr, kmax_parr = k_z_positive.min(), k_z.max()
        else:
            kmin_parr, kmax_parr = kmin_perp, kmax_perp  # 2D fallback

        #kmin_parr, kmax_parr = k_z[k_z>0].min(), k_z.max()
        kmin, kmax = np.min((kmin_perp, kmin_parr)), np.max((kmax_perp, kmax_parr))
        #dkk = 0 #np.max((self.delta_k_over_k_perp,self.delta_k_over_k_par))

        k_bins_perp = self.make_bintab(kmin_perp, kmax_perp, kmin_perp, self.delta_k_over_k_perp)
        k_bins_parr = self.make_bintab(kmin_parr, kmax_parr, kmin_parr, self.delta_k_over_k_par)
        k_edges =  self.make_bintab(   kmin,      kmax,      kmin, self.delta_k_over_k_par)


        # Bin centers
        self.k_out = 0.5 * (k_edges[1:] + k_edges[:-1])
        k_out_perp = 0.5 * (k_bins_perp[1:] + k_bins_perp[:-1])
        k_out_parr = 0.5 * (k_bins_parr[1:] + k_bins_parr[:-1])
        self.k_sphere_bins = k_edges
        

        self.k_sphere = k_sphere
        self.k_transv_3d = k_transv_3d
        self.k_z_3d = k_z_3d
        self.k_out_perp = k_out_perp
        self.k_out_parr = k_out_parr
        self.k_bins_perp = k_bins_perp
        self.k_bins_parr = k_bins_parr

        self.kperp_flat = self.k_transv_3d.flatten()
        self.kpar_flat = self.k_z_3d.flatten()
        self.ksphere_flat = self.k_sphere.flatten()


        #--------------------------------------------------
        """
        """
        #-----------------------------n --------------------------

    # ------------------------------------------------------------
    # Main P(k) estimator
    # ------------------------------------------------------------
    def p3(self):
            
        """
        Estimates the spherically-averaged power spectrum in Jy**2.Mpc3 of an 3D cube in Jy/sr and pixel size in Mpc.

        Parameters
        ----------

        Returns
        -------
        P2D_avg: 2d array
            the power amplitudes in the k_perp - k_par space
        Nmodes: 2d array
            the number of modes associated to 2d array
        k_bins_parr: 1d array
            the k bins in the radial direction
        k_bins_perp: 1d array
            the k bins the transverse direction
        P1d_sum: 1d array
            the spherically-averaged power spectrum
        k_sphere_bins: 1d array
            the spherical k bins
        Nmodes1d: 1d array
          the number of modes associated to P1d_sum
        k_out: 1d array
            the address of the spherical k bins
        k_map_3d: 3d array
            the 3D cube of spherical values of k
        k_z_3d: 3d array
            the 3D cube of radial values of k

        """

        self.set_k_infos()

        #cube_Mpc = self.angspec_cube_to_comoving_cube()

        norm = (self.resx*self.resy*self.resz) / (self.nz*self.ny*self.nx)
        # Create a mask (1 where valid, 0 where NaN)

        
        mask = np.isfinite(self.cube.astype(float))
        # Fill NaNs with 0 (or the mean, depending on your normalization)
        map_filled = np.nan_to_num(self.cube, nan=0.0)
        ft = np.fft.fftn(map_filled) 
        ft2 = ft
        '''
        #if self.cube2 is None:
        else:
            map_filled_2 = np.nan_to_num(self.cube2, nan=0.0)
            ft2 = np.fft.fftn(map_filled_2)
        '''
        
        p2map = (ft * np.conj(ft2)).real * norm

        # 1. Flatten everything
        pow_flat = p2map.flatten()
        valid_flat = np.ones_like(pow_flat).astype(bool)
        
        # 5b. Compute number of modes per bin
        Nmodes, _, _ = np.histogram2d(
            self.kpar_flat[valid_flat], self.kperp_flat[valid_flat],
            bins=[self.k_bins_parr, self.k_bins_perp] )
        
        # 5a. Compute total power per (kpar, kperp) bin
        P2D_sum, kpar_edges_out, kperp_edges_out = np.histogram2d(
            self.kpar_flat[valid_flat], self.kperp_flat[valid_flat],
            bins=[self.k_bins_parr, self.k_bins_perp],
            weights=pow_flat[valid_flat] )
        
        with np.errstate(divide='ignore', invalid='ignore'):
            P2D_avg = np.where(Nmodes > 0, P2D_sum / Nmodes, np.nan)
        # 5b. Compute number of modes per bin
        Nmodes1d, _ = np.histogram( self.ksphere_flat[valid_flat], bins=self.k_sphere_bins)
        # 5a. Compute total power per (kpar, kperp) bin
        P1d_sum, _ = np.histogram(self.ksphere_flat[valid_flat], bins=self.k_sphere_bins, weights=pow_flat[valid_flat])
        with np.errstate(divide='ignore', invalid='ignore'): P1d_sum = np.where(Nmodes1d > 0, P1d_sum / Nmodes1d, np.nan)


        #----------------------------------------------------
        '''
        kpar = self.k_z_3d  # shape (nz, ny, nx)
        delta_r = self.resz   # since your cube spacing is resz
        W2 = 0.5*(1 + np.cos(kpar * delta_r))
        eps = 1e-10
        valid = W2 > eps
        valid_flat = W2.flatten() > eps
        valid_flat = np.ones_like(p2map).flatten().astype(bool)
        W2_modes = W2.flatten()
        W2_sum, _ = np.histogram( self.ksphere_flat[valid_flat],   bins=self.k_sphere_bins,  weights=W2_modes[valid_flat])
        self.W2_sum = W2_sum
        '''
        #----------------------------------------------------

        return P2D_avg, Nmodes, self.k_bins_parr, self.k_bins_perp, P1d_sum, self.k_sphere_bins, Nmodes1d, self.k_out, self.k_map_3d, self.k_z_3d

class threedim_power_spectrum_for_angular_cubes(threedim_power_spectrum_for_comoving_cubes):
    """
    Class to measure an angular power spectrum out of a 2D map

    Parameters
    ----------

    Returns
    -------
    """

    def __init__(self, cube, res, nu, dnu, nu0, delta_k_over_k_perp=0,delta_k_over_k_par=0, correct_spectral_window=False):

        """
        Create an instance of the class.

        Parameters
        ----------

        cube : 3D array
            The cube to analysed, in Jy/sr
        res : float
            pixel size of the map in rad
        nu_min: float
            the lower E.M frequency channel in GHz 
        nu0: float
            the rest frame frequency in GHz
        dnu: float
            the linear frequency bin width.
        dlognu: 1d array
            the log frequency bin width
        delta_k_over_k_perp: float
            the log binning of the k space along the transverse direction. If set to 0, the binning is linear with k_min as bin size.
        delta_k_over_k_par: float
            the log binning of the k space along the radial direction. If set to 0, the binning is linear with k_min as bin size.

        Returns
        -------
        """

        self.cube = cube
        self.res = res #rad
        self.nu0 = nu0 #GHz
        self.nu = nu
        self.dnu = dnu
        self.delta_k_over_k_perp = delta_k_over_k_perp
        self.delta_k_over_k_par = delta_k_over_k_par
        self.nz, self.ny, self.nx = cube.shape
        self.correct_spectral_window=correct_spectral_window

    # ------------------------------------------------------------
    # convert angles to Mpc
    # ------------------------------------------------------------
    def angspec_cube_to_comoving_cube(self):

        """
        Convert angular-frequency axes to comoving axes

        Parameters
        ----------

        Returns
        -------
        """

        z_list = self.nu0 / self.nu -1
        Dc_center = cosmo.comoving_distance(z_list).value
        chi = cosmo.comoving_distance(z_list).value
        self.Vvoxels = self.res**2 * chi**2 * (cst.c*1e-3) / cosmo.H(z_list).value * (1+z_list)**2 * self.dnu / self.nu0
        self.Vvoxel = self.Vvoxels.mean()
        Delta_Dc = cst.c*1e-3*(1+z_list) / cosmo.H(z_list).value * self.dnu / self.nu
        res_pix = Dc_center * self.res
        self.resx = res_pix.mean()
        self.resy = res_pix.mean()
        self.resz = Delta_Dc.mean() 

        return 0
    
    # ------------------------------------------------------------
    # Main P(k) estimator
    # ------------------------------------------------------------
    def ap3(self,mask_correction=False):
            
        """
        Estimates the spherically-averaged power spectrum in Jy**2.Mpc3 of an angular-spectral cube in Jy/sr. 

        Parameters
        ----------

        Returns
        -------      
        """

        self.angspec_cube_to_comoving_cube()

        self.set_k_infos()

        return self.p3()
