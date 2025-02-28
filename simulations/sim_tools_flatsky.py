"""
simulations/python/sim_tools_flatsky.py

Contains necessary function for performing flatsky analysis and 
generating correlated Gaussian realisations across different bands. 
"""

import numpy as np, sys, os, warnings

def get_lxly(map_shape, pixel_res_radians):

    """
    return lx, ly modes (kx, ky Fourier modes) for a flatsky map grid.
    
    Parameters
    ----------
    map_shape: array_like, shape (2 x 1)
        dimension on the flatskymap.        
    pixel_res_radians: float
        map pixel resolution in radians.

    Returns
    -------
    lx, ly: array, shape is map_shape.
    """

    ny, nx = map_shape
    lx, ly = np.meshgrid( np.fft.fftfreq( nx, pixel_res_radians ), np.fft.fftfreq( ny, pixel_res_radians ) )
    lx *= 2* np.pi
    ly *= 2* np.pi

    return lx, ly

def map2cl(map_shape, pixel_res_radians, flatskymap1, flatskymap2 = None, minbin = 0, maxbin = 10000, binsize = 100, mask = None, filter_2d = None):

    """
    map2cl module - get the power spectra of map/maps

    Parameters
    ----------
    map_shape: array_like, shape (2 x 1)
        dimension on the flatskymap.
    pixel_res_radians: float
        map pixel resolution in radians.
    flatskymap1: array
        flatsky map for power spectrum calculation.
        Computes the auto-power spectrum is flatskymap2 is None.
        else computes the cross-power spectrium for flatskymap1 and flatskymap2.
    flatskymap2: array
        flatsky map 2 for cross-power spectrum calculation.
        default is None
    minbin: int
        mininum scale for power spectrum calculation.
        default is \ell_min = 0
    maxbin: int
        maximum scale for power spectrum calculation.
        default is \ell_max = 15000
    binsize: int
        binning factor.
        default is \Delta_\ell = 100
    mask: array
        (Apodisation) mask for the maps for power spectrum calculations.
        Assumes that the mask is already applied to the mask.
        Default is None.
    filter_2d: array
        Filter transfer function that has information about the modes that are filtered.
        Default is None.

    Returns
    -------
    el: array.
        Multipoles (scales) over which the power spectrum is defined.
    cl: array.
        1d power spectrum.
        Azimuthally averaged (binned) equivalent of FFT( abs(map)^2 ).
    """

    ny, nx = map_shape
    lx, ly = get_lxly(map_shape, pixel_res_radians)

    if binsize == None:
        binsize = lx.ravel()[1] -lx.ravel()[0]

    if flatskymap2 is None:
        flatskymap_psd = abs( np.fft.fft2(flatskymap1) * pixel_res_radians)** 2 / (nx * ny)
    else: #cross spectra now
        assert flatskymap1.shape == flatskymap2.shape
        flatskymap_psd = np.fft.fft2(flatskymap1) * pixel_res_radians * np.conj( np.fft.fft2(flatskymap2) ) * pixel_res_radians / (nx * ny)

    if filter_2d is not None:
        flatskymap_psd = flatskymap_psd / filter_2d
        flatskymap_psd[np.isnan(flatskymap_psd) | np.isinf(flatskymap_psd)] = 0.


    rad_prf = radial_profile(flatskymap_psd, (lx,ly), binsize = binsize, minbin = minbin, maxbin = maxbin)
    el, cl = rad_prf[:,0], rad_prf[:,1]

    if mask is not None:
        fsky = np.mean(mask**2.)
        cl /= fsky

    return el, cl
    
def cl_to_cl2d(el, cl, map_shape, pixel_res_radians, left = 0., right = 0.):
    
    """
    Interpolating a 1d power spectrum (cl) defined on multipoles (el) to 2D assuming azimuthal symmetry (i.e:) isotropy.

    Parameters
    ----------
    el: array
        Multipoles over which the power spectrium is defined.
    cl: array
        1d power spectrum that needs to be interpolated on the 2D grid.
    map_shape: array_like, shape (2 x 1)
        dimension on the flatskymap.
    pixel_res_radians: float
        map pixel resolution in radians.
    left: float
        value to be used for interpolation outside of the range (lower side).
        default is zero.
    right: float
        value to be used for interpolation outside of the range (higher side).
        default is zero.

    Returns
    -------
    cl2d: array, shape is map_shape.
        interpolated power spectrum on the 2D grid.
    """
    
    lx, ly = get_lxly(map_shape, pixel_res_radians)
    el_grid = np.sqrt(lx**2. + ly**2.)
    cl2d = np.interp(el_grid.flatten(), el, cl, left = left, right = right).reshape(el_grid.shape)
    return cl2d

def make_gaussian_realisations(el, cl_dict, map_shape, pixel_res_radians):

    """
    return (correlated) Gaussian realisations of flat sky maps with an underlying power spectrum
    defined by cl_dict.

    Parameters
    ----------
    el: array
        Multipoles over which the power spectrium is defined.
    cl_dict: dictionary
        Contains the auto- and cross-power spectra of multiple bands.
        Keys are simple band incides.
        Keys must be 00, 11, 01 for 2 maps; 
        Keys must be 00, 11, 22, 01, 02, 12 for 3 maps; and so on
        where
        00 - 90x90 auto.
        11 - 150x150 auto.
        22 - 220x220 auto.
        01 - 90x150 cross.
        02 - 90x220 cross.
        12 - 150x220 cross.
    map_shape: array_like, shape (2 x 1)
        dimension on the flatskymap.        
    pixel_res_radians: float
        map pixel resolution in radians.

    Returns
    -------
    sim_maps: array
        Correlated simulated maps.
        returns N maps for N(N+1)/2 spectra.

    """

    #----------------------------------------
    #refine cl_dict to remove redundant spectra.
    cl_dict_mod = {} 
    for keyname in cl_dict:
        keyname_rev = keyname[::-1]
        if keyname_rev in cl_dict_mod: continue
        cl_dict_mod[keyname] = cl_dict[keyname]
    cl_dict = cl_dict_mod
    #----------------------------------------


    #----------------------------------------
    #solve quadratic equation to get the number of maps
    """
    For "N" maps, we will have N (N +1)/2 = total spectra which is total_spec
    (N^2 + N)/2 = total_spec
    N^2 + N - 2 * total_spec = 0
    a = 1, b = 1, c = - (2 * total_spec)
    solution is: N = ( -b + np.sqrt( b**2 - 4 * a * c) ) / (2 * a)
    """

    total_spec = len( cl_dict )
    a, b, c = 1, 1, -2 * total_spec
    total_maps = int( ( -b + np.sqrt( b**2 - 4 * a * c) ) / (2 * a) )
    assert total_maps == int(total_maps)
    #----------------------------------------

    #----------------------------------------
    #map stuff
    ny, nx = map_shape
    dx = dy = pixel_res_radians

    #norm stuff of maps
    norm = np.sqrt(1./ (dx * dy))
    #----------------------------------------

    #----------------------------------------
    #gauss reals
    gauss_reals_fft_arr = []
    for iii in range(total_maps):
        curr_gauss_reals = np.random.standard_normal([nx,ny])
        curr_gauss_reals_fft = np.fft.fft2( curr_gauss_reals )
        gauss_reals_fft_arr.append( curr_gauss_reals_fft )
    gauss_reals_fft_arr = np.asarray( gauss_reals_fft_arr )
    #----------------------------------------

    #----------------------------------------
    tmpcl = list( cl_dict.values() )[0]
    ndim_for_cl = np.ndim( tmpcl )
    assert ndim_for_cl in [1, 2]

    if ndim_for_cl == 1: #1d to 2D spec
        cl_twod_dic = {}
        for ij in cl_dict:
            i, j = ij
            curr_cl_twod = cl_to_cl2d(el, cl_dict[(i,j)], map_shape, pixel_res_radians)
            cl_twod_dic[(i,j)] = cl_twod_dic[(j,i)] = np.copy( curr_cl_twod )
    elif ndim_for_cl == 2:
        cl_twod_dic = {}
        for ij in cl_dict:
            i, j = ij
            cl_twod_dic[(i,j)] = cl_twod_dic[(j,i)] = np.copy( cl_dict[(i,j)] )
    #----------------------------------------
    
    #----------------------------------------
    #get FFT amplitudes of reals now. Appendix of https://arxiv.org/pdf/0801.4380
    map_index_combs = []
    for i in range(total_maps):
        for j in range(total_maps):
            key = [j, i]
            key_rev = [i, j]
            if key_rev in map_index_combs: continue            
            map_index_combs.append( key )

    tij_dic = {}
    for ij in map_index_combs:
        i, j = ij
        kinds = np.arange(j)
        if i == j:
            t1 = cl_twod_dic[(i,j)]
            t2 = np.zeros( (ny, nx) )
            for k in kinds:
                t2 = t2 + tij_dic[(i,k)]**2.
            tij_dic[(i,j)] = tij_dic[(j,i)]= np.sqrt( t1-t2 )
        elif i>j:
            t1 = cl_twod_dic[(i,j)]
            t2 = np.zeros( (ny, nx) )
            for k in kinds: #range(j-1):
                t2 += tij_dic[(i,k)] * tij_dic[(j,k)]
            t3 = tij_dic[(j,j)]
            tij_dic[(i,j)] = (t1-t2)/t3
    for ij in tij_dic: #remove nans
        tij_dic[ij][np.isnan(tij_dic[ij])] = 0.
    #----------------------------------------
                
    #----------------------------------------
    #FFT amplitudes times gauss reals and ifft back
    sim_maps = []
    for i in range(total_maps):
        if i == 0:
            curr_map_fft = gauss_reals_fft_arr[i] * tij_dic[(i,i)]            
        else:
            curr_map_fft = np.zeros( (ny, nx) )
            for a in range(total_maps): #loop over tij_dic
                if a>i: continue
                curr_map_fft = curr_map_fft + gauss_reals_fft_arr[a] * tij_dic[(i,a)]
        curr_map_fft = curr_map_fft * norm
        curr_map = np.fft.ifft2( curr_map_fft ).real
        curr_map = curr_map - np.mean( curr_map )
        sim_maps.append( curr_map )
    #----------------------------------------

    sim_maps = np.asarray( sim_maps )

    return sim_maps    


def apod_mask(x_grid, y_grid, mask_radius, perform_apod = True, mask_shape = 'circle', taper_radius_fac = 6.):

    """
    Interpolating a 1d power spectrum (cl) defined on multipoles (el) to 2D assuming azimuthal symmetry (i.e:) isotropy.

    Parameters
    ----------
    x_grid: array
        x grid of the map (like ra).
    y_grid: array
        y grid of the map (like dec).
    mask_radius: float
        mask radius in same units as x_grid and y_grid
    perform_apod: boolean
        Apodise the binary mask.
        Default is True.
    mask_shape: str
        circle or a square mask.
        default is circle.
        Must be on of ['circle', 'square']
    taper_radius_fac: float
        radius for apodisation.
        taper_radius = taper_radius_fac * mask_radius.
        default is 6 (FIX ME: which works well but need to be explored further).

    Returns
    -------
    mask: array, shape is x_grid.shape.
        binary or apodised mask.
    """

    assert mask_shape in ['circle', 'square']

    import scipy as sc
    import scipy.ndimage as ndimage

    mask = np.ones( y_grid.shape )

    if mask_shape == 'circle':
        radius = np.sqrt( (x_grid**2. + y_grid**2.) )
        inds_to_mask = np.where((radius<=mask_radius))
    elif mask_shape == 'sqaure':
        inds_to_mask = np.where( (abs(x_grid)>mask_radius) & (abs(y_grid)>mask_radius))

    if apod_mask:
        mask[inds_to_mask[0], inds_to_mask[1]] = 1.
    else:
        mask[inds_to_mask[0], inds_to_mask[1]] = 0.

    taper_radius = mask_radius * taper_radius_fac #
    if perform_apod:
        ##imshow(mask); colorbar(); show(); sys.exit()
        ker=np.hanning(taper_radius)
        ker2d=np.asarray( np.sqrt(np.outer(ker,ker)) )
        mask=ndimage.convolve(mask, ker2d)
        mask/=mask.max()

    return mask


def radial_profile(z, xy = None, minbin = 0., maxbin = 10., binsize = 1., get_errors = False):

    """
    get the radial profile of an image (both real and fourier space).
    Can be used to compute radial profile of stacked profiles or 2D power spectrum.

    Parameters
    ----------
    z: array
        image to get the radial profile.
    xy: array
        x and y grid. Same shape as the image z.
        Default is None.
        If None, 
        x, y = np.indices(image.shape)
    minbin: float
        minimum bin for radial profile
        default is 0.
    maxbin: float
        minimum bin for radial profile
        default is 10.
    binsize: float
        radial binning factor.
        default is 1.
    get_errors: float
        obtain scatter in each bin.
        This is not the error due to variance. Just the sample variance.
        Default is False.

    Returns
    -------
    radprf: array.
        Array with three elements cotaining
        radprf[:,0] = radial bins
        radprf[:,1] = radial binned values
        if get_errors:
        radprf[:,2] = radial bin errors.
    """

    z = np.asarray(z)
    if xy is None:
        x, y = np.indices(image.shape)
    else:
        x, y = xy

    #radius = np.hypot(X,Y) * 60.
    radius = (x**2. + y**2.) ** 0.5
    
    binarr=np.arange(minbin, maxbin, binsize)
    radprf=np.zeros((len(binarr),3))

    hit_count=[]

    for b,bin in enumerate(binarr):
        ind=np.where((radius>=bin) & (radius<bin+binsize))
        radprf[b,0]=(bin+binsize/2.)
        hits = len(np.where(abs(z[ind])>0.)[0])

        if hits>0:
            radprf[b,1]=np.sum(z[ind])/hits
            radprf[b,2]=np.std(z[ind])
        hit_count.append(hits)

    hit_count=np.asarray(hit_count)
    std_mean=np.sum(radprf[:,2]*hit_count)/np.sum(hit_count)
    if get_errors:
        errval=std_mean/(hit_count)**0.5
        radprf[:,2]=errval

    return radprf

################################################################################################################
################################################################################################################
################################################################################################################
