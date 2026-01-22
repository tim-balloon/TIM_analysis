import numpy as np
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from astropy.io import fits
from astropy.coordinates import SkyCoord, AltAz, EarthLocation
from astropy.time import Time
import astropy.units as u
from astropy.visualization import ZScaleInterval
zscale = ZScaleInterval()
import src.beam as bm
import src.pointing as pt  
from scipy.ndimage import shift
from mpl_toolkits.axes_grid1 import make_axes_locatable
from IPython import embed
from scan_fcts import *

def gen_tod(wcs, Map, hdr, ybins, xbins, pointing_paths, T=None):

    """
    Generate the sky amplitude TODs for a set of detectors from the same (electromagnetic) frequency channel, using a simulated sky map.

    Parameters
    ----------
    wcs: astropy.wcs.wcs.WCS
        The wcs used to generate the TODs
    Map: 2d array
        the 2d angular map of the sky in a frequency channel
    ybins: array
        edges of the pixels
    xbins: array
        edges of the pixels
    pointing_paths: list of 2d array
        coordinates of the sky scan path of each pixel, for pixels seeing the same frequency band
    Returns
    -------
    hist: 2d array
        the reconstructed sky map given the pointing paths 
    norm: 2d array
        the hitmap
    samples: list
        list of the amplitude timestreams of each detectors
    positions_x: list
        list of RA coordinates timestreams of each detectors
    positions_y: list
        list of DEC coordinates timestreams of each detectors 
    """ 
    
    positions_x = np.zeros((len(pointing_paths), len(pointing_paths[0][:,0])))
    positions_y = np.zeros((len(pointing_paths), len(pointing_paths[0][:,0])))
    samples = np.zeros((len(pointing_paths), len(pointing_paths[0][:,0])))
    
    #We sample the map for each detector, following its path on the sky. 
    for detector, path in enumerate(pointing_paths):
        
        #Convert the path on the sky from WCS to pixel coordinates.
        y_pixel_coords, x_pixel_coords = wcs.world_to_pixel_values(pointing_paths[detector][:,0], pointing_paths[detector][:,1])    
        # Round the positions and convert to integer indices
        x_pixel_coords_rounded = np.round(x_pixel_coords).astype(int)
        y_pixel_coords_rounded = np.round(y_pixel_coords).astype(int)
        # Create a mask for positions within bounds of the map
        valid_mask = (
            (x_pixel_coords_rounded >= 0) & (x_pixel_coords_rounded < hdr['NAXIS1'] - 1) &  # x within bounds
            (y_pixel_coords_rounded >= 0) & (y_pixel_coords_rounded < hdr['NAXIS2'] - 1) )  # y within bounds
        # Initialize the output array with zeros
        values = np.zeros_like(x_pixel_coords_rounded, dtype=float)
        # Assign values from the map for valid positions
        values[valid_mask] = Map[x_pixel_coords_rounded[valid_mask], y_pixel_coords_rounded[valid_mask]]
        samples[detector,:] = np.asarray(values.astype(float))
        positions_x[detector,:] = x_pixel_coords
        positions_y[detector,:] = y_pixel_coords

        norm, edges = np.histogramdd(sample=(x_pixel_coords.ravel(), y_pixel_coords.ravel()), bins=(xbins,ybins),  )
        hist, edges = np.histogramdd(sample=(x_pixel_coords.ravel(), y_pixel_coords.ravel()), bins=(xbins,ybins), weights=samples[detector].ravel())
        #plt.figure()
        #plt.imshow(hist/norm, origin = 'lower')
        #plt.title(f'detector={detector}')
    #plt.show()

    #Compute the number of times each sky pixel is hit by the detectors.
    norm, edges = np.histogramdd(sample=(positions_x.ravel(), positions_y.ravel()), bins=(xbins,ybins),  )
    hist, edges = np.histogramdd(sample=(positions_x.ravel(), positions_y.ravel()), bins=(xbins,ybins), weights=samples.ravel())

    #Create the observed map: 
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", RuntimeWarning)  # Catch all runtime warnings
        hist /= norm  # Perform the division
        
    return hist, norm, samples, positions_x, positions_y

#-----------------------------------------------------------------
dtype_map = {
    '16': np.float16, 'float16': np.float16, 'half': np.float16,
    '32': np.float32, 'float32': np.float32, 'single': np.float32,
    '64': np.float64, 'float64': np.float64, 'double': np.float64
}

int_map = {
    '16': np.int16, 'float16': np.int16, 'half': np.int16,
    '32': np.int32, 'float32': np.int32, 'single': np.int32,
    '64': np.int64, 'float64': np.int64, 'double': np.int64
}
#-----------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------------
dec = -27.80833 *u.deg
ra = 0 *u.deg
lat = -77.83 *u.deg
lon = 0 * u.deg      # you implicitly assumed this
LST_astro = 0 * u.hourangle
#lon = 0 * u.deg      # you implicitly assumed this
#LST = 0 * u.hourangle
#---------------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------------
file = '/home/mvancuyck/Desktop/TIM_analysis/timestream_maker/fits_and_hdf5/cube_1source_with_1xbigger_sigma_PSF.fits'
pix = 100
map_value = fits.getdata(file)[0][pix:-pix, pix:-pix]
hdr = fits.getheader(file)
hdr['CRPIX1'] -= pix
hdr['CRPIX2'] -= pix
hdr['NAXIS1'], hdr['NAXIS2'] = map_value.shape
hdr['CRVAL1'] = ra.value
hdr['CRVAL2'] = dec.value
xbins = np.arange(-0.5, hdr['NAXIS1']+0.5, 1)
ybins = np.arange(-0.5, hdr['NAXIS2']+0.5, 1)
wcs = WCS(hdr, naxis=2) 
zscale = ZScaleInterval()
vmin, vmax = zscale.get_limits(map_value)
res = (hdr['CDELT1'] * u.Unit(hdr['CUNIT1'])).to(u.deg).value
#---------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------
#Load the scan duration and generate the time coordinates with the desired acquisition rate. 
T_duration = 1 #h
dt = 1/100/3600*np.pi/3.14 #Make the timestep non rational to avoid some stripes in the hitmap. 
spf = int(1/np.round(dt*3600,3)) #sample per frame defined here as the acquisition rate in Hz. 
T = np.arange(0,T_duration,dt) * 3600 #s
#local sideral time
LST = np.arange(-T_duration/2,T_duration/2,dt) #hours
az, alt, flag = genLocalPath(az_size=0.05, alt_size=0.1, alt_step=5/3600, acc=0.05, scan_v=0.1, dt=np.round(dt*3600,3))
scan_path, scan_flag = genScanPath(T, dt, alt, az, flag)

LST_mean = 0
alt_ref = elevationAngle(dec.value,lat.value,LST_mean)
azi_ref = azimuthAngle(dec.value,lat.value,LST_mean)
print('np.degrees(azi_ref), np.degrees(alt_ref):', np.degrees(azi_ref), np.degrees(alt_ref))

#Generate the pointing on the sky for the center of the arrays
scan_path_sky, azel = genPointingPath(T, scan_path, LST, lat.value, dec.value, azel=True) 

###--------------------------------
#alt_offsets = EL * u.deg
#az_offsets  = XEL / np.cos(np.radians(alt)) 
#alt_scan = alt + alt_offsets
#az_scan  = azi  + az_offsets

EL = np.asarray((-7, 0, 7)) * 15 / 3600  # deg
XEL = np.zeros(len(EL))                 # deg
pixel_offsets = pixels_rotations(EL, XEL, 0)
pointing_path = [genPointingPath(T, scan_path, LST, lat.value, dec.value, offsets) for offsets in pixel_offsets]
pointing_path_ref = genPointingPath(T, scan_path, LST, lat.value, dec.value, np.array([0,0]))

#hist, norm, samples, positions_x, positions_y = gen_tod(wcs, map_value, hdr, ybins, xbins, pointing_path, T=None)

x_el = azi_ref * np.cos(alt_ref) - np.radians(XEL)
azi_point = x_el / np.cos(alt_ref)
alt_point = np.radians(EL) + alt_ref

dec_point = declinationAngle(np.degrees(azi_point), np.degrees(alt_point), lat.value)
ha_point  = hourAngle(       np.degrees(azi_point), np.degrees(alt_point), lat.value)
ra_point = (LST_mean*np.pi/12-ha_point)
ra_unwrapped = np.unwrap(ra_point) #( ra + np.pi) % (2 * np.pi) - np.pi
###--------------------------------

#----------------------------------------
#Plot a scan route
if(False):
    BS = 10; plt.rc('font', size=BS); plt.rc('axes', titlesize=BS); plt.rc('axes', labelsize=BS)
    fig, axs = plt.subplots(2,2,figsize=(7,7), dpi=160,)# sharey=True, sharex=True)

    axradec, ax, axr, axc = axs[0,0], axs[1,1], axs[0,1], axs[1,0]
    #---
    axc.scatter(scan_path_sky[:,0], scan_path_sky[:,1], s=0.1,c='r')
    axc.plot(ra.value, dec.value, 'ok')
    axc.plot(ra_point,np.degrees(dec_point), 'ob')

    axc.set_xlabel('RA [deg]')
    axc.set_ylabel('Dec [deg]')
    axc.set_aspect('auto')
    #---
    axradec.plot(az-az.max()/2,alt-alt.max()/2,'k', )
    axradec.set_aspect('auto')
    axradec.set_xlabel('RA [deg]')
    axradec.set_ylabel('Dec [deg]')
    #---
    hitmap, xedges, yedges = np.histogram2d(scan_path_sky[:,0], scan_path_sky[:,1], bins=int(2/res))
    im = ax.imshow(hitmap.T, origin='lower', cmap='binary',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    cbar = fig.colorbar(im, ax=ax, label='1 detector counts')
    ax.set_xlabel('RA [deg]')
    ax.set_ylabel('Dec [deg]')
    ax.set_aspect('auto')
    #---
    az_unwrapped = (np.radians(azel[:,0]) + np.pi) % (2 * np.pi) - np.pi
    axr.plot(np.degrees(az_unwrapped),azel[:,1],'b')
    az_unwrapped = (np.radians(azi_ref) + np.pi) % (2 * np.pi) - np.pi
    axr.plot(np.degrees(azi_ref), np.degrees(alt_ref), 'ok')
    axr.plot(np.degrees(azi_point),np.degrees(alt_point),'og')
    axr.set_xlabel('Az [deg]')
    axr.set_ylabel('El [deg]')
    #-----
    if(True): 
        coords = SkyCoord(alt=np.degrees(az_unwrapped)*u.deg, az=azel[:,1]*u.deg, frame='altaz')
        separations = coords[:-1].separation(coords[1:])
        total_length = np.sum(separations)
        title = f'{total_length:.1f} scanned.'
    else: title = ''
    fig.suptitle(title)
    patchs = []
    fig.tight_layout()
#----------------------------------------

#---------------------------------------------------------------------------------------------------------------------
# 1. Sky → pixels
x_pix, y_pix = wcs.wcs_world2pix(np.degrees(ra_point), np.degrees(dec_point), 0)
dx = x_pix - hdr['CRPIX1']
dy = y_pix - hdr['CRPIX2']
print('dx in pixel: ', dx)
print('dy in pixel: ', dy)
#---------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------
map_values = []

#Convert the path on the sky from WCS to pixel coordinates.
y_pixel_coords, x_pixel_coords = wcs.world_to_pixel_values(pointing_path_ref[:,0], pointing_path_ref[:,1])  

for p, dyi, dxi in zip(pointing_path, dy, dx):

    _, _, samples, _, _ = gen_tod(wcs, map_value, hdr, ybins, xbins, (p,), T=None)
    if(False): plt.figure(); plt.imshow(hist, origin='lower', vmin=vmin, vmax=vmax)

    norm, edges = np.histogramdd(sample=(x_pixel_coords.ravel(), y_pixel_coords.ravel()), bins=(xbins,ybins),  )
    hist, edges = np.histogramdd(sample=(x_pixel_coords.ravel(), y_pixel_coords.ravel()), bins=(xbins,ybins), weights=samples.ravel())
    if(False): plt.figure(); plt.imshow(hist/norm, origin='lower',vmin=vmin, vmax=vmax)
    map_values.append(hist/norm)
#------------------------------------------------------------------------------------------------------------

x_peaks = []
y_peaks = []

for shifted in map_values:

    beam_value = bm.beam(shifted, )#param = self.beamparam
    beam_map = beam_value.beam_fit()
    param = beam_map[1]

    if isinstance(beam_map[0], str): print(beam_map[0])
    else: 
        if(False):
            fig, ax = plt.subplots(1,2,figsize=(9, 3), )
            ax[0].set_title('Data')
            im = ax[0].imshow(shifted, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
            divider = make_axes_locatable(ax[0])
            cax = divider.append_axes("right", size="5%", pad=0.05)  # size can be a percentage or absolute
            fig.colorbar(im, cax=cax, label='Jy/sr')         
            ax[0].set_xlabel('X pixel')
            ax[0].set_ylabel('Y pixel')           

            ax[1].contour(beam_map[0], levels=2, colors='red')
            ax[1].imshow(beam_map[0], origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
            divider = make_axes_locatable(ax[1])
            cax = divider.append_axes("right", size="5%", pad=0.05)  # size can be a percentage or absolute
            fig.colorbar(im, cax=cax, label='Jy/sr')         
            ax[1].set_title('2D Gaussian Fit')
            ax[1].set_xlabel('X pixel')
            ax[1].set_ylabel('Y pixel')
            fig.tight_layout()

        params = beam_map[1]
        cov = beam_map[2]
        uncertainties = np.sqrt(np.diag(cov))
        print(f'xo={params[2]:.2f} pm {uncertainties[2]:.2f} | yo={params[1]:.2f} pm {uncertainties[1]:.2f}' )
        #w = np.where(shifted == shifted.max());   print(f'The peak is at {w[0][0]} - {w[1][0]}' ) 
        x_peaks.append(params[1]); y_peaks.append(params[2])

x_fits = np.asarray(x_peaks)
y_fits = np.asarray(y_peaks)
ra_deg, dec_deg = wcs.wcs_pix2world(x_fits, y_fits, 1)
print('ra: ',ra_deg)

EL_dets = elevationAngle(dec_deg,lat.value,LST_mean) 
AZ_dets = azimuthAngle(dec_deg,lat.value,LST_mean) 

daz = np.degrees(AZ_dets) - np.degrees(azi_ref)
xel = daz * np.cos(alt_ref)
delv = np.degrees(EL_dets) - np.degrees(alt_ref)

fig,axs = plt.subplots(1,2, figsize=(12,4))
#---
axs[0].plot(np.degrees(azi_point),np.degrees(alt_point),'ok')
axs[0].plot(np.degrees(azi_ref), np.degrees(alt_ref), '.r')
axs[0].plot(np.degrees(AZ_dets), np.degrees(EL_dets), 'xg')    
axs[0].set_xlim(np.degrees(azi_ref) -10*15/3600, np.degrees(azi_ref) +10*15/3600)
axs[0].set_xlabel('Az [deg]')
axs[0].set_ylabel('El [deg]')

axs[1].plot(XEL, EL, 'ok')
axs[1].plot(xel, delv, '.r')              

axs[1].set_xlim(0-10*15/3600, 0+10*15/3600)
axs[1].set_xlabel('$\\rm \\Delta$x-EL [deg]')
axs[1].set_ylabel('$\\rm \\Delta$El [deg]')
plt.show()

embed()