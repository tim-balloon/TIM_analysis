import sys, os, gc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from astropy.io import fits
from astropy.coordinates import SkyCoord, AltAz, EarthLocation
from astropy.time import Time
import astropy.units as u
from astropy.visualization import ZScaleInterval
zscale = ZScaleInterval()
from scipy.ndimage import shift
from mpl_toolkits.axes_grid1 import make_axes_locatable
from IPython import embed
import src.beam as bm
import src.pointing as pt  
from astropy import wcs
import src.quaternion as quat

DT = np.float64
IT = np.int64

#---------------------------------------------------------------------
EL = np.asarray((-7, 0, 7)) * 15 / 3600  # deg
XEL = np.zeros(len(EL))                 # deg
scan_path = np.vstack((XEL, EL)).T      # (dAz, dEl) in deg
dec = -27.80833 * u.deg
ra = 0 * u.deg
lat = -77.83 * u.deg
lon = 0 * u.deg      # you implicitly assumed this
LST = 0 * u.hourangle

location = EarthLocation(lat=lat, lon=lon)
date = "2025-01-01T00:00:00"; t0 = Time(date, scale="utc", location = location)
lst0 = t0.sidereal_time("apparent")
dt = (lst0 / (24 * u.hourangle)) * u.sday
t = t0 - dt

sky_center = SkyCoord( ra=ra, dec=dec, frame="icrs")
altaz_frame = AltAz(obstime=t, location=location)
center_altaz = sky_center.transform_to(altaz_frame)

alt_offsets = scan_path[:, 1] * u.deg
az_offsets  = scan_path[:, 0] * u.deg / np.cos(np.radians(alt_offsets)) 

alt_scan = center_altaz.alt + alt_offsets
az_scan  = center_altaz.az  + az_offsets

#---------------------------------------------------------------------------------------------------------------------
file = '../../datasets/cube_1source_with_1xbigger_sigma_PSF.fits'
pix = 100
map_value = fits.getdata(file)[0][pix:-pix, pix:-pix]
hdr = fits.getheader(file)
hdr['CRPIX1'] -= pix
hdr['CRPIX2'] -= pix
hdr['NAXIS1'], hdr['NAXIS2'] = map_value.shape
hdr['CRVAL1'] = ra.value
hdr['CRVAL2'] = dec.value
wcs = WCS(hdr, naxis=2) 
zscale = ZScaleInterval()
vmin, vmax = zscale.get_limits(map_value)
#---------------------------------------------------------------------------------------------------------------------

# 1. Sky → pixels
scan_altaz = SkyCoord( az=az_scan, alt=alt_scan, frame=altaz_frame)
scan_icrs = scan_altaz.transform_to("icrs")
ra_point  = scan_icrs.ra.wrap_at(180 * u.deg)
dec_point = scan_icrs.dec
x_pix, y_pix = wcs.wcs_world2pix(ra_point.to(u.deg).value, dec_point.to(u.deg).value, 0)
dx = x_pix - hdr['CRPIX1']
dy = y_pix - hdr['CRPIX2']
print('dx in pixel: ', dx)
print('dy in pixel: ', dy)

if(False):
    # 2. Pixels → sky
    ra_back, dec_back = wcs.wcs_pix2world(x_pix, y_pix, 0)

    # 3. Pixels → sky → pixels (optional, should match x_pix, y_pix closely)
    x_check, y_check = wcs.wcs_world2pix(ra_back, dec_back, 0)

    print("Original pixels:", x_pix, y_pix)
    print("Roundtrip pixels:", x_check, y_check)
    print("Original RA/Dec (deg):", ra_point.to(u.deg).value, dec_point.to(u.deg).value)
    print("Roundtrip RA/Dec:", ra_back, dec_back)

    ra_orig = SkyCoord(ra=ra_point, dec=dec_point)
    ra_back = SkyCoord(ra=ra_back*u.deg, dec=dec_back*u.deg)
    print(ra_back.ra.wrap_at(180*u.deg))  # RA wrapped to [-180,180]

    fig, axs = plt.subplots(1,2)
    axs[0].plot(x_pix, y_pix, 'ok')
    axs[0].plot(x_check, y_check, '.r')
    axs[1].plot(ra_point.to(u.deg).value, dec_point.to(u.deg).value, 'ok')
    axs[1].plot(ra_back.ra.wrap_at(180*u.deg), dec_back, '.r')
    plt.show()

if(False):
    beam_value = bm.beam(map_value, )#param = self.beamparam
    beam_map = beam_value.beam_fit()
    param = beam_map[1]

    if isinstance(beam_map[0], str): print(beam_map[0])
    else: 
        if(True):
            fig, ax = plt.subplots(1,2,figsize=(9, 3), )
            ax[0].set_title('Data')
            im = ax[0].imshow(map_value, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
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

plt.show()

x_peaks = []
y_peaks = []

for dxi, dyi in zip(dx, dy):

    #shifted = np.roll(map_value, shift=(i0, j0), axis=(0, 1),)
    shifted = shift(map_value, shift=(dyi, dxi), order=3, mode='constant', cval=0.0)

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
        print(f'xo={params[1]:.2f} pm {uncertainties[1]:.2f} | yo={params[2]:.2f} pm {uncertainties[2]:.2f}' )
        #w = np.where(shifted == shifted.max());   print(f'The peak is at {w[0][0]} - {w[1][0]}' ) 
        x_peaks.append(params[1]); y_peaks.append(params[2])

x_fits = np.asarray(x_peaks)
y_fits = np.asarray(y_peaks)
ra_deg, dec_deg = wcs.wcs_pix2world(x_fits, y_fits, 0)

#-------------------------------------------------------------------
#CASE A
sky_icrs = SkyCoord( ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
sky_altaz = sky_icrs.transform_to(altaz_frame)
az = sky_altaz.az.to(u.deg).value
el = sky_altaz.alt.to(u.deg).value
daz = az - center_altaz.az.to(u.deg).value
delv = el - center_altaz.alt.to(u.deg).value
xel = daz * np.cos(np.radians(center_altaz.alt.to(u.deg).value))
#-------------------------------------------------------------------

#-----------------------------------------------------------------------------
#CASE B
corr = pt.apply_offset('RA and DEC', (wcs.wcs.crval[0],), (wcs.wcs.crval[1],), 'AZ and EL',
                       xsc_offset=np.array([0.,0.]), DT=DT, IT=IT, lst = LST.value, lat = lat.value, )
azi_ref, alt_ref = corr.correction()
conv = pt.apply_offset('RA and DEC', np.zeros_like(ra_deg), dec_deg, 'AZ and EL',
                        xsc_offset=np.array([0.,0.]), DT=DT, IT=IT, lst = LST.value, lat = lat.value, )
AZ_dets_mine, EL_dets_mine = conv.correction()
daz = AZ_dets_mine - azi_ref
#daz = (daz + 180) % 360 - 180  # unwrap safely
xel_mine = daz * np.cos(alt_ref)#
delv_mine = EL_dets_mine - alt_ref
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
#CASE C
sky_icrs = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
# ICRS → AltAz (THIS is sky_altaz)
sky_altaz = sky_icrs.transform_to(altaz_frame)
ref_altaz = SkyCoord(az=center_altaz.az, alt=center_altaz.alt, frame=altaz_frame)
d_lon, d_lat = ref_altaz.spherical_offsets_to(sky_altaz)
xel_astropy = d_lon.to(u.deg).value   # ΔXEL
el_astropy  = d_lat.to(u.deg).value   # ΔEL
#----------------------------------------------------------------------------

fig,axs = plt.subplots(1,2, figsize=(8,4), dpi=200)
#---
axs[0].plot(az_scan,alt_scan, 'ok')
axs[0].plot(center_altaz.az.to(u.deg).value, center_altaz.alt.to(u.deg).value, '.r')
axs[0].plot(az,                      el, 'xg')
axs[0].set_xlim(center_altaz.az.to(u.deg).value -10*15/3600, center_altaz.az.to(u.deg).value +10*15/3600)
axs[0].set_xlabel('Az [deg]')
axs[0].set_ylabel('El [deg]')
axs[1].plot(XEL, EL, 'ok')
axs[1].plot(xel, delv, '.r')
axs[1].plot(xel_astropy, el_astropy, 'xb')
axs[1].plot(xel_mine, delv_mine, 'xr')
axs[1].set_xlim(0-10*15/3600, 0+10*15/3600)
axs[1].set_xlabel('$\\rm \\Delta$x-EL [deg]')
axs[1].set_ylabel('$\\rm \\Delta$El [deg]')
fig.tight_layout()
plt.show()