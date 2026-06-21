import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from astropy.wcs import WCS
from astropy.io import fits
import astropy.units as u
import matplotlib.pyplot as plt
import astropy.convolution as conv
from astropy.coordinates import SkyCoord, AltAz, EarthLocation, Angle
from astropy.time import Time
import src.beam as bm
from scipy.ndimage import shift
from IPython import embed
from astropy.table import Table
from astropy.visualization import ZScaleInterval
zscale = ZScaleInterval()
from mpl_toolkits.axes_grid1 import make_axes_locatable
import src.pointing as pt
import pickle
import warnings

warnings.filterwarnings("ignore")
np.random.seed(43)
dictname = 'results_relatdiff.p'

t = Table.read("LW_pop.fits")
lambda_mm = t["lambda_mm"]   # wavelength in mm
xel_rad = t["xel_rad"]       # cross-elevation offset from boresight, radians
el_rad = t["el_rad"]         # elevation-direction offset from boresight, radians
lambda_um = lambda_mm * 1000
xel_arcsec = xel_rad * 180 / np.pi * 3600
xel_deg = xel_rad * 180 / np.pi 
el_arcsec = el_rad * 180 / np.pi * 3600
lambda_um = np.array(t["lambda_mm"]) * 1000.0 # mm -> um
el_deg = np.array(t["el_rad"]) * 180/np.pi # rad -> deg
EL_list = np.asarray((el_deg.min(), el_deg.mean(), el_deg.max()))
XEL_list = np.asarray((xel_deg.min(),xel_deg.mean(),xel_deg.max() ))+ 0.033

sigma = 8.10839017e-05 *u.rad
# ------------------------------------------------------------
# SOURCE POSITION
# ------------------------------------------------------------
ra  = 0 * u.deg
dec = -27.80833 * u.deg
lat = -77.83 * u.deg
lon = 0 * u.deg
location = EarthLocation(lat=lat, lon=lon)

# --------------------------------------------------------------------------------------
# Build time with LST = 0
# --------------------------------------------------------------------------------------
date = "2025-01-01T00:00:00"
#t0 = Time(date, scale="utc", location=location)
#lst0 = t0.sidereal_time("apparent")
#dt = (lst0 / (24 * u.hourangle)) * u.sday
t = Time("2025-01-01T00:00:00", scale="utc", location=location)
LST = t.sidereal_time("apparent").hour
print("Actual LST:", t.sidereal_time("apparent").hour)  # should be ~0 but may not be exactly
EL = np.array([-4.5990390e-01, -1.6690978e-04,  4.5946336e-01]); XEL = np.array([-0.00019651,  0.00050358,  0.00154953]) ; lambda_umm = np.asarray((311.66602, 370.5099, 425.31))              
realizations = 100
SNR_list = np.array([1.00000000e+00, 1.45544321e+00, 2.11831493e+00, 3.08308708e+00,
       4.48725815e+00, 6.53094940e+00, 9.50542595e+00, 1.38346076e+01,
       2.01354857e+01, 2.93060560e+01, 4.26533001e+01, 6.20794560e+01,
       9.03531225e+01, 1.31503839e+02, 1.91396369e+02, 2.78566545e+02,
       4.05437786e+02, 5.90091672e+02, 8.58844916e+02, 1.25000000e+03])
flux_list = (1,  3.25,  5.5 ,  7.75, 10)
res_list = (5,10,15,20)*u.arcsec

if __name__ == "__main__":

    results_relatdiff = {'results':np.zeros((realizations, len(res_list), len(flux_list), len(SNR_list))), 
                        'SNR_list':SNR_list,  'backgrounbd_list':flux_list, 'res_list':res_list, 'realizations':realizations}

    from progress.bar import Bar
    bar = Bar(f'Processing', max=results_relatdiff['results'].size)
    for real in range(realizations):
        for ires, res in enumerate(res_list):
            for ib, flux in enumerate(flux_list):
                for isnr, SNR in enumerate(SNR_list):
                    # ------------------------------------------------------------
                    # MAP SIZE + RESOLUTION
                    # ------------------------------------------------------------
                    map_deg_size=1*u.deg
                    #res  *= 1 * u.arcsec   # pixel size
                    npix = int(map_deg_size / res.to(u.deg)); 
                    if(npix%2==0): npix+=1
                    #print('npix=', npix)
                    w = WCS(naxis=2)
                    w.wcs.crpix = [npix//2, npix//2]     # center pixel
                    w.wcs.crval = [ra.value, dec.value]
                    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
                    w.wcs.cdelt = np.array([ -res.to(u.deg).value,   res.to(u.deg).value])
                    # ------------------------------------------------------------
                    # DATA MAP (Jy/sr)
                    # ------------------------------------------------------------
                    map_data = np.zeros((npix, npix), dtype=float)
                    map_data[npix//2, npix//2] = flux
                    
                    sigma_pix = sigma / res.to(u.rad)
                    kernel_channel = conv.Gaussian2DKernel(x_stddev=sigma_pix, x_size=npix)
                    kernel_channel.normalize(mode="peak")
                    #size of the beam in pix2
                    beam_area_pix2 = np.sum(kernel_channel.array)
                    map_data = conv.convolve_fft(map_data, kernel_channel, normalize_kernel=True, boundary = 'wrap')
                    map_data /= (res**2).to(u.sr).value
                    sigma_noise = flux/SNR
                    noise = np.random.normal(0.0, sigma_noise, size=map_data.shape)
                    noise /= (res**2).to(u.sr).value
                    map_data_noisy = map_data + noise
                    vmin, vmax = zscale.get_limits(map_data_noisy)
                    # ======================================================================================
                    # SKY CENTER
                    # ======================================================================================
                    sky_center = SkyCoord(ra=ra, dec=dec, frame="icrs",)
                    altaz_frame = AltAz( obstime=t, location=location,)
                    center_altaz = sky_center.transform_to(altaz_frame)
                    cos_alt = np.cos(center_altaz.alt.to(u.rad).value)
                    az_ref = center_altaz.az.to(u.deg).value
                    el_ref = center_altaz.alt.to(u.deg).value
                    corr = pt.apply_offset('RA and DEC',(w.wcs.crval[0],),(w.wcs.crval[1],),'AZ and EL',lst=LST,lat=lat.value)
                    azi_ref_namap, alt_ref_namap = corr.correction()
                    azi_ref_namap = np.asarray(azi_ref_namap)[0]
                    alt_ref_namap = np.asarray(alt_ref_namap)[0]
                    alt_offsets = EL * u.deg
                    az_offsets = (XEL / cos_alt) * u.deg
                    alt_scan = center_altaz.alt + alt_offsets
                    az_scan  = center_altaz.az  + az_offsets
                    scan_altaz = SkyCoord( az=az_scan, alt=alt_scan, frame=altaz_frame,)
                    # ------------------------------------------------------------------
                    # SKY PROJECTION OF SCAN (ALTAZ -> ICRS)
                    # ------------------------------------------------------------------
                    scan_icrs = scan_altaz.transform_to("icrs")
                    ra_point = scan_icrs.ra.wrap_at(180 * u.deg)
                    dec_point = scan_icrs.dec
                    pixel_offsets = np.array([EL, XEL]).T
                    genpath = pt.utils((w.wcs.crval[0],),(w.wcs.crval[1],), LST, lat.value)
                    pointing_paths = np.asarray( [genpath.genPointingPath(offsets) for offsets in pixel_offsets] )
                    # pixel coordinates of scan positions
                    x_pix, y_pix = w.wcs_world2pix( ra_point.to(u.deg).value,dec_point.to(u.deg).value,0)
                    x_pix_mine, y_pix_mine = w.wcs_world2pix( pointing_paths[:,0,0],pointing_paths[:,0,1],0)
                    # ------------------------------------------------------------------
                    # OFFSET RELATIVE TO TRUE CENTER (SELF-CONSISTENT DEFINITION)
                    # ------------------------------------------------------------------
                    dx = x_pix - w.wcs.crpix[0]; dy = y_pix - w.wcs.crpix[1]
                    dx_mine = x_pix_mine - w.wcs.crpix[0]; dy_mine = y_pix_mine - w.wcs.crpix[1]
                    # ------------------------------------------------------------------
                    # FIT STORAGE
                    # ------------------------------------------------------------------
                    x_peaks = []
                    y_peaks = []
                    x_peaks_mine = []
                    y_peaks_mine = []
                    # ------------------------------------------------------------------
                    # LOOP OVER SCAN POINTS
                    # ------------------------------------------------------------------
                    cut_size = int(20*res.value/5)
                    half = cut_size // 2
                    yc0 = npix // 2
                    xc0 = npix // 2
                    for j, (dxi, dyi, dxi_mine, dyi_mine) in enumerate(zip(dx, dy, dx_mine, dy_mine)):

                        shifted = shift(map_data ,shift=(dyi, dxi),order=3,mode="constant",cval=0.0    )
                        shifted += noise
                        xc = int(np.round(xc0 + dxi))
                        yc = int(np.round(yc0 + dyi))
                        x0 = max(0, xc - half)
                        x1 = min(npix, xc + half)
                        y0 = max(0, yc - half)
                        y1 = min(npix, yc + half)
                        submap = shifted[y0:y1, x0:x1]
                        beam_value = bm.beam(submap)
                        beam_map = beam_value.beam_fit()
                        if isinstance(beam_map[0], str):
                            #fig, ax = plt.subplots(figsize=(3, 3))
                            #ax.imshow(submap, origin="lower", vmin=vmin, vmax=vmax)
                            x_peaks.append(np.nan); y_peaks.append(np.nan)
                        else:
                            params = beam_map[1]
                            cov = beam_map[2]
                            uncertainties = np.sqrt(np.diag(cov))
                            xo_sub = params[1]
                            yo_sub = params[2]
                            xo = xo_sub + x0
                            yo = yo_sub + y0
                            x_peaks.append(xo)
                            y_peaks.append(yo)
                        shifted = shift(map_data ,shift=(dyi_mine, dxi_mine),order=3,mode="constant",cval=0.0    )
                        shifted += noise
                        xc = int(np.round(xc0 + dxi))
                        yc = int(np.round(yc0 + dyi))
                        x0 = max(0, xc - half)
                        x1 = min(npix, xc + half)
                        y0 = max(0, yc - half)
                        y1 = min(npix, yc + half)
                        submap = shifted[y0:y1, x0:x1]
                        beam_value = bm.beam(submap)
                        beam_map = beam_value.beam_fit()
                        if isinstance(beam_map[0], str):
                            x_peaks_mine.append(np.nan)
                            y_peaks_mine.append(np.nan)
                        else: 
                            params = beam_map[1]
                            cov = beam_map[2]
                            uncertainties = np.sqrt(np.diag(cov))
                            xo_sub = params[1]
                            yo_sub = params[2]
                            xo = xo_sub + x0
                            yo = yo_sub + y0
                            x_peaks_mine.append(xo)
                            y_peaks_mine.append(yo)

                    plt.close()
                    # ------------------------------------------------------------------
                    # PIXEL -> SKY (CONSISTENT WCS)
                    # ------------------------------------------------------------------
                    x_fits = np.asarray(x_peaks); y_fits = np.asarray(y_peaks)
                    x_fits_mine = np.asarray(x_peaks_mine); y_fits_mine = np.asarray(y_peaks_mine)

                    ra_deg, dec_deg = w.wcs_pix2world(x_fits, y_fits, 0)
                    ra_deg_mine, dec_deg_mine = w.wcs_pix2world(x_fits_mine, y_fits_mine, 0)
                    # ------------------------------------------------------------------
                    # SKY -> ALTAZ
                    # ------------------------------------------------------------------
                    sky_icrs_fit = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg, frame="icrs")
                    sky_altaz_fit = sky_icrs_fit.transform_to(altaz_frame)
                    az_fit = sky_altaz_fit.az.to(u.deg).value
                    el_fit = sky_altaz_fit.alt.to(u.deg).value
                    # ------------------------------------------------------------------
                    # TRUE SCAN REFERENCE
                    # ------------------------------------------------------------------
                    daz = az_fit - az_ref
                    delv = el_fit - el_ref
                    xel_fit = daz * cos_alt
                    ref_altaz = SkyCoord(az=center_altaz.az,alt=center_altaz.alt,frame=altaz_frame)
                    fit_altaz = SkyCoord(az=az_fit*u.deg,alt=el_fit*u.deg,frame=altaz_frame)
                    d_lon, d_lat = ref_altaz.spherical_offsets_to(fit_altaz)
                    xel_astropy = d_lon.to(u.deg).value
                    el_astropy = d_lat.to(u.deg).value
                    # ------------------------------------------------------------------
                    # Namap
                    # ------------------------------------------------------------------
                    if(False): ra_to_use, dec_to_use = ra_deg,dec_deg
                    else:  ra_to_use, dec_to_use = ra_deg_mine,dec_deg_mine
                    apply = pt.apply_offset('RA and DEC',ra_to_use,dec_to_use,'AZ and EL',lst=LST,lat=lat.value)
                    AZ_dets_mine, EL_dets_mine = apply.correction()
                    daz2 = AZ_dets_mine - azi_ref_namap
                    daz2 = (daz2 + 180) % 360 - 180
                    xel_mine = daz2 * np.cos(np.radians(alt_ref_namap))  #alt_ref_namap
                    delv_mine = EL_dets_mine - alt_ref_namap
                    relatdiff_xel = np.nanmean((xel_mine-XEL)/XEL)
                    relatdiff_el = np.nanmean((delv_mine-EL)/EL)
                    results_relatdiff['results'][real,ires, ib, isnr] = np.nanmean((relatdiff_xel,relatdiff_el))
                    bar.next()
    bar.finish
    pickle.dump(results_relatdiff, open(dictname, 'wb'))