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

import numpy as np
import gc
from astropy import wcs
from IPython import embed
import src.quaternion as quat
import matplotlib.pyplot as plt

class utils(object):

    '''
    class to handle conversion between different coodinates sytem 
    Parameters
    ----------
    Returns
    -------
    '''

    def __init__(self, coord1, coord2, lst = None, lat = None):

        '''
        class to handle conversion between different coodinates sytem 
        Parameters
        ----------
        coord1: 1d array
            array of coord 1 converted in degrees   
        coord2: 1d array
            array of coord 2 converted in degrees   
        lst: 1d array
            local Sideral Time in hours
        lat: 1d array
            latitude converted in degrees

        Returns
        -------
        '''
        
        self.coord1 = np.radians(coord1)
        self.coord2 = np.radians(coord2)

        self.lst = None if lst is None else np.asarray(lst) * np.pi / 12.0
        self.lat = None if lat is None else np.radians(lat)

    def zenithAngle(self,ha):
        """
        source zenith angle (rad)
        latitutde and coord2 need to be in degrees.

        Parameters
        ----------
        HA: array
            hour angle in radians
        Returns
        -------
        za: array
            zenith angle in radians
        """
        return np.arccos( np.sin(self.lat) * np.sin(self.coord2) + np.cos(self.lat) * np.cos(self.coord2) * np.cos(ha) )

    def azimuthAngle(self, ha):
        """
        source azimuth angle (rad)
        latitude and coord2 need to be in degrees.

        Parameters
        ----------
        HA: array
            hour angle in radians

        Returns
        -------
        aa: array
            source azimuth angle (rad)
        """ 

        za = self.zenithAngle(ha)

        cosA = ( np.sin(self.coord2) - np.sin(self.lat) * np.cos(za) ) / (np.cos(self.lat) * np.sin(za))

        sinA = ( -np.sin(ha) * np.cos(self.coord2) / np.sin(za) )

        return np.arctan2(sinA, cosA)


    def declinationAngle(self):
        """
        source declination angle (rad)
        latitude and cooord2 need to be in degrees.
        coord1 needs to be in radians

        Parameters
        ----------

        Returns
        -------
        Dec: array
            source declination angle (rad)
        """ 

        azi = self.coord1; alt =self.coord2 
        sinDec = np.sin(np.radians(alt))*np.sin(np.radians(self.lat)) + np.cos(np.radians(alt))*np.cos(np.radians(self.lat))*np.cos(azi)
        return np.arcsin(sinDec)
    
    def azeltoha(self):

        """
        source hour angle (rad)
        latitude and coord2 need to be in degrees
        coord1 needs to be in radians

        Parameters
        ----------

        Returns
        -------
        ha: array
            source hour angle (rad)
        """ 

        tanHA = - np.sin(self.coord1) / (np.tan(np.radians(self.coord2)) * np.cos(np.radians(self.lat)) - np.cos(self.coord1)*np.sin(np.radians(self.lat)))
        HA = np.arctan(tanHA)

        return HA

    def ra2ha(self):

        '''
        Return the hour angle in radians given the lst in hours and RA in radians
        i.e. lst needs to be in hours, ra in needs to be in radians 
        Parameters
        ----------
        Returns
        ----------
        ha: array
            hour angle in hour
        ''' 

        if self.lst is None: raise ValueError("LST must be provided for RA→HA conversion")
        return self.lst - self.coord1
    
    def ha2ra(self, ha):

        '''
        Return the right ascension in radians given the lst in hours and the hour angle in radians
        i.e. lst needs to be in hours, hour angle in needs to be in radians 
        Parameters
        ----------
        hour_angle: array
            source hour angle in radians
        Returns
        ----------
        ra: array
            Right Ascension angle in hour
        '''

        if self.lst is None: raise ValueError("LST must be provided for HA→RA conversion")
        return self.lst - ha
    
    def radec2azel(self):

        '''
        Function to convert RA and DEC to AZ and EL
        Parameters
        ----------
        Returns
        ----------
        az: array
            Azimuth angle in degree.
        el: array
            Elevation angle in degree.
        '''
        ha = self.ra2ha()

        za = self.zenithAngle(ha)
        el = np.pi / 2.0 - za
        az = self.azimuthAngle(ha)

        print('hour angle in wrong [rad]: ', ha)
        print('zenith angle in wrong [rad]: ', za)
        print('elevation in wrong [deg]: ', np.degrees(el))
        print('RA in wrong [deg]: ', np.degrees(az))

        return np.degrees(az), np.degrees(el)

    def azel2radec(self):

        '''
        Function to convert AZ and EL to RA and DEC
        Parameters
        ----------
        Returns
        ----------
        ra: array
            Right Ascension angle in degree.
        dec: array
            Declination angle in degree.
        '''

        az = self.coord1
        el = self.coord2

        sin_dec = ( np.sin(el) * np.sin(self.lat) + np.cos(el) * np.cos(self.lat) * np.cos(az) )
        dec = np.arcsin(sin_dec)

        ha = np.arctan2( -np.sin(az), np.tan(el) * np.cos(self.lat) - np.cos(az) * np.sin(self.lat) )

        ra = self.ha2ra(ha)

        return np.degrees(ra), np.degrees(dec)

class apply_offset(object):
    """
    Class to apply the offset to different coordinates

    Parameters
    ----------
    Returns
    -------
    """    

    def __init__(self, input_ctype, coord1, coord2, ctype, xsc_offset = (0., 0.), det_offset = np.array([[0., 0.]]), lst = None, lat = None):
        
        """
        Return an instance of the apply_offset class

        Parameters
        ----------
        coord1: array
            Array of coordinate 1
        coord2: array
            Array of coordinate 2
        ctype: array
            Ctype of the map
        xsc_offset: tuple
            Offset with respect to star cameras in xEL and EL
        det_offset: 2d array
            Offset with respect to the central detector in xEL and EL
        lst: array
            Local Sideral Time array
        lat: array
            Latitude array
        Returns
        -------
        """    
        self.input_ctype = input_ctype          #Ctype of the coordinates
        self.coord1 = coord1                    #Array of coordinate 1
        self.coord2 = coord2                    #Array of coordinate 2
        self.ctype = ctype                      #Ctype of the map
        self.xsc_offset = xsc_offset            #Offset with respect to star cameras in xEL and EL
        self.det_offset = det_offset            #Offset with respect to the central detector in xEL and EL
        self.lst = lst                          #Local Sideral Time array
        self.lat = lat                          #Latitude array

    def correction(self):
        """
        Apply offset
        Parameters
        ----------
        Returns
        -------
        ra_corrected: array
            corrected array of coordinates one
        dec_corrected: array
            corrected array of coordinates two
        """  

        if self.ctype.lower() == 'ra and dec':

            if(self.input_ctype.lower() == 'ra and dec'): 
                conv2azel = utils(self.coord1, self.coord2, self.lst, self.lat) #hour, deg, hour, deg
                az, el = conv2azel.radec2azel()
            elif(self.input_ctype.lower() == 'az and el'):
                az, el = self.coord1, self.coord2
            else: 
                el = self.coord2
                az = np.degrees(np.radians(self.coord1)/np.cos(np.radians(el)))

            xEL = np.degrees(np.radians(az)*np.cos(np.radians(el)))
            ra_corrected = np.zeros((int(np.size(self.det_offset)/2), len(az)))  
            dec_corrected = np.zeros((int(np.size(self.det_offset)/2), len(az)))

            for i in range(int(np.size(self.det_offset)/2)):
                
                quaternion = quat.quaternions()
                xsc_quat = quaternion.eul2quat(self.xsc_offset[0], self.xsc_offset[1], 0)
                det_quat = quaternion.eul2quat(self.det_offset[i,0], self.det_offset[i,1], 0)
                off_quat = quaternion.product(det_quat, xsc_quat)

                xEL_offset, EL_offset, roll_offset = quaternion.quat2eul(off_quat)
                EL_corrected_temp = el + EL_offset
                xEL_corrected_temp = xEL - xEL_offset
                AZ_corrected_temp = np.degrees(np.radians(xEL_corrected_temp)/np.cos(np.radians(el)))

                conv2radec = utils(AZ_corrected_temp, EL_corrected_temp, self.lst, self.lat) #deg, deg, hour, deg
                ra_corrected[i,:], dec_corrected[i,:] = conv2radec.azel2radec()

            del EL_corrected_temp
            del AZ_corrected_temp
            gc.collect()

            return ra_corrected, dec_corrected
        
        elif self.ctype.lower() == 'az and el':
            if(self.input_ctype.lower() == 'ra and dec'): 
                conv2azel = utils(self.coord1, self.coord2, self.lst, self.lat) #hour, deg, hour, deg
                az, el = conv2azel.radec2azel()
            elif(self.input_ctype.lower() == 'az and el'):
                az, el = self.coord1, self.coord2
            else: 
                el = self.coord2
                az = np.degrees(np.radians(self.coord1)/np.cos(np.radians(el)))

            xEL = np.degrees(np.radians(az)*np.cos(np.radians(el)))
            cos_el = np.cos(np.radians(el))
            el_corrected = np.zeros((int(np.size(self.det_offset)/2), len(self.coord2)))
            az_corrected = np.zeros((int(np.size(self.det_offset)/2), len(self.coord1)))

            for i in range(int(np.size(self.det_offset)/2)):
                
                #xsc_quat = quaternion.eul2quat(self.xsc_offset[0], self.xsc_offset[1], 0)
                #det_quat = quaternion.eul2quat(self.det_offset[i,0], self.det_offset[i,1], 0)
                el_corrected[i, :] = np.asarray(el)+self.xsc_offset[1]+self.det_offset[i, 1]

                az_corrected[i, :] = (xEL-self.xsc_offset[0]-self.det_offset[i, 0]) / cos_el

               
            return az_corrected, el_corrected

        else:

            if(self.input_ctype.lower() == 'ra and dec'): 
                conv2azel = utils(self.coord1, self.coord2, self.lst, self.lat) #hour, deg, hour, deg
                az, el = conv2azel.radec2azel()
                xEL = np.degrees(np.radians(az)*np.cos(np.radians(el)))
            elif(self.input_ctype.lower() == 'az and el'):
                az, el = self.coord1, self.coord2
                xEL = np.degrees(np.radians(az)*np.cos(np.radians(el)))
            else: 
                el = self.coord2
                xEL = self.coord1 
                

            el_corrected = np.zeros((int(np.size(self.det_offset)/2), len(self.coord1)))
            xel_corrected = np.zeros((int(np.size(self.det_offset)/2), len(self.coord2)))
            for i in range(int(np.size(self.det_offset)/2)):
                xel_corrected[i, :] = xEL-self.xsc_offset[0]-self.det_offset[i, 0]
                el_corrected[i, :]  = el+self.xsc_offset[1]+self.det_offset[i, 1]

            return xel_corrected,el_corrected


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
file = '/home/mvancuyck/Desktop/TIM_analysis/timestream_maker/fits_and_hdf5/cube_1source_with_1xbigger_sigma_PSF.fits'
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
        if(False):
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
corr = apply_offset('RA and DEC', (wcs.wcs.crval[0],), (wcs.wcs.crval[1],), 'AZ and EL', lst = LST.value, lat = lat.value, )
azi_ref, alt_ref = corr.correction()
conv = apply_offset('RA and DEC', np.zeros_like(ra_deg), dec_deg, 'AZ and EL', lst = LST.value, lat = lat.value, )
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

fig,axs = plt.subplots(1,2, figsize=(12,4))
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
plt.show()