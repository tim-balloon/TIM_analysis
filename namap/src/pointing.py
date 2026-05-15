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
        self.coord1 = np.radians(coord1)  #Array of coord 1 converted in degrees   
        self.coord2 = coord2  #Array of coord 2 converted in degrees   
        self.lst = lst        #Local Sideral Time in hours
        self.lat = lat        #Latitude converted in degrees

    def zenithAngle(self,HA):
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

        za = np.arccos(np.sin(np.radians(self.lat)) * np.sin(np.radians(self.coord2)) + np.cos(np.radians(self.lat)) * np.cos(np.radians(self.coord2)) * np.cos(np.radians(HA)))

        return za

    def azimuthAngle(self, HA):
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

        za = self.zenithAngle(HA)
        #cosAz = (np.sin(np.radians(self.coord2)) - np.sin(np.radians(self.lat)) * np.cos(za))/(np.cos(np.radians(self.lat)) * np.sin(za))
        #sinAz = - np.sin(np.radians(HA)) * np.cos(np.radians(self.coord2)) / np.sin(za)

        sinAz =  -np.sin(np.radians(HA)) * np.cos(np.radians(self.coord2)) / np.sin(za)
        cosAz = (np.sin(np.radians(self.coord2)) - np.sin(np.radians(self.lat)) * np.cos(za)) / (np.cos(np.radians(self.lat)) * np.sin(za))
        az = np.arctan2(sinAz, cosAz) % (2 * np.pi)
        
        return az

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
        sinDec = np.sin(np.radians(alt))*np.sin(np.radians(self.lat)) + np.cos(np.radians(alt))*np.cos(np.radians(self.lat))*np.cos(np.radians(azi))
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
        HA = np.arctan2(tanHA)

        return HA

    def ra2ha(self):

        '''
        Return the hour angle in radians given the lst in hours and RA in radians
        i.e. lst needs to be in hours, ra in needs to be in radians 
        Parameters
        ----------
        Returns
        -------
        ha: array
            hour angle in hour
        ''' 
        ha = self.lst*np.pi/12 - self.coord1
        return  (ha + np.pi) % (2 * np.pi) - np.pi

    def ha2ra(self, hour_angle):

        '''
        Return the right ascension in radians given the lst in hours and the hour angle in radians
        i.e. lst needs to be in hours, hour angle in needs to be in radians 
        Parameters
        ----------
        hour_angle: array
            source hour angle in radians
        Returns
        -------
        ra: array
            Right Ascension angle in hour
        '''
        return self.lst*np.pi/12 - hour_angle

    def radec2azel(self):

        '''
        Function to convert RA and DEC to AZ and EL
        Parameters
        ----------
        Returns
        -------
        az: array
            Azimuth angle in degree.
        el: array
            Elevation angle in degree.
        '''

        hour_angle = self.ra2ha()
        el = np.pi/2 - self.zenithAngle(np.degrees(hour_angle))
        az = self.azimuthAngle(np.degrees(hour_angle))     
        
        return np.degrees(az), np.degrees(el)

    def elevationAngle(self, HA): 
        """
        elevation angle (rad)

        Parameters
        ----------
        dec: float 
            declination angle in degrees     
        lat: float
            latitude angle in degrees
        HA: array
            hour angle in hour

        Returns
        -------
        ea: array
            elevation angle in degree
        """ 

        return np.pi/2 - self.zenithAngle(HA)

    def declinationAngle(self, azi, alt):
        """
        source declination angle (rad)

        Parameters
        ----------
        azi: float 
            azimuth in degrees     
        alt: float
            latitude  angle in degrees
        lat: float
            latitude angle in degree

        Returns
        -------
        Dec: float
            source declination angle (rad)
        """ 
        sinDec =  np.sin(np.radians(alt))*np.sin(np.radians(self.lat)) + np.cos(np.radians(alt))*np.cos(np.radians(self.lat))*np.cos(np.radians(azi))
        return np.arcsin(sinDec)
    
    def hourAngle(self, azi, alt):
        
        tanHA = - np.sin(np.radians(azi)) / (np.tan(np.radians(alt)) * np.cos(np.radians(self.lat)) - np.cos(np.radians(azi))*np.sin(np.radians(self.lat)))
        HA = np.arctan(tanHA)
        '''
        sin_dec = np.sin(np.radians(alt))*np.sin(np.radians(lat)) + np.cos(np.radians(alt))*np.cos(np.radians(lat))*np.cos(np.radians(azi))
        dec = np.arcsin(sin_dec)
        
        sin_HA = -np.sin(np.radians(azi))*np.cos(np.radians(alt)) / np.cos(dec)
        cos_HA = (np.sin(np.radians(alt)) - np.sin(dec)*np.sin(np.radians(lat))) / (np.cos(dec)*np.cos(np.radians(lat)))
        HA = np.arctan2(sin_HA, cos_HA)
        '''
        return HA

    def genPointingPath(self, offsets=np.zeros(2), azel=False):
        """
        Function that takes local paths and generates the pointing on sky vs time.
        Parameters
        ----------
        offsets: array [EL_offset_deg, XEL_offset_deg]
            EL and cross-EL offsets in degrees
        Returns
        -------
        path: nd array
            the RA/Dec coordinates of the pointing, in degrees
        """
        # Hour angle in radians (ra2ha returns radians)
        ha = self.ra2ha()  # radians

        # zenithAngle/azimuthAngle expect HA in DEGREES internally (they call np.radians on it)
        ha_deg = np.degrees(ha)

        # Elevation and azimuth of the phase center (radians)
        el = self.elevationAngle(ha_deg)   # radians
        az = self.azimuthAngle(ha_deg)     # radians

        # Apply EL and XEL offsets (offsets[0]=EL deg, offsets[1]=XEL deg)
        el_off  = np.radians(offsets[0])
        xel_off = np.radians(offsets[1])

        # XEL is perpendicular to EL in the Az direction, scaled by 1/cos(el)
        az_off = xel_off / np.cos(el)

        az_new = az + az_off
        el_new = el + el_off

        # Convert back: az/el (radians) -> Dec and HA
        dec_point = self.declinationAngle(np.degrees(az_new), np.degrees(el_new))  # radians

        # Hour angle from az/el using arctan2 for correct quadrant
        azi_r = np.radians(np.degrees(az_new))  # keep as radians for clarity
        alt_r = el_new

        sin_HA = -np.sin(azi_r) * np.cos(alt_r) / np.cos(dec_point)
        cos_HA = (np.sin(alt_r) - np.sin(dec_point) * np.sin(np.radians(self.lat))) / \
                (np.cos(dec_point) * np.cos(np.radians(self.lat)))
        ha_point = np.arctan2(sin_HA, cos_HA)  # radians

        # RA = LST - HA  (LST in radians = lst_hours * pi/12)
        lst_rad = self.lst * np.pi / 12
        ra = lst_rad - ha_point

        ra_unwrapped = np.unwrap(ra)

        path = np.vstack((np.degrees(ra_unwrapped), np.degrees(dec_point))).T
        
        return path
    
class convert_to_telescope(object):

    '''
    Class to convert from sky equatorial coordinates to telescope coordinates
    Parameters
    ----------
    Returns
    ----------
    '''

    def __init__(self, coord1, coord2, lst, lat):

        self.coord1 = coord1           #RA, needs to be in hours       
        self.coord2 = coord2           #DEC
        self.lst = lst 
        self.lat = lat

    def conversion(self):

        '''
        This function rotates the coordinates projected on the plane using the parallactic angle
        Parameters
        ----------
        Returns
        -------
        '''
        
        parang = utils(self.coord1, self.coord2, self.lst, self.lat)
        pa = parang.parallactic_angle()

        x_tel = np.radians(self.coord1*15)*np.cos(pa)-np.radians(self.coord2)*np.sin(pa)
        y_tel = np.radians(self.coord2)*np.cos(pa)+np.radians(self.coord1*15)*np.sin(pa)

        return np.degrees(x_tel), np.degrees(y_tel)

class apply_offset(object):
    """
    Class to apply the offset to different coordinates

    Parameters
    ----------
    Returns
    -------
    """    

    def __init__(self, input_ctype, coord1, coord2, ctype, xsc_offset=(0,0), DT=np.float64, IT=np.int64, det_offset = np.zeros((1, 2)),\
                 lst = None, lat = None):
        
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
        self.DT = DT                            #Float precision required
        self.IT = IT                            #Integer precision required

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
            ra_corrected = np.zeros((int(np.size(self.det_offset)/2), len(az))).astype(self.DT)
            dec_corrected = np.zeros((int(np.size(self.det_offset)/2), len(az))).astype(self.DT)

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
            el_corrected = np.zeros((int(np.size(self.det_offset)/2), len(self.coord2))).astype(self.DT)
            az_corrected = np.zeros((int(np.size(self.det_offset)/2), len(self.coord1))).astype(self.DT)

            for i in range(int(np.size(self.det_offset)/2)):
                
                #xsc_quat = quaternion.eul2quat(self.xsc_offset[0], self.xsc_offset[1], 0)
                #det_quat = quaternion.eul2quat(self.det_offset[i,0], self.det_offset[i,1], 0)
                el_corrected[i, :] = el+self.det_offset[i, 1]+self.xsc_offset[1]
                az_corrected[i, :] = (xEL-self.xsc_offset[0]-self.det_offset[i, 0]) / cos_el

            return az_corrected, el_corrected

        else:

            if(self.input_ctype == self.input_ctype.lower() == 'ra and dec'): 
                conv2azel = utils(self.coord1, self.coord2, self.lst, self.lat) #hour, deg, hour, deg
                az, el = conv2azel.radec2azel()
                xEL = np.degrees(np.radians(az)*np.cos(np.radians(el)))
            elif(self.input_ctype == self.input_ctype.lower() == 'az and el'):
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
        







        

        
        


