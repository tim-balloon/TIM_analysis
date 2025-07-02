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
        cosAz = (np.sin(np.radians(self.coord2)) - np.sin(np.radians(self.lat)) * np.cos(za))/(np.cos(np.radians(self.lat)) * np.sin(za))
        sinAz = - np.sin(np.radians(HA)) * np.cos(np.radians(self.coord2)) / np.sin(za)
        return np.arctan2(sinAz,cosAz)

    def declinationAngle(self):
        """
        source declination angle (rad)
        latitude and cooord2 need to be in degrees.
        coord1 needs to be in radians

        Parameters
        ----------

        Returns
        -------
        Dec: float
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

        return self.lst*np.pi/12 - self.coord1

    def ha2ra(self, hour_angle):

        '''
        Return the right ascension in radians given the lst in hours and the hour angle in radians
        i.e. lst needs to be in hours, hour angle in needs to be in radians 
        Parameters
        ----------
        Returns
        ----------
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
        ----------
        az: array
            Azimuth angle in degree.
        el: array
            Elevation angle in degree.
        '''

        hour_angle = self.ra2ha()
        el = np.pi/2 - self.zenithAngle(np.degrees(hour_angle))
        az = self.azimuthAngle(np.degrees(hour_angle))     
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

        dec = self.declinationAngle()
        hour_angle  = self.azeltoha()
        ra = self.ha2ra(hour_angle)

        return np.degrees(ra), np.degrees(dec)

    def parallactic_angle(self):

        '''
        Compute the parallactic angle which is returned in degrees  

        Parameters
        ----------
        Returns
        ----------
        pa: array
            Parallactic angle angle in degree.
        '''

        hour_angle = self.ra2ha() 
        index, = np.where(hour_angle<0)
        hour_angle[index] += 2*np.pi

        pa = np.arctan2(np.sin(hour_angle), np.cos(np.radians((self.coord2))) * np.tan(np.radians(self.lat)) - np.sin(np.radians((self.coord2))) * np.cos(hour_angle))

        #y_pa = np.cos(self.lat)*np.sin(hour_angle)
        #x_pa = np.sin(self.lat)*np.cos(self.coord2)-np.cos(hour_angle)*np.cos(self.lat)*np.sin(self.coord2)
        #pa = np.arctan2(y_pa, x_pa)

        return np.degrees(pa)

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
        ----------
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

    def __init__(self, coord1, coord2, ctype, xsc_offset, det_offset = np.array([0.,0.]),\
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
        ----------
        Returns
        -------
        """    
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
        ----------
        Returns
        ra_corrected: array
            corrected array of coordinates one
        dec_corrected: array
            corrected array of coordinates two
        -------
        """  
        if self.ctype.lower() == 'ra and dec':

            conv2azel = utils(self.coord1, self.coord2, self.lst, self.lat) #hour, deg, hour, deg
            az, el = conv2azel.radec2azel()
            #xEL = np.degrees(np.radians(az)*np.cos(np.radians(el)))
            ra_corrected = np.zeros((int(np.size(self.det_offset)/2), len(az)))
            dec_corrected = np.zeros((int(np.size(self.det_offset)/2), len(az)))

            for i in range(int(np.size(self.det_offset)/2)):
                
                quaternion = quat.quaternions()
                xsc_quat = quaternion.eul2quat(self.xsc_offset[0], self.xsc_offset[1], 0)
                det_quat = quaternion.eul2quat(self.det_offset[i,0], self.det_offset[i,1], 0)
                off_quat = quaternion.product(det_quat, xsc_quat)

                xEL_offset, EL_offset, roll_offset = quaternion.quat2eul(off_quat)
                EL_corrected_temp = el + EL_offset
                AZ_corrected_temp = az + xEL_offset #np.degrees(np.radians(xEL_corrected_temp)/np.cos(np.radians(el)))
                conv2radec = utils(AZ_corrected_temp, EL_corrected_temp, self.lst, self.lat) #deg, deg, hour, deg
                ra_corrected[i,:], dec_corrected[i,:] = conv2radec.azel2radec()

            del EL_corrected_temp
            del AZ_corrected_temp
            gc.collect()

            return ra_corrected, dec_corrected
        
        elif self.ctype.lower() == 'az and el':

            xEL = np.degrees(np.radians(self.coord1)*np.cos(np.radians(self.coord2)))
            el_corrected = np.zeros((int(np.size(self.det_offset)/2), len(self.coord2)))
            az_corrected = np.zeros((int(np.size(self.det_offset)/2), len(self.coord1)))

            for i in range(int(np.size(self.det_offset)/2)):
                
                quaternion = quat.quaternions()
                xsc_quat = quaternion.eul2quat(self.xsc_offset[0], self.xsc_offset[1], 0)
                det_quat = quaternion.eul2quat(self.det_offset[i,0], self.det_offset[i,1], 0)
                off_quat = quaternion.product(det_quat, xsc_quat)
                xEL_offset, EL_offset, roll_offset = quaternion.quat2eul(off_quat)
                xEL_corrected_temp = xEL+xEL_offset
                az_corrected[i, :]  = np.degrees(np.radians(xEL_corrected_temp)/np.cos(np.radians(self.coord2)))
                el_corrected[i, :] = self.coord2+self.xsc_offset[1]+self.det_offset[i, 1]

                '''
                el_corrected[i, :] = self.coord2+self.xsc_offset[1]+self.det_offset[i, 1]

                az_corrected[i, :] = (self.coord1*np.cos(np.radians(self.coord2))-self.xsc_offset[0]- ##!! self.xsc_offset[i]
                                      self.det_offset[i, 0])/np.cos(np.radians(el_corrected[i, :]))
                '''
               
            return az_corrected, el_corrected

        else:

            el_corrected = np.zeros((int(np.size(self.det_offset)/2), len(self.coord1)))
            xel_corrected = np.zeros((int(np.size(self.det_offset)/2), len(self.coord2)))
            for i in range(int(np.size(self.det_offset)/2)):
                xel_corrected[i, :] = self.coord1-self.xsc_offset[0]-self.det_offset[i, 0]
                el_corrected[i, :]  = self.coord2+self.xsc_offset[1]+self.det_offset[i, 1]
            return xel_corrected,el_corrected
        
class compute_offset(object):
    '''
    Parameters
    ----------
    Returns
    -------
    '''

    def __init__(self, coord1_ref, coord2_ref, map_data, \
                 pixel1_coord, pixel2_coord, wcs_trans, ctype, \
                 lst, lat):

        self.coord1_ref = coord1_ref           #Reference value of the map along the x axis in RA and DEC
        self.coord2_ref = coord2_ref           #Reference value of the map along the y axis in RA and DEC
        self.map_data = map_data               #Maps 
        self.pixel1_coord = pixel1_coord       #Array of the coordinates converted in pixel along the x axis
        self.pixel2_coord = pixel2_coord       #Array of the coordinates converted in pixel along the y axis
        self.wcs_trans = wcs_trans             #WCS transformation 
        self.ctype = ctype                     #Ctype of the map
        self.lst = lst                         #Local Sideral Time
        self.lat = lat                         #Latitude

    def centroid(self, threshold=0.275):

        '''
        For more information about centroid calculation see Shariff, PhD Thesis, 2016
        Parameters
        ----------
        Returns
        -------`
        '''

        maxval = np.max(self.map_data)
        #minval = np.min(self.map_data)
        y_max, x_max = np.where(self.map_data == maxval)

        #lt_inds = np.where(self.map_data < threshold*maxval)
        gt_inds = np.where(self.map_data > threshold*maxval)

        weight = np.zeros((self.map_data.shape[0], self.map_data.shape[1]))
        weight[gt_inds] = 1.
        a = self.map_data[gt_inds]
        flux = np.sum(a)

        x_coord_max = np.floor(np.amax(self.pixel1_coord))+1
        x_coord_min = np.floor(np.amin(self.pixel1_coord))

        x_arr = np.arange(x_coord_min, x_coord_max)

        y_coord_max = np.floor(np.amax(self.pixel2_coord))+1
        y_coord_min = np.floor(np.amin(self.pixel2_coord))

        y_arr = np.arange(y_coord_min, y_coord_max)

        xx, yy = np.meshgrid(x_arr, y_arr)
        
        x_c = np.sum(xx*weight*self.map_data)/flux
        y_c = np.sum(yy*weight*self.map_data)/flux

        return np.rint(x_c), np.rint(y_c)
    
    def value(self):
        '''
        Parameters
        ----------
        Returns
        -------
        '''

        #Centroid of the map
        x_c, y_c = self.centroid()
               
        if self.ctype.lower() == 'ra and dec':
            map_center = wcs.utils.pixel_to_skycoord(x_c, y_c, self.wcs_trans)
            print('Centroid', map_center)
            print(self.wcs_trans.all_pix2world(np.array([[x_c,y_c]]), 0))
            x_map = map_center.ra.hour
            y_map = map_center.dec.degree
            print('b1', x_c, y_c, x_map*15., y_map, np.average(self.lst), np.average(self.lat))
            centroid_conv = utils(x_map, y_map, np.average(self.lst), np.average(self.lat))

            coord1_reference = self.coord1_ref/15.

            az_centr, el_centr = centroid_conv.radec2azel()
            xel_centr = az_centr*np.cos(np.radians(el_centr))

        else:
            map_center = self.wcs_trans.wcs_pix2world(x_c, y_c, 1)
            coord1_reference = self.coord1_ref
            el_centr = y_map
            if self.cytpe.lower() == 'xel and el':
                xel_centr = x_map            
            else:
                xel_centr = x_map/np.cos(np.radians(el_centr))
            

        ref_conv = utils(coord1_reference, self.coord2_ref, np.average(self.lst), \
                         np.average(self.lat))

        az_ref, el_ref = ref_conv.radec2azel()

        xel_ref = az_ref*np.cos(np.radians(el_ref))

        return xel_centr-xel_ref, el_ref+el_centr




        








        

        
        


