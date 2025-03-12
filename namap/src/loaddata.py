import pygetdata as gd
import numpy as np
from scipy.interpolate import interp1d
import os
import astropy.table as tb
from IPython import embed
import src.detector as det 
import h5py
import matplotlib.pyplot as plt
 

def loadspf(file, field):
    """
    Load the sample per frame of a field from a .hdf5 
    Parameters
    ----------
    file: string
        the name of the .hdf5 file
    field: string
        the field for which to get the spf
    Returns
    -------
    spf: int
        number of sample per frame
    """    
    H = h5py.File(file, "a")
    f = H[field]
    if('spf' in f.keys()): spf = f['spf'][()]
    else: spf = None
    H.close()
    return spf

class data_value():
    
    '''
    Class for reading the values of the TODs (detectors and coordinates) from a DIRFILE
    Parameters
    ----------
    Returns
    -------
    '''

    def __init__(self, det_path, det_name, coord_path, coord1_name, \
                 coord2_name, lst, lat,  startframe, numframes, 
                 roach_number, telemetry=False):

        """
        Class to load the data from a .hdf5 
        Parameters
        ----------
        det_path: string
            Path of the detector dirfile
        det_name: string
            Detector name to be analyzed
        coord_path: string
            Path of the coordinates dirfile
        coord1_name: string
            Coordinates 1 name, e.g. RA or AZ
        coord2_name: string
            Coordinates 2 name
        lst: bool
            if True, load the LST  coordinate
        lat: bool
            if True, load the LAT  coordinate
        startframe:
            Starting frame to be analyzed
        numframes:
            Ending frame to be analyzed

        Returns
        -------
        self: class
            Instance of the data_value class
        """    
        self.det_path = det_path                    #Path of the detector dirfile
        self.det_name = det_name                    #Detector name to be analyzed
        self.coord_path = coord_path                #Path of the coordinates dirfile
        self.coord1_name = coord1_name              #Coordinates 1 name, e.g. RA or AZ
        self.coord2_name = coord2_name              #Coordinates 2 name
        self.lst_file_type = lst #LST bool
        self.lat_file_type = lat #LAT bool
        self.startframe = startframe        #Starting frame to be analyzed
        self.numframes = numframes           #Ending frame to be analyzed

        if self.startframe < 100:
            self.bufferframe = int(0)  #Buffer frames to be loaded before and after the starting and ending frame
        else:
            self.bufferframe = int(100)

        self.telemetry = telemetry
        self.roach_number = roach_number

    
    def loaddata(file, field, num_frames=None, first_frame=None):
        """
        Load the data from a .hdf5 
        Equivalent to d.getdata()
        Parameters
        ----------
        file: string
            the name of the .hdf5 file
        field: string
            the field to be loaded
        num_frame: int
            the number of frames to load, with N=spf samples in each frame.
        first_frame: int
            the first frame to load. 

        Returns
        -------
        data: array
            values stores in field, from first_frame*spf to (first_frame+num_frames)*spf. 
        """    
        fF=file
        file = 'fits_and_hdf5/TOD_pySIDES_from_uchuu_TIM_tile0_1deg_1deg_res20arcsec_dnu4.0GHz_CII_de_Looze_smoothed_MJy_sr.hdf5'

        if os.path.isfile(file): H = h5py.File(file, "a")
        else: print('no file')
        f = H[field]
        if(('spf' in f.keys()) and (num_frames is not None) and (first_frame is not None)):
            spf = f['spf'][()]
            data = f['data'][first_frame*spf:(first_frame+num_frames)*spf]
        else: 
            data = f['data'][()]
        H.close()
        return data

    def values(self):
        """      
        Load the RA, Dec, and amplitudes timestreams for a given list of detectors
        Parameters
        ----------
        Returns
        -------
        det_data: array
            Amplitude timestream of a detector
        coord1_data: array
            Coord 1 timestream of a detector
        coord2_data: array
            Coord 2 timestream of a detector
        hwp_data: array
            HWP
        lst: array
            longitude timestream of a detector
        lat: array
            latitude timestream of a detector
        spf_data: int
            the number of saqmple per frame of data. 
        spf_coord: int
            the number of saqmple per frame of coord. 
        hwp_spf: int
            the number of saqmple per frame of hwp. 
        lat_spf: int
            the number of saqmple per frame of lat. 

        """    
        num = self.numframes+self.bufferframe*2
        first_frame = self.startframe+self.bufferframe
        print(num,first_frame)
        kid_num  = self.det_name


        det_data = []

        for kid in kid_num: 
            I_data = data_value.loaddata(self.det_path, f'I_kid{kid}_roach', num, first_frame)
            Q_data = data_value.loaddata(self.det_path, f'Q_kid{kid}_roach', num, first_frame)
            kidutils = det.kidsutils()
            det_data.append(kidutils.KIDmag(I_data, Q_data))
            # Assume all the data have the same spf            
            spf_data = loadspf(self.det_path, f'I_kid{kid}_roach')

        #---------------------------------------------------------------------------------
        coord1_data = data_value.loaddata(self.coord_path, self.coord1_name, num, first_frame)
        coord2_data = data_value.loaddata(self.coord_path, self.coord2_name, num, first_frame)
        spf_coord = loadspf(self.coord_path, self.coord2_name, )
        #---------------------------------------------------------------------------------
        if (self.lat_file_type and self.lst_file_type):
            
            lat = data_value.loaddata(self.coord_path, 'lat',num, first_frame)
            lst = data_value.loaddata(self.coord_path, 'lst',num, first_frame)
            lat_spf = loadspf(self.coord_path, 'lst')

            return det_data, coord1_data, coord2_data, lst, lat, spf_data, spf_coord,lat_spf
        else:
            return det_data, coord1_data, coord2_data, None, None, spf_data, spf_coord,  0

class convert_dirfile():

    '''
    Class for converting TODs from dirfile value to real value, 
    considering a linear conversion
    '''

    def __init__(self, data, param1, param2):

        self.data = data        #DIRFILE TOD
        self.param1 = param1    #gradient of the conversion
        self.param2 = param2    #intercept of the conversion

    def conversion(self):

        self.data = self.param1*self.data+self.param2

class frame_zoom_sync():
    """
    This class is designed to extract the frames of interest from the complete timestream and 
    sync detector and coordinates timestream given a different sampling of the two
    the telemetry name
    Parameters
    ----------
    Returns
    -------
    """    
    def __init__(self, det_path, det_data, det_sample_frame, det_fs, coord1_data, coord2_data, 
                 coord_fs, coord_sample_frame, startframe, numframes, lst_data, lat_data, 
                 lstlatfreq, lstlat_sample_frame, offset, xystage=False):
        """
        Create an instance of frame_zoom_sync class
        Parameters
        ----------
        det_path: string
            Path of the detector dirfile
        det_name: string
            Detector name to be analyzed
        coord_path: string
            Path of the coordinates dirfile
        coord1_name: string
            Coordinates 1 name, e.g. RA or AZ
        coord2_name: string
            Coordinates 2 name
        lst: bool
            if True, load the LST  coordinate
        lat: bool
            if True, load the LAT  coordinate
        startframe:
            Starting frame to be analyzed
        numframes:
            Ending frame to be analyzed

        Returns
        -------
        self: class
            Instance of the data_value class
        """   

        self.det_path = det_path                    #Path of the detector dirfile
        self.det_data = det_data                                #Detector data timestream
        self.det_fs = det_fs                                    #Detector frequency sampling
        self.det_sample_frame = det_sample_frame                #Detector samples in each frame of the timestream
        self.coord1_data = coord1_data                          #Coordinate 1 data timestream
        self.coord_fs = coord_fs                                #Coordinates frequency sampling
        self.coord_sample_frame = coord_sample_frame            #Coordinates samples in each frame of the time stream
        self.coord2_data = coord2_data                          #Coordinate 2 data timestream
        self.startframe = startframe                            #Start frame
        self.numframes = numframes           #Ending frame to be analyzed
        self.lst_data = lst_data                                #LST timestream (if correction is required and coordinates are RA-DEC)
        self.lat_data = lat_data                                #LAT timestream (if correction is required and coordinates are RA-DEC)
        self.lstlatfreq = lstlatfreq                            #LST-LAT sampling frequency (if correction is required and coordinates are RA-DEC)
        self.lstlat_sample_frame = lstlat_sample_frame          #LST-LAT samples per frame (if correction is required and coordinates are RA-DEC)
        self.offset = offset                                    #Time offset between detector data and coordinates


        self.xystage=xystage                                   #Flag to check if the coordinates data are coming from an xy stage scan                       #Flag to check if the coordinates data are coming from an xy stage scan
        
        if self.startframe < 100:
            self.bufferframe = int(0)  #Buffer frames to be loaded before and after the starting and ending frame
        else:
            self.bufferframe = int(100)

        

    def frame_zoom(self, data, sample_frame, fs, fps, offset = None):
        """
        Not used
        Selecting the frames of interest and associate a timestamp for each value.

        Parameters
        ----------
        data: array
            amplitude timestreams
        sample_frame: int
            all the spf?
        fs: int
            all the spf?
        fps: int
            all the spf?
        offset: float
            time offset ? 

        Returns
        -------
        """    

        frames = fps.copy()

        frames[0] = fps[0]*sample_frame
        if fps[1] == -1:
            frames[1] = len(data)*sample_frame
        else:
            frames[1] = fps[1]*sample_frame+1

        if offset is not None:
            delay = offset*np.floor(fs)/1000.
            frames = frames.astype(float)+delay

        if len(np.shape(data)) == 1:
            time = (np.arange(np.diff(frames))+frames[0])/np.floor(fs)
            return time, data[int(frames[0]):int(frames[1])]
        else:
            time = np.arange(len(data[0, :]))/np.floor(fs)
            time = time[int(frames[0]):int(frames[1])]
            return  time, data[:,int(frames[0]):int(frames[1])]

    def coord_int(self, coord1, coord2, time_acs, time_det):
        """
        Not used
        Interpolates the coordinates values to compensate for the smaller frequency sampling

        Parameters
        ----------

        Returns
        -------
        """    

        coord1_int = interp1d(time_acs, coord1, kind='linear')
        coord2_int = interp1d(time_acs, coord2, kind= 'linear')

        return coord1_int(time_det), coord2_int(time_det)

    def sync_data(self, telemetry=False):
        """
        #Needs to be modify !
        
        Wrapper for the previous functions to return the slices of the detector and coordinates TODs,  
        and the associated time

        Parameters
        ----------

        Returns
        -------
        """    

        num = self.numframes+self.bufferframe*2
        first_frame = self.startframe+self.bufferframe
        print(num,first_frame)

        ctime_mcp = data_value.loaddata(self.det_path, 'time', first_frame=first_frame, num_frames=num) 
        ctime_mcp += self.offset/1000.

        ctime_start = ctime_mcp[0]
        ctime_end = ctime_mcp[-1]

        coord1 = self.coord1_data 
        coord2 = self.coord2_data

        if self.xystage is True:
            freq_array = np.append(0, np.cumsum(np.repeat(1/self.coord_sample_frame, self.coord_sample_frame*num-1)))
            coord1time = ctime_start+freq_array
            coord2time = coord1time.copy()
        else:
            coord1time = ctime_mcp.copy()
            coord2time = ctime_mcp.copy()


        coord1int = interp1d(coord1time, coord1, kind='linear')
        coord2int = interp1d(coord2time, coord2, kind= 'linear')
        dettime = ctime_mcp

        coord1_inter_list = coord1int(dettime)
        coord2_inter_list = coord2int(dettime)

        #plt.plot(ctime_mcp, coord1_inter_list)
        #plt.plot(ctime_mcp, coord1)

        if telemetry:
            #---------------------------------------------------------------------------------------------------------------
            #Needs to be modify
            kidutils = det.kidsutils()
            frames = np.array([first_frame, first_frame + num], dtype='int')
            dettime, pps_bins = kidutils.det_time(self.roach_pps_path, self.roach_number, frames, \
                                                    ctime_start, ctime_mcp[-1], self.det_fs)

            idx_roach_start = np.argmin(np.abs(dettime - ctime_start), axis=1)
            idx_roach_end =   np.argmin(np.abs(dettime - ctime_end),   axis=1)
            data_resampled = []

            for i in range(len(self.det_data)): 
                #self.det_data[i] = kidutils.interpolation_roach(self.det_data[i], pps_bins[i][pps_bins[i]>350], self.det_fs)
                a = kidutils.interpolation_roach(self.det_data[i], pps_bins[i][pps_bins[i]>350], self.det_fs)
                b = a[idx_roach_start[0]:idx_roach_end[0]]
                data_resampled.append(b)

            self.det_data = np.asarray(data_resampled)
            dettime = dettime[:,idx_roach_start[0]:idx_roach_end[0]]
            #---------------------------------------------------------------------------------------------------------------

        if self.lat_data and self.lat_data:
            lst = self.lst_data[self.bufferframe*self.coord_sample_frame:self.bufferframe*self.coord_sample_frame+\
                                num*self.coord_sample_frame]
            lat = self.lat_data[self.bufferframe*self.coord_sample_frame:self.bufferframe*self.coord_sample_frame+\
                                num*self.coord_sample_frame]
            lsttime = ctime_mcp.copy()
            lattime = ctime_mcp.copy()
            lstint = interp1d(lsttime, lst, kind='linear')
            latint = interp1d(lattime, lat, kind= 'linear')
            lst_inter = lstint(dettime)
            lat_inter = latint(dettime)
            del lst
            del lat
            #if np.size(np.shape(self.det_data)) > 1:
            #    return (dettime, self.det_data, coord1_inter, coord2_inter, hwp_inter, lst_inter, lat_inter)
            #else:
            return       (dettime, self.det_data, coord1_inter_list, coord2_inter_list,  lst_inter, lat_inter)
        else:     return (dettime, self.det_data, coord1_inter_list, coord2_inter_list, None, None)

class xsc_offset():
    """
    #Needs to be modify !
    class to read star camera offset files
    Parameters
    ----------
    Returns
    -------
    """    


    def __init__(self, xsc, frame1, frame2):

        self.xsc = xsc #Star Camera number
        self.frame1 = frame1 #Starting frame
        self.frame2 = frame2 #Ending frame

    def read_file(self):

        '''
        Function to read a star camera offset file and return the coordinates 
        offset
        '''

        path = os.getcwd()+'/xsc_'+str(int(self.xsc))+'.txt'

        xsc_file = np.loadtxt(path, skiprows = 2)

        index, = np.where((xsc_file[0]>=float(self.frame1)) & (xsc_file[1]<float(self.frame2)))

        if np.size(index) > 1:
            index = index[0]

        return xsc_file[2], xsc_file[3]

class det_table():

    '''
    Class to read detector tables.
    '''

    def __init__(self, dets, pathtable):

        self.name = dets
        self.pathtable = pathtable

    def loadtable(self):

        det_off = np.zeros((np.size(self.name), 2))
        noise = np.ones(np.size(self.name))
        resp = np.zeros(np.size(self.name))

        path = self.pathtable
        btable = tb.Table.read(path, format='ascii.tab')

        for i, kid in enumerate(self.name):

            index, = np.where(btable['Name'] == kid)
            det_off[i, 1] = btable['EL'][index] 
            det_off[i, 0] = btable['XEL'][index] 

            noise[i] = btable['WhiteNoise'][index]
            resp[i] = btable['Resp.'][index]*-1.


        return det_off, noise, resp

 