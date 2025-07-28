#import pygetdata as gd
import numpy as np
from scipy.interpolate import interp1d
import os
import astropy.table as tb
from IPython import embed
import src.detector as det 
import h5py
import matplotlib.pyplot as plt

def load_params(path):
    """
    Return as a dictionary the parameters stores in a .par file
    
    Parameters
    ----------
    path: string
        name of the .par file       
    Returns
    -------
    params: dictionary
        dictionary containing the loaded parameters
    """    
    file = open(path)

    params = {}
    for line in file:
        line = line.strip()
        if not line.startswith("#"):
            no_comment = line.split('#')[0]
            key_value = no_comment.split("=")
            if len(key_value) == 2:
                params[key_value[0].strip()] = key_value[1].strip()

    for key in params.keys():
        params[key] = eval(params[key])

    return params

class data_value():
    
    '''
    Class for reading the values of the TODs (detectors and coordinates) from a DIRFILE
    Parameters
    ----------
    Returns
    -------
    '''

    def __init__(self, det_path, det_name, coord1_name, \
                 coord2_name, startframe, numframes, telemetry=False):

        """
        Class to load the data from a .hdf5 
        Parameters
        ----------
        det_path: string
            Path of the .hdf5
        det_name: string
            Detector names to be analyzed
        coord1_name: string
            Coordinates 1 name, e.g. RA or AZ
        coord2_name: string
            Coordinates 2 name
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
        self.coord1_name = coord1_name              #Coordinates 1 name, e.g. RA or AZ
        self.coord2_name = coord2_name              #Coordinates 2 name
        self.startframe = startframe        #Starting frame to be analyzed
        self.numframes = numframes           #Ending frame to be analyzed

        if self.startframe < 100:
            self.bufferframe = int(0)  #Buffer frames to be loaded before and after the starting and ending frame
        else:
            self.bufferframe = int(100)

        self.telemetry = telemetry
 
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
    
    def load_acquisition_frequency(file, field):
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
        if('acquisition frequency' in f.keys()): spf = f['acquisition frequency'][()]
        else: spf = None
        H.close()
        return spf
    
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

        if os.path.isfile(file): H = h5py.File(file, "a")
        else: print('no file')
        f = H[field]
        if(('spf' in f.keys()) and (num_frames is not None) and (first_frame is not None)):
            spf = f['spf'][()]
            #if(not 'roach' in field): data = f['data'][first_frame*spf:(first_frame+num_frames)*spf]
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
        det_data: list
            list of amplitude timestreams
        coord1_data: array
            Coord 1 timestream
        coord2_data: array
            Coord 2 timestream
        lst: array
            local sideral time timestream
        lat: array
            latitude timestream of a detector
        spf_data: int
            the number of sample per frame of the amplitude timestreams. 
        spf_coord: int
            the number of sample per frame of the coordinates timestreams
        lst_lat_spf: int
            the number of sample per frame of lat and lst. 

        """    
        num = self.numframes+self.bufferframe*2
        first_frame = self.startframe+self.bufferframe
        kid_num  = self.det_name
        det_data = []

        for kid in kid_num: 
            kidutils = det.kidsutils()
            det_data.append(data_value.loaddata(self.det_path, f'kid_{kid}_roach', num, first_frame) ) #kidutils.KIDmag(I_data, Q_data))
            # Assume all the data have the same spf       

        spf_data = data_value.loadspf(self.det_path, f'kid_{kid}_roach')
        acqfreq_data = data_value.load_acquisition_frequency(self.det_path, f'kid_{kid}_roach')
        #---------------------------------------------------------------------------------

        coord2_data = data_value.loaddata(self.det_path, f'{self.coord2_name}', num, first_frame) 
        if self.coord1_name.lower() == 'xel': 
            coord1_data = data_value.loaddata(self.det_path, 'EL', num, first_frame) 
            coord1_data *= np.cos(np.radians(coord2_data)) 
        else: coord1_data = data_value.loaddata(self.det_path, f'{self.coord1_name}', num, first_frame) 

        spf_coord = data_value.loadspf(self.det_path, self.coord2_name, )
        acqfreq_coord = data_value.load_acquisition_frequency(self.det_path, self.coord2_name, )

        #---------------------------------------------------------------------------------
        lat = data_value.loaddata(self.det_path, 'lat',num, first_frame)
        lst = data_value.loaddata(self.det_path, 'lst',num, first_frame)
        lst_lat_spf = data_value.loadspf(self.det_path, 'lst')
        acqfreq_lstlat = data_value.load_acquisition_frequency(self.det_path, 'lst')

        return det_data, coord1_data, coord2_data, lst, lat, spf_data, spf_coord, lst_lat_spf, acqfreq_data, acqfreq_coord, acqfreq_lstlat

class xsc_offset():
    """
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
        Parameters
        ----------
        Returns
        -------
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
    Parameters
    ----------
    Returns
    -------
    '''

    def __init__(self, dets, pathtable):

        self.name = dets
        self.pathtable = pathtable

    def loadtable(self):
        '''
        Function to load the detectors info from the dectector file. 
        Parameters
        ----------
        Returns
        -------
        det_off: list
            list of angular offsets from the center of the array for each detectors. 
        noise: array
            list of detectors white noise.
        resp: array
            list of detectors response. 
        '''

        det_off = np.zeros((np.size(self.name), 2))
        noise = np.ones(np.size(self.name))
        resp = np.zeros(np.size(self.name))

        path = self.pathtable
        btable = tb.Table.read(path, format='ascii.tab')

        for i, kid in enumerate(self.name):

            index, = np.where(btable['Name'] == kid)
            det_off[i, 0] = btable['XEL'][index] 
            det_off[i, 1] = btable['EL'][index] 

            noise[i] = btable['WhiteNoise'][index]
            resp[i] = btable['Resp.'][index]#*-1.


        return det_off, noise, resp

class frame_zoom_sync():

    '''
    This class is designed to extract the frames of interest from the complete timestream and 
    sync detector and coordinates timestream given a different sampling of the two
    '''

    def __init__(self, det_path, det_data,det_fs, det_sample_frame,\
                 coord1_data, coord2_data, coord_fs, coord_sample_frame, \
                 startframe, numframes, lst_data, lat_data, lstlat_fs, lstlat_sample_frame, \
                 offset = None, roach_number= None, roach_pps_path= None, \
                 hwp_sample_frame=None, xystage=False):

        self.det_path = det_path                                 #Path of the detector dirfile
        self.det_data = det_data                                 #Detector data timestream
        self.det_fs = float(det_fs)                              #Detector frequency sampling
        self.det_sample_frame = int(float(det_sample_frame))     #Detector samples in each frame of the timestream
        self.coord1_data = coord1_data                           #Coordinate 1 data timestream
        self.coord_fs = float(coord_fs)                          #Coordinates frequency sampling
        self.coord_sample_frame = int(float(coord_sample_frame)) #Coordinates samples in each frame of the time stream
        self.coord2_data = coord2_data                           #Coordinate 2 data timestream
        self.startframe = int(float(startframe))                 #Start frame
        self.numframes = int(float(numframes))                   #Number of frames
        self.lst_data = lst_data                                 #LST timestream (if correction is required and coordinates are RA-DEC)
        self.lat_data = lat_data                                 #LAT timestream (if correction is required and coordinates are RA-DEC)
        self.lstlatfreq = lstlat_fs                              #LST-LAT sampling frequency (if correction is required and coordinates are RA-DEC)
        self.lstlat_sample_frame = lstlat_sample_frame           #LST-LAT samples per frame (if correction is required and coordinates are RA-DEC)
        if roach_number is not None:
            self.roach_number = int(float(roach_number))         #If BLAST-TNG is the experiment, this gives the number of the roach used to read the detector
        else:
            self.roach_number = roach_number
        self.roach_pps_path = roach_pps_path                     #Pulse per second of the roach used to sync the data
        self.offset = offset                                     #Time offset between detector data and coordinates

    def frame_zoom(self, data, sample_frame, fs, fps, offset = None):

        '''
        Selecting the frames of interest and associate a timestamp for each value.
        '''

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

        '''
        Interpolates the coordinates values to compensate for the smaller frequency sampling
        '''

        coord1_int = interp1d(time_acs, coord1, kind='linear')
        coord2_int = interp1d(time_acs, coord2, kind= 'linear')

        return coord1_int(time_det), coord2_int(time_det)

    def sync_data(self, telemetry=True):

        '''
        Wrapper for the previous functions to return the slices of the detector and coordinates TODs,  
        and the associated time
        '''

        #----------------------------------------
        #Load the timestamps and pulse per second
        #Assume that "data_time" is the data timestamps
        time_data = data_value.loaddata(self.det_path, f'data_time', self.numframes, self.startframe) 
        spf_time_data = data_value.loadspf(self.det_path, f'data_time')
        pps_data = data_value.loaddata(self.det_path, f'data_pps', self.numframes, self.startframe) 
        bn = np.bincount(pps_data)
        pps_bins = bn[bn>0]

        #----------------------------------------

        kidutils = det.kidsutils()
        for i in range(len(self.det_data)):
            #self.det_data[i] = kidutils.interpolation_roach(self.det_data[i], pps_bins, self.det_fs)
            a = self.det_data[i]
            b = kidutils.interpolation_roach(self.det_data[i], pps_bins, self.det_fs)


        #----------------------------------------
        #Load the timestamps
        #Assume that "time" is the coordinates timestamps. 
        ctime = data_value.loaddata(self.det_path, f'coords_time', self.numframes, self.startframe) 
        spf_time = data_value.loadspf(self.det_path, f'coords_time')
        pps = data_value.loaddata(self.det_path, f'coords_pps', self.numframes, self.startframe) 
        bn = np.bincount(pps)
        pps_bins = bn[bn>0]

        embed()
        if pps_bins[0] < spf_time:
            pps = pps[pps_bins[0]:]
            ctime = ctime[pps_bins[0]:]
        if pps_bins[-1] < spf_time:
            pps = pps[:-pps_bins[-1]]
            ctime = ctime[:-pps_bins[-1]]

        pps_duration =  pps[-1]-pps[0]+1
        pps_final =  pps[0]+np.arange(0, pps_duration, 1/self.coord_fs) 
        #----------------------------------------
        
        kidutils = det.kidsutils()
        coord1 = kidutils.interpolation_roach(self.coord1_data, pps_bins, self.coord_fs)
        coord2 = kidutils.interpolation_roach(self.coord2_data, pps_bins, self.coord_fs)
        #-----------------------------------------


        time = data_value.loaddata(self.det_path, f'data_time', self.numframes, self.startframe) 
        spf_time = data_value.loadspf(self.det_path, f'data_time')

        '''
        idx_roach_start, = np.where(np.abs(dettime-ctime_start) == np.amin(np.abs(dettime-ctime_start)))
        idx_roach_end, = np.where(np.abs(dettime-ctime_end) == np.amin(np.abs(dettime-ctime_end)))
        '''

        #ctime_start = ctime_mcp+ctime_usec/1e6+0.2
        #----------------------------------------
       
        #----------------------------------------
        #load the pulses per second (pps)
        '''
        kidutils = det.kidsutils()
        
        start_det_frame = self.startframe-self.bufferframe
        end_det_frame = self.endframe+self.bufferframe

        frames = np.array([start_det_frame, end_det_frame], dtype='int')

        dettime, pps_bins = kidutils.det_time(self.roach_pps_path, self.roach_number, frames, \
                                                ctime_start, ctime_mcp[-1], self.det_fs)
        '''
        pps = data_value.loaddata(self.det_path, f'data_pps', self.numframes, self.startframe) 
        bn = np.bincount(pps)
        pps_bins = bn[bn>0]
        
        #--------------------------

        #interpolate

        coord1int = interp1d(coord1time, coord1, kind='linear')
        coord2int = interp1d(coord2time, coord2, kind= 'linear')

        kidutils = det.kidsutils()
        for i in range(len(self.det_data)):
            self.det_data[i] = kidutils.interpolation_roach(self.det_data[i], pps_bins, self.det_fs)


        coord1_inter, coord2_inter = self.coord_int(coord1, coord2, \
                                                    coord1time, dettime[index1[0]+10:index2[0]-10])


        if self.lat_data is not None and self.lat_data is not None:

            if self.experiment.lower() == 'blastpol':
                lsttime, lst = self.frame_zoom(self.lst_data, self.lstlat_sample_frame, \
                                                self.lstlatfreq, np.array([self.startframe,self.endframe]))

                lattime, lat = self.frame_zoom(self.lat_data, self.lstlat_sample_frame, \
                                                self.lstlatfreq, np.array([self.startframe,self.endframe]))

                lsttime = lsttime-lsttime[0]
                index1, = np.where(np.abs(dettime-lsttime[0]) == np.amin(np.abs(dettime-lsttime[0])))
                index2, = np.where(np.abs(dettime-lsttime[-1]) == np.amin(np.abs(dettime-lsttime[-1])))

                lst_inter, lat_inter = self.coord_int(lst, lat, \
                                                        lsttime, dettime[index1[0]+10:index2[0]-10])

            else:
                lst = self.lst_data[self.bufferframe*self.coord_sample_frame:self.bufferframe*self.coord_sample_frame+\
                                    interval*self.coord_sample_frame]
                lat = self.lat_data[self.bufferframe*self.coord_sample_frame:self.bufferframe*self.coord_sample_frame+\
                                    interval*self.coord_sample_frame]

                lsttime = ctime_mcp.copy()
                lattime = ctime_mcp.copy()

                lstint = interp1d(lsttime, lst, kind='linear')
                latint = interp1d(lattime, lat, kind= 'linear')

                lst_inter = lstint(dettime)
                lat_inter = latint(dettime)

            del lst
            del lat

            return 0