#import pygetdata as gd
import numpy as np
from scipy.interpolate import interp1d
import os
import astropy.table as tb
from IPython import embed
import src.detector as det 
import h5py
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.signal import resample_poly
from fractions import Fraction
from scipy.signal import resample

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
                 coord2_name, startframe, numframes, DT, IT, telemetry=False):

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
        self.DT=DT #Float precision required 
        self.IT=IT #Int precision required 

        if self.startframe < 100:
            self.bufferframe = int(0)  #Buffer frames to be loaded before and after the starting and ending frame
        else:
            self.bufferframe = int(100)

        self.telemetry = telemetry
 
    def loadspf(file, field, DT):
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
        return DT(spf)
    
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
    
    def loaddata(file, field, DT, num_frames=None, first_frame=None):
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
            data = f['data'][int(first_frame*spf):int((first_frame+num_frames)*spf)]
        else: 
            data = f['data'][()]
        H.close()
        return data.astype(DT, copy=False)

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
            det_data.append(data_value.loaddata(self.det_path, f'kid_{kid}_roach', self.DT, num, first_frame) ) #kidutils.KIDmag(I_data, Q_data))
            # Assume all the data have the same spf       

        spf_data = data_value.loadspf(self.det_path, f'kid_{kid}_roach', self.DT)
        acqfreq_data = data_value.load_acquisition_frequency(self.det_path, f'kid_{kid}_roach')
        #---------------------------------------------------------------------------------

        coord2_data = data_value.loaddata(self.det_path, f'{self.coord2_name}', self.DT, num, first_frame) 
        if self.coord1_name.lower() == 'xel': 
            coord1_data = data_value.loaddata(self.det_path, 'EL', self.DT, num, first_frame) 
            coord1_data *= np.cos(np.radians(coord2_data)) 
        else: coord1_data = data_value.loaddata(self.det_path, f'{self.coord1_name}', self.DT, num, first_frame) 

        spf_coord = data_value.loadspf(self.det_path, self.coord2_name, self.DT)
        acqfreq_coord = data_value.load_acquisition_frequency(self.det_path, self.coord2_name, )

        #---------------------------------------------------------------------------------
        lat = data_value.loaddata(self.det_path, 'lat',self.DT, num, first_frame)
        lst = data_value.loaddata(self.det_path, 'lst',self.DT, num, first_frame)
        lst_lat_spf = data_value.loadspf(self.det_path, 'lst',self.DT)
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
                 DT, IT, offset = None, roach_number= None, roach_pps_path= None, \
                 hwp_sample_frame=None, xystage=False):

        self.det_path = det_path                                 #Path of the detector dirfile
        self.det_data = det_data                             #Detector data timestream
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
        self.DT = DT
        self.IT = IT
        if roach_number is not None:
            self.roach_number = int(float(roach_number))         #If BLAST-TNG is the experiment, this gives the number of the roach used to read the detector
        else:
            self.roach_number = roach_number
        self.roach_pps_path = roach_pps_path                     #Pulse per second of the roach used to sync the data
        self.offset = offset                                     #Time offset between detector data and coordinates
        self.DT=DT #Float precision required 
        self.IT=IT #Int precision required 

    def coord_int(self, coord1, coord2, time_acs, time_det):

        '''
        Interpolates the coordinates values to compensate for the smaller frequency sampling
                Parameters
        ----------
        coord1: array
            Coordinates 1 to be interpolated. 
        coord2: array
            Coordinates 2 to be interpolated. 
        time_acs: array
            Timesteamps of the coordinates
        time_det: int
            Data timestamps to interpolate the coordinates to. 

        Returns
        -------
        coord1_int: array
            the coordinates 1 interpolated.
        coord2_int: array
            the coordinates 2 interpolated.
        '''

        coord1_int = interp1d(time_acs, coord1, kind='linear')
        coord2_int = interp1d(time_acs, coord2, kind= 'linear')

        return coord1_int(time_det), coord2_int(time_det)
    
    def resampling(self, X, spf_start, spf_end, DT):

        '''
        Interpolates an array with a sample per frame to a different sample per frame 
        Parameters
        ----------
        X: array
            The arrray to be interpolated. 
        spf_start: int
            the sample per frame of X
        spf_end: int
            the final sample per frame wanted. 
        Returns
        -------
        x: array
            the array with the new sample per frame. 
        '''
        """
        ratio = spf_start / spf_end
        interper = PchipInterpolator(np.arange(0, len(X)), X)
        x = interper(np.arange(0, len(X), ratio)) # t -= t[0]
        """

        # Ensure X is in DT
        X = np.array(X, dtype=DT)

        # Compute ratio in DT
        ratio = DT(spf_start) / DT(spf_end)

        # Create interpolator
        #if(DT == np.float16): embed()
        interper = PchipInterpolator(np.arange(0, len(X), dtype=DT), X)

        # New sample points
        new_points = np.arange(0, len(X), ratio, dtype=DT)

        # Interpolated values
        x = interper(new_points)

        return x.astype(DT)

    def downsizing(self, X, pps, spf_end, DT, IT): 
        
        # prepare output lists
        downsampled_data = []
        downsampled_pps = []

        # loop over each second (unique PPS value)
        for sec in np.unique(pps):
            # select data for this second
            mask = (pps == sec)
            data_sec = X[mask]

            # select evenly spaced indices
            indices = np.linspace(0, len(data_sec) - 1, spf_end, dtype=int)
            
            # keep only those samples
            downsampled_data.append(data_sec[indices])

        # concatenate back into single arrays
        downsampled_data = np.concatenate(downsampled_data)

        return downsampled_data
    
    def downsizing_scipy(self, X, pps, spf_end, DT, IT): 
        
        # Find unique seconds
        unique_seconds = np.unique(pps)

        # Number of different seconds
        num_seconds = len(unique_seconds)


        downsampled_data = resample(X, num=spf_end*num_seconds) 

        return downsampled_data
    
    def downsizing_scipy_poly(self, X, spf_start, spf_end, DT, IT): 

        # get integer up/down ratio using Fraction
        frac = Fraction(spf_end, int(spf_start)).limit_denominator()
        up, down = frac.numerator, frac.denominator

        # polyphase resample
        downsampled_data = resample_poly(X, up, down)

        return downsampled_data

    def sync_data(self, freq_target = 100, telemetry=True):

        '''
        Wrapper for the previous functions to return the slices of the detector and coordinates TODs,  
        and the associated time. The turnarounda are also excluded from the TODs.

        Parameters
        ----------
        freq_target: int
            the sampling frequency to which the coordinates, timestamps and coordinates will be reprojected to.  
        Returns
        -------
        dettime: array
            Data timestamps
        det_data: list
            list of detector TODs
        coord1_data: array
            Coordinate 1 TOD
        coord2_data: array
            Coordinate 2 TOD
        lst_data: array
            Local Sideral Time TOD
        lat_data: array
            latitude TOD.
        '''

        print('IN SYNC')
        #--------------------------------------------------------------
        #Load the timestamps and pulse per second of the data. 
        spf_time = data_value.loadspf(self.det_path, f'data_time', self.DT)
        pps = data_value.loaddata(self.det_path, f'data_pps', self.IT, self.numframes, self.startframe) 
        #pps -= pps.min()

        subsec = data_value.loaddata(self.det_path, f'data_subsecond_ps',self.DT, self.numframes, self.startframe) 
        dettime = pps.astype(self.DT)+subsec
        #dettime -= dettime.min()
        #--------------------------------------------------------------

        #---------------------------------------------------------------
        #Cut the edge frames that dont have all their samples, and put the right number of samples in the other frames
        #bn = np.bincount(pps) #<-- to change bc faster 
        _, bn = np.unique(pps, return_counts=True)
        pps_bins = bn[bn>0]

        if pps_bins[0] < spf_time:
            pps = pps[pps_bins[0]:]
            dettime = dettime[pps_bins[0]:]
            for i in range(len(self.det_data)):
                self.det_data[i] = self.det_data[i][pps_bins[0]:]

        if pps_bins[-1] < spf_time:
            pps = pps[:-pps_bins[-1]]
            dettime = dettime[:-pps_bins[-1]]
            for i in range(len(self.det_data)):
                self.det_data[i] = self.det_data[i][:-pps_bins[-1]]
        
        #bn = np.bincount(pps) 
        #_, bn = np.unique(pps, return_counts=True)
        #pps_bins = bn[bn>0]

        #kidutils = det.kidsutils()
        for i in range(len(self.det_data)):
            #self.det_data[i] = kidutils.interpolation_roach(self.det_data[i], pps_bins, self.det_fs, self.DT, self.IT)
            self.det_data[i] = self.resampling(self.det_data[i], spf_time, freq_target, self.DT)
        #dettime = kidutils.interpolation_roach(dettime, pps_bins, self.det_fs, self.DT, self.IT)
        dettime = self.resampling(dettime, spf_time, freq_target, self.DT)
        #---------------------------------------------------------------

        #---------------------------------------------------------------
        # Load the timestamps associated with the coordinates, latitude and lst. 
        # PPS is the pulse per second (pps). It indicates when, more precisely at which second, an element has been recorded. 
        pps = data_value.loaddata(self.det_path, f'coords_pps',self.IT, self.numframes, self.startframe) 
        #pps -= pps.min()
        subsec = data_value.loaddata(self.det_path, f'coords_subsecond_ps',self.DT, self.numframes, self.startframe) 
        ctime  = pps.astype(self.DT)+subsec
        turnaround_flags = data_value.loaddata(self.det_path, f'turnaround_flags', self.DT, self.numframes, self.startframe) 
        spf_ctime        = data_value.loadspf(self.det_path,  f'coords_time', self.DT)
        #---------------------------------------------------------------

        #---------------------------------------------------------------
        #Because the acquisition frequency is not an integer, some frames have more or fewer samples than others. 
        #First, we cut the edge frames that dont have all their samples
        #---------------------
        _, bn = np.unique(pps, return_counts=True)
        pps_bins = bn[bn>0]
        if pps_bins[0] < self.coord_sample_frame:
            pps = pps[pps_bins[0]:]
            ctime = ctime[pps_bins[0]:]
            self.coord1_data = self.coord1_data[pps_bins[0]:]
            self.coord2_data = self.coord2_data[pps_bins[0]:]
            self.lat_data = self.lat_data[pps_bins[0]:]
            self.lst_data = self.lst_data[pps_bins[0]:]
            turnaround_flags = turnaround_flags[pps_bins[0]:]
        if pps_bins[-1] < self.coord_sample_frame:
            pps = pps[:-pps_bins[-1]]
            ctime = ctime[:-pps_bins[-1]]
            self.coord1_data = self.coord1_data[:-pps_bins[-1]]
            self.coord2_data = self.coord2_data[:-pps_bins[-1]]
            self.lat_data = self.lat_data[:-pps_bins[-1]] 
            self.lst_data = self.lst_data[:-pps_bins[-1]]
            turnaround_flags = turnaround_flags[:-pps_bins[-1]]
        #---------------------

        #---------------------
        #_, bn = np.unique(pps, return_counts=True)
        #pps_bins = bn[bn>0]
        #---------------------

        #--------------------------------------------------------------
        # Resample the coordinates and their timestamps from their acquisition frequency to the freq_target.
        ctime= self.resampling(ctime, spf_ctime, freq_target, self.DT)
        self.coord1_data = self.resampling(self.coord1_data, spf_ctime, freq_target, self.DT)
        self.coord2_data = self.resampling(self.coord2_data, spf_ctime, freq_target, self.DT)
        self.lst_data = self.resampling(self.lst_data, spf_ctime, freq_target, self.DT)
        self.lat_data = self.resampling(self.lat_data, spf_ctime, freq_target, self.DT)
        turnaround_flags = np.round(self.resampling(turnaround_flags, spf_ctime, freq_target, self.DT)).astype(self.IT)
        #---------------------------------------------------------------


        #---------------------------------------------------------------
        #Get the data samples whose timestamps are shared with the coordinates timestamps
        ctime_start = ctime[0]
        ctime_end = ctime[-1]
        idx_roach_start, = np.where(np.abs(dettime-ctime_start) == np.amin(np.abs(dettime-ctime_start)))
        idx_roach_end, = np.where(np.abs(dettime-ctime_end) == np.amin(np.abs(dettime-ctime_end)))   
        #---------------------------------------------------------------

        #---------------------------------------------------------------
        #Keep only the previous samples
        for i in range(len(self.det_data)):
            self.det_data[i] = self.det_data[i][idx_roach_start[0]:idx_roach_end[0]]
        dettime = dettime[idx_roach_start[0]:idx_roach_end[0]]
        #---------------------------------------------------------------


        #---------------------------------------------------------------
        #Match the number of coordinates samples (coord1, coord2, lat, lst and the turnaround flags) to data samples.
        self.coord1_data, self.coord2_data = self.coord_int(self.coord1_data, self.coord2_data, ctime, dettime)
        self.lst_data, self.lat_data       = self.coord_int(self.lst_data, self.lat_data, ctime, dettime)
        f = interp1d(ctime, turnaround_flags, kind='linear')
        turnaround_flags_interp = np.round(f(dettime)).astype(self.IT)
        #---------------------------------------------------------------

        #---------------------------------------------------------------
        #Filter out the turnarounds
        for i in range(len(self.det_data)):
            self.det_data[i] = self.det_data[i][turnaround_flags_interp==1]
        dettime = dettime[turnaround_flags_interp==1]
        self.lst_data = self.lst_data[turnaround_flags_interp==1]
        self.lat_data = self.lat_data[turnaround_flags_interp==1]
        self.coord2_data = self.coord2_data[turnaround_flags_interp==1]
        self.coord1_data = self.coord1_data[turnaround_flags_interp==1] 
        #---------------------------------------------------------------
        return dettime, self.det_data, self.coord1_data, self.coord2_data, self.lst_data, self.lat_data
    
    def sync_data_v2(self, freq_target = 100, telemetry=True):

        '''
        Wrapper for the previous functions to return the slices of the detector and coordinates TODs,  
        and the associated time. The turnarounda are also excluded from the TODs.

        Parameters
        ----------
        freq_target: int
            the sampling frequency to which the coordinates, timestamps and coordinates will be reprojected to.  
        Returns
        -------
        dettime: array
            Data timestamps
        det_data: list
            list of detector TODs
        coord1_data: array
            Coordinate 1 TOD
        coord2_data: array
            Coordinate 2 TOD
        lst_data: array
            Local Sideral Time TOD
        lat_data: array
            latitude TOD.
        '''

        #--------------------------------------------------------------
        #Load the timestamps and pulse per second of the data. 
        spf_time = data_value.loadspf(self.det_path, f'data_time', self.DT)
        pps = data_value.loaddata(self.det_path, f'data_pps', self.IT, self.numframes, self.startframe) 
        pps -= pps.min()

        subsec = data_value.loaddata(self.det_path, f'data_subsecond_ps',self.DT, self.numframes, self.startframe) 
        dettime = pps.astype(self.DT)+subsec
        dettime -= dettime.min()
        #--------------------------------------------------------------

        #---------------------------------------------------------------
        #Cut the edge frames that dont have all their samples, and put the right number of samples in the other frames
        bn = np.bincount(pps)
        pps_bins = bn[bn>0]

        if pps_bins[0] < spf_time:
            pps = pps[pps_bins[0]:]
            dettime = dettime[pps_bins[0]:]
            for i in range(len(self.det_data)):
                self.det_data[i] = self.det_data[i][pps_bins[0]:]

        if pps_bins[-1] < spf_time:
            pps = pps[:-pps_bins[-1]]
            dettime = dettime[:-pps_bins[-1]]
            for i in range(len(self.det_data)):
                self.det_data[i] = self.det_data[i][:-pps_bins[-1]]
        
        bn = np.bincount(pps)
        pps_bins = bn[bn>0]

        kidutils = det.kidsutils()

        for i in range(len(self.det_data)):
            self.det_data[i]  = self.downsizing(self.det_data[i], pps, freq_target, self.DT, self.IT)
            #self.det_data[i] = kidutils.interpolation_roach(self.det_data[i], pps_bins, self.det_fs, self.DT, self.IT)
            #self.det_data[i] = self.resampling(self.det_data[i], spf_time, freq_target, self.DT)

        #dettime = kidutils.interpolation_roach(dettime, pps_bins, self.det_fs, self.DT, self.IT)
        #dettime = self.resampling(dettime, spf_time, freq_target, self.DT)
        dettime = self.downsizing(dettime, pps, freq_target, self.DT, self.IT)
        #---------------------------------------------------------------

        #---------------------------------------------------------------
        # Load the timestamps associated with the coordinates, latitude and lst. 
        pps = data_value.loaddata(self.det_path, f'coords_pps',self.IT, self.numframes, self.startframe) 
        pps -= pps.min()
        subsec = data_value.loaddata(self.det_path, f'coords_subsecond_ps',self.DT, self.numframes, self.startframe) 
        ctime  = pps.astype(self.DT)+subsec
        turnaround_flags = data_value.loaddata(self.det_path, f'turnaround_flags', self.DT, self.numframes, self.startframe) 
        spf_ctime        = data_value.loadspf(self.det_path,  f'coords_time', self.DT)
        # Load the pulse per second (pps). It indicates when, more precisely at which second, an element has been recorded. 
        #---------------------------------------------------------------

        #---------------------------------------------------------------
        #Because the acquisition frequency is not an integer, some frames have more or fewer samples than others. 
        #First, we cut the edge frames that dont have all their samples
        #---------------------
        bn = np.bincount(pps)
        pps_bins = bn[bn>0]
        if pps_bins[0] < self.coord_sample_frame:
            pps = pps[pps_bins[0]:]
            ctime = ctime[pps_bins[0]:]
            self.coord1_data = self.coord1_data[pps_bins[0]:]
            self.coord2_data = self.coord2_data[pps_bins[0]:]
            self.lat_data = self.lat_data[pps_bins[0]:]
            self.lst_data = self.lst_data[pps_bins[0]:]
            turnaround_flags = turnaround_flags[pps_bins[0]:]
        if pps_bins[-1] < self.coord_sample_frame:
            pps = pps[:-pps_bins[-1]]
            ctime = ctime[:-pps_bins[-1]]
            self.coord1_data = self.coord1_data[:-pps_bins[-1]]
            self.coord2_data = self.coord2_data[:-pps_bins[-1]]
            self.lat_data = self.lat_data[:-pps_bins[-1]] 
            self.lst_data = self.lst_data[:-pps_bins[-1]]
            turnaround_flags = turnaround_flags[:-pps_bins[-1]]
        #---------------------

        #---------------------
        
        #2nd, we put the right number of samples in the frames.  
        bn = np.bincount(pps)
        pps_bins = bn[bn>0]
        
        # Resample the coordinates and their timestamps from their acquisition frequency to the freq_target.
        ctime= self.downsizing(ctime, pps, freq_target, self.DT, self.IT)
        self.coord1_data = self.downsizing(self.coord1_data,pps, freq_target, self.DT, self.IT)
        self.coord2_data = self.downsizing(self.coord2_data,pps, freq_target, self.DT, self.IT)
        self.lst_data = self.downsizing(self.lst_data,pps, freq_target, self.DT, self.IT)
        self.lat_data = self.downsizing(self.lat_data,pps, freq_target, self.DT, self.IT)
        turnaround_flags_interp =  np.round(self.downsizing(turnaround_flags,pps, freq_target, self.DT, self.IT)).astype(bool)
        #---------------------------------------------------------------

        #---------------------------------------------------------------
        #Get the data samples whose timestamps are shared with the coordinates timestamps
        #---------------------------------------------------------------
        #Filter out the turnarounds
        for i in range(len(self.det_data)):
            self.det_data[i] = self.det_data[i][turnaround_flags_interp==1]
        dettime = dettime[turnaround_flags_interp==1]
        self.lst_data = self.lst_data[turnaround_flags_interp==1]
        self.lat_data = self.lat_data[turnaround_flags_interp==1]
        self.coord2_data = self.coord2_data[turnaround_flags_interp==1]
        self.coord1_data = self.coord1_data[turnaround_flags_interp==1] 
        #---------------------------------------------------------------
        
        return dettime, self.det_data, self.coord1_data, self.coord2_data, self.lst_data, self.lat_data
    
    def sync_data_v3(self, freq_target = 100, telemetry=True):

        #--------------------------------------------------------------
        #Load the timestamps and pulse per second of the data. 
        spf_time = data_value.loadspf(self.det_path, f'data_time', self.DT)
        pps = data_value.loaddata(self.det_path, f'data_pps', self.IT, self.numframes, self.startframe) 
        pps -= pps.min()

        subsec = data_value.loaddata(self.det_path, f'data_subsecond_ps',self.DT, self.numframes, self.startframe) 
        dettime = pps.astype(self.DT)+subsec
        dettime -= dettime.min()
        #--------------------------------------------------------------

        #---------------------------------------------------------------
        #Cut the edge frames that dont have all their samples, and put the right number of samples in the other frames
        bn = np.bincount(pps)
        pps_bins = bn[bn>0]

        if pps_bins[0] < spf_time:
            pps = pps[pps_bins[0]:]
            dettime = dettime[pps_bins[0]:]
            for i in range(len(self.det_data)):
                self.det_data[i] = self.det_data[i][pps_bins[0]:]

        if pps_bins[-1] < spf_time:
            pps = pps[:-pps_bins[-1]]
            dettime = dettime[:-pps_bins[-1]]
            for i in range(len(self.det_data)):
                self.det_data[i] = self.det_data[i][:-pps_bins[-1]]
        
        bn = np.bincount(pps)
        pps_bins = bn[bn>0]

        kidutils = det.kidsutils()

        for i in range(len(self.det_data)):
            self.det_data[i]  = self.downsizing_scipy(self.det_data[i], pps, freq_target, self.DT, self.IT)
        dettime = self.downsizing_scipy(dettime, pps, freq_target, self.DT, self.IT)
        #---------------------------------------------------------------

        #---------------------------------------------------------------
        # Load the timestamps associated with the coordinates, latitude and lst. 
        pps = data_value.loaddata(self.det_path, f'coords_pps',self.IT, self.numframes, self.startframe) 
        pps -= pps.min()
        subsec = data_value.loaddata(self.det_path, f'coords_subsecond_ps',self.DT, self.numframes, self.startframe) 
        ctime  = pps.astype(self.DT)+subsec
        turnaround_flags = data_value.loaddata(self.det_path, f'turnaround_flags', self.DT, self.numframes, self.startframe) 
        spf_ctime        = data_value.loadspf(self.det_path,  f'coords_time', self.DT)
        # Load the pulse per second (pps). It indicates when, more precisely at which second, an element has been recorded. 
        #---------------------------------------------------------------

        #---------------------------------------------------------------
        #Because the acquisition frequency is not an integer, some frames have more or fewer samples than others. 
        #First, we cut the edge frames that dont have all their samples
        #---------------------
        bn = np.bincount(pps)
        pps_bins = bn[bn>0]
        if pps_bins[0] < self.coord_sample_frame:
            pps = pps[pps_bins[0]:]
            ctime = ctime[pps_bins[0]:]
            self.coord1_data = self.coord1_data[pps_bins[0]:]
            self.coord2_data = self.coord2_data[pps_bins[0]:]
            self.lat_data = self.lat_data[pps_bins[0]:]
            self.lst_data = self.lst_data[pps_bins[0]:]
            turnaround_flags = turnaround_flags[pps_bins[0]:]
        if pps_bins[-1] < self.coord_sample_frame:
            pps = pps[:-pps_bins[-1]]
            ctime = ctime[:-pps_bins[-1]]
            self.coord1_data = self.coord1_data[:-pps_bins[-1]]
            self.coord2_data = self.coord2_data[:-pps_bins[-1]]
            self.lat_data = self.lat_data[:-pps_bins[-1]] 
            self.lst_data = self.lst_data[:-pps_bins[-1]]
            turnaround_flags = turnaround_flags[:-pps_bins[-1]]
        #---------------------

        #---------------------
        
        #2nd, we put the right number of samples in the frames.  
        bn = np.bincount(pps)
        pps_bins = bn[bn>0]

        # Resample the coordinates and their timestamps from their acquisition frequency to the freq_target.
        ctime= self.downsizing_scipy(ctime, pps, freq_target, self.DT, self.IT)
        self.coord1_data = self.downsizing_scipy(self.coord1_data,pps, freq_target, self.DT, self.IT)
        self.coord2_data = self.downsizing_scipy(self.coord2_data,pps, freq_target, self.DT, self.IT)
        self.lst_data = self.downsizing_scipy(self.lst_data,pps, freq_target, self.DT, self.IT)
        self.lat_data = self.downsizing_scipy(self.lat_data,pps, freq_target, self.DT, self.IT)
        turnaround_flags_interp =  np.round(self.downsizing_scipy(turnaround_flags,pps, freq_target, self.DT, self.IT)).astype(bool)
        #---------------------------------------------------------------

        #---------------------------------------------------------------
        #Filter out the turnarounds
        for i in range(len(self.det_data)):
            self.det_data[i] = self.det_data[i][turnaround_flags_interp==1]
        dettime = dettime[turnaround_flags_interp==1]
        self.lst_data = self.lst_data[turnaround_flags_interp==1]
        self.lat_data = self.lat_data[turnaround_flags_interp==1]
        self.coord2_data = self.coord2_data[turnaround_flags_interp==1]
        self.coord1_data = self.coord1_data[turnaround_flags_interp==1] 
        #---------------------------------------------------------------
        
        return dettime, self.det_data, self.coord1_data, self.coord2_data, self.lst_data, self.lat_data
    
    def sync_data_v4(self, freq_target = 100, telemetry=True):

        #--------------------------------------------------------------
        #Load the timestamps and pulse per second of the data. 
        spf_time = data_value.loadspf(self.det_path, f'data_time', self.DT)
        pps = data_value.loaddata(self.det_path, f'data_pps', self.IT, self.numframes, self.startframe) 
        pps -= pps.min()

        subsec = data_value.loaddata(self.det_path, f'data_subsecond_ps',self.DT, self.numframes, self.startframe) 
        dettime = pps.astype(self.DT)+subsec
        dettime -= dettime.min()
        #--------------------------------------------------------------

        #---------------------------------------------------------------
        #Cut the edge frames that dont have all their samples, and put the right number of samples in the other frames
        bn = np.bincount(pps)
        pps_bins = bn[bn>0]

        if pps_bins[0] < spf_time:
            pps = pps[pps_bins[0]:]
            dettime = dettime[pps_bins[0]:]
            for i in range(len(self.det_data)):
                self.det_data[i] = self.det_data[i][pps_bins[0]:]

        if pps_bins[-1] < spf_time:
            pps = pps[:-pps_bins[-1]]
            dettime = dettime[:-pps_bins[-1]]
            for i in range(len(self.det_data)):
                self.det_data[i] = self.det_data[i][:-pps_bins[-1]]
        
        bn = np.bincount(pps)
        pps_bins = bn[bn>0]

        kidutils = det.kidsutils()

        for i in range(len(self.det_data)):
            self.det_data[i]  = self.downsizing_scipy_poly(self.det_data[i], spf_time, freq_target, self.DT, self.IT)
        dettime = self.downsizing_scipy_poly(dettime, spf_time, freq_target, self.DT, self.IT)
        #---------------------------------------------------------------

        #---------------------------------------------------------------
        # Load the timestamps associated with the coordinates, latitude and lst. 
        pps = data_value.loaddata(self.det_path, f'coords_pps',self.IT, self.numframes, self.startframe) 
        pps -= pps.min()
        subsec = data_value.loaddata(self.det_path, f'coords_subsecond_ps',self.DT, self.numframes, self.startframe) 
        ctime  = pps.astype(self.DT)+subsec
        turnaround_flags = data_value.loaddata(self.det_path, f'turnaround_flags', self.DT, self.numframes, self.startframe) 
        spf_ctime        = data_value.loadspf(self.det_path,  f'coords_time', self.DT)
        # Load the pulse per second (pps). It indicates when, more precisely at which second, an element has been recorded. 
        #---------------------------------------------------------------

        #---------------------------------------------------------------
        #Because the acquisition frequency is not an integer, some frames have more or fewer samples than others. 
        #First, we cut the edge frames that dont have all their samples
        #---------------------
        bn = np.bincount(pps)
        pps_bins = bn[bn>0]
        if pps_bins[0] < self.coord_sample_frame:
            pps = pps[pps_bins[0]:]
            ctime = ctime[pps_bins[0]:]
            self.coord1_data = self.coord1_data[pps_bins[0]:]
            self.coord2_data = self.coord2_data[pps_bins[0]:]
            self.lat_data = self.lat_data[pps_bins[0]:]
            self.lst_data = self.lst_data[pps_bins[0]:]
            turnaround_flags = turnaround_flags[pps_bins[0]:]
        if pps_bins[-1] < self.coord_sample_frame:
            pps = pps[:-pps_bins[-1]]
            ctime = ctime[:-pps_bins[-1]]
            self.coord1_data = self.coord1_data[:-pps_bins[-1]]
            self.coord2_data = self.coord2_data[:-pps_bins[-1]]
            self.lat_data = self.lat_data[:-pps_bins[-1]] 
            self.lst_data = self.lst_data[:-pps_bins[-1]]
            turnaround_flags = turnaround_flags[:-pps_bins[-1]]
        #---------------------

        #---------------------
        
        #2nd, we put the right number of samples in the frames.  
        bn = np.bincount(pps)
        pps_bins = bn[bn>0]

        # Resample the coordinates and their timestamps from their acquisition frequency to the freq_target.
        ctime= self.downsizing_scipy_poly(ctime,  spf_ctime, freq_target, self.DT, self.IT)
        self.coord1_data = self.downsizing_scipy_poly(self.coord1_data, spf_ctime, freq_target, self.DT, self.IT)
        self.coord2_data = self.downsizing_scipy_poly(self.coord2_data, spf_ctime, freq_target, self.DT, self.IT)
        self.lst_data = self.downsizing_scipy_poly(self.lst_data, spf_ctime, freq_target, self.DT, self.IT)
        self.lat_data = self.downsizing_scipy_poly(self.lat_data, spf_ctime, freq_target, self.DT, self.IT)
        turnaround_flags_interp =  np.round(self.downsizing_scipy_poly(turnaround_flags, spf_ctime, freq_target, self.DT, self.IT)).astype(bool)
        #---------------------------------------------------------------


        #---------------------------------------------------------------
        #Get the data samples whose timestamps are shared with the coordinates timestamps
        ctime_start = ctime[0]
        ctime_end = ctime[-1]
        idx_roach_start, = np.where(np.abs(dettime-ctime_start) == np.amin(np.abs(dettime-ctime_start)))
        idx_roach_end, = np.where(np.abs(dettime-ctime_end) == np.amin(np.abs(dettime-ctime_end)))   
        #---------------------------------------------------------------

        #---------------------------------------------------------------
        #Keep only the previous samples
        for i in range(len(self.det_data)):
            self.det_data[i] = self.det_data[i][idx_roach_start[0]:idx_roach_end[0]]
        dettime = dettime[idx_roach_start[0]:idx_roach_end[0]]
        ctime  = ctime[idx_roach_start[0]:idx_roach_end[0]]

        self.lst_data = self.lst_data[idx_roach_start[0]:idx_roach_end[0]]
        self.lat_data = self.lat_data[idx_roach_start[0]:idx_roach_end[0]]
        self.coord2_data = self.coord2_data[idx_roach_start[0]:idx_roach_end[0]]
        self.coord1_data = self.coord1_data[idx_roach_start[0]:idx_roach_end[0]]
        turnaround_flags_interp = turnaround_flags_interp[idx_roach_start[0]:idx_roach_end[0]]
        #---------------------------------------------------------------

        #---------------------------------------------------------------
        #Filter out the turnarounds
        for i in range(len(self.det_data)):
            self.det_data[i] = self.det_data[i][turnaround_flags_interp==1]
        dettime = dettime[turnaround_flags_interp==1]
        self.lst_data = self.lst_data[turnaround_flags_interp==1]
        self.lat_data = self.lat_data[turnaround_flags_interp==1]
        self.coord2_data = self.coord2_data[turnaround_flags_interp==1]
        self.coord1_data = self.coord1_data[turnaround_flags_interp==1] 
        #---------------------------------------------------------------
        
        return dettime, self.det_data, self.coord1_data, self.coord2_data, self.lst_data, self.lat_data