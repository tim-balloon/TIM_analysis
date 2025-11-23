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
import h5py
import os
import shutil

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
    Class for reading the values of the TODs (detectors and coordinates) from a .hdf5 
    Parameters
    ----------
    Returns
    -------
    '''

    def __init__(self, det_path, det_name, coord1_name, \
                 coord2_name, startframe, numframes, DT, IT):

        """
        Init a instance of the class to load the data from a .hdf5 
        Parameters
        ----------
        det_path: string
            Path of the .hdf5
        det_name: list
            list of detector names to be analyzed
        coord1_name: string
            Coordinates 1 name, e.g. RA or AZ
        coord2_name: string
            Coordinates 2 name
        startframe: int
            Starting frame to be analyzed
        numframes: int
            Number of frames to be analyzed
        DT: type
            Float precision required
        IT: type
            Integer precision required
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

    def loadspf(file, field, IT):
        """
        Load the sample per frame of a field from a .hdf5 
        Parameters
        ----------
        file: string
            the name of the .hdf5 file
        field: string
            the field for which to get the spf
        IT: type
            Integer precision required
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
        return IT(spf)
    
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
        DT: type
            Float precision required
        num_frame: int
            the number of frames to load, with N=spf samples in each frame. 
            If number of frames or the first frame is not specified, load the whole timestream. 
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
        Load the coordinates and amplitudes timestreams for a given list of detectors
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

        num = self.numframes#+self.bufferframe
        first_frame = self.startframe+self.bufferframe
        kid_num  = self.det_name
        det_data = []

        for kid in kid_num: 
            kidutils = det.kidsutils()
            det_data.append(data_value.loaddata(self.det_path, f'kid_{kid}_roach', self.DT, num, first_frame) ) #kidutils.KIDmag(I_data, Q_data))
            # Assume all the data have the same spf       

        spf_data = data_value.loadspf(self.det_path, f'kid_{kid}_roach', self.DT)
        #acqfreq_data = data_value.load_acquisition_frequency(self.det_path, f'kid_{kid}_roach')
        #---------------------------------------------------------------------------------

        coord2_data = data_value.loaddata(self.det_path, f'{self.coord2_name}', self.DT, num, first_frame) 
        if self.coord1_name.lower() == 'xel': 
            coord1_data = data_value.loaddata(self.det_path, 'EL', self.DT, num, first_frame) 
            coord1_data *= np.cos(np.radians(coord2_data)) 
        else: coord1_data = data_value.loaddata(self.det_path, f'{self.coord1_name}', self.DT, num, first_frame) 

        spf_coord = data_value.loadspf(self.det_path, self.coord2_name, self.DT)
        #acqfreq_coord = data_value.load_acquisition_frequency(self.det_path, self.coord2_name, )

        #---------------------------------------------------------------------------------
        lat = data_value.loaddata(self.det_path, 'lat',self.DT, num, first_frame)
        lst = data_value.loaddata(self.det_path, 'lst',self.DT, num, first_frame)
        lst_lat_spf = data_value.loadspf(self.det_path, 'lst',self.DT)
        #acqfreq_lstlat = data_value.load_acquisition_frequency(self.det_path, 'lst')

        return det_data, coord1_data, coord2_data, lst, lat, spf_data, spf_coord, lst_lat_spf #, acqfreq_data, acqfreq_coord, acqfreq_lstlat

class xsc_offset():
    """
    class to read star camera offset files
    Parameters
    ----------
    Returns
    -------
    """    
    def __init__(self, xsc, frame1, frame2):
        """
        function to create an instance of the class to read star camera offset files
        Parameters
        xcs: int
            #Star Camera number
        frame1: int
            Starting frame
        frame2: int
            End frame
        ----------
        Returns
        -------
        """  

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

        '''
        function to create an instance of the class to read detector tables.
        Parameters
        ----------
        dets: list
            list of detector names for which to load the boresight offset
        pathtable: str
            path to the file storing the boresight-offset table.
        Returns
        -------
        '''

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

class compress_tods():
    
    '''
    Class to compress timestreams and save them into an .hdf5 file. 
    Parameters
    ----------
    Returns
    -------
    '''

    def __init__(self, tods_path,kid_num, det_data, det_sample_frame, det_timestamps,\
                 coord1, coord2, coord1_data, coord2_data, startframe, numframes, lst_data, lat_data, P,
                 DT, IT, int8=False):
        
        '''
        Class to compress timestreams and save them into an .hdf5 file. 
        Parameters
        tods_path: str
            path and name of the .hdf5 file in which to save the timestreams. 
        kid_num: list
            list of detector names
        det_data: list
            list of cleaned data timestreams, ordered by detectors like in kid_num 
        det_sample_frame: float
            sample frequency of the cleaned data
        det_timestamps: 1d array
            timestamps associated with the cleaned data
        coord1: str
            Coordinate 1 type (RA, AZ...)
        coord2: str
            Coordinate 2 type (DEC, EL...)
        coord1_data: 1d array
            Coordinate 1 timestream
        coord2_data: 1d array    
            Coordinate 2 timestream
        startframe: int
            the first loaded frame
        numframes: int
            the number of loaded frames
        lst_data: 1d array
            Local Sideral Time timestream
        lat_data: 1d array    
            Latitude timestream    
        P: dictionary
            The parameters dictionary that will be saved in the metadata. 
        DT: type
            Float precision required
        IT: type
            Integer precision required
        int8: bool
            If to save data in 1 byte
        ----------
        Returns
        -------
        '''

        self.tods_path = tods_path                               #Path of the timestreams hdf5
        self.kid_num = kid_num                                   #Dectector name list
        self.det_data = det_data                                 #Detector data timestream
        self.det_sample_frame = int(float(det_sample_frame))     #Detector samples in each frame of the timestream
        self.det_timestamps = det_timestamps                     #Detector timestamps
        self.coord1 = coord1                                     #Coordinate 1 name  
        self.coord2 = coord2                                     #Coordinate 2 name
        self.coord1_data = coord1_data                           #Coordinate 1 data timestream                        
        self.coord2_data = coord2_data                           #Coordinate 2 data timestream
        self.startframe = int(float(startframe))                 #Start frame
        self.numframes = int(float(numframes))                   #Number of frames
        self.lst_data = lst_data                                 #LST timestream (if correction is required and coordinates are RA-DEC)
        self.lat_data = lat_data                                 #LAT timestream (if correction is required and coordinates are RA-DEC)
        self.P = P                                               #Parameter dictionary
        self.DT=DT                                               #Float precision required 
        self.IT=IT                                               #Int precision required 
        self.int8 = int8                                         #If to save data in 1 byte

    def save_tods(self):

        '''
        Save the timestreams in an .hdf5 
        Parameters
        ----------
        Returns
        -------
        '''

        # Create the file if it doesn't exist, otherwise open it
        if not os.path.exists(self.tods_path):
            # 'w' creates a new file (overwrites if exists)
            with h5py.File(self.tods_path, "w") as f:
                print(f"Created empty file: {self.tods_path}")
        else:
            # 'a' opens existing file (read/write mode)
            with h5py.File(self.tods_path, "a") as f:
                print(f"Opened existing file: {self.tods_path}")

        data = np.asarray(self.det_data)
        if(self.int8): data, min, max = self.to8bit_intprecision(data)
        else: data, min, max = self.DT(data), None, None
        self.save_array_to_hdf5('TODs', data, self.kid_num, spf=self.det_sample_frame, min=min, max=max)

        coords = np.vstack((self.coord1_data,self.coord2_data, self.lst_data, self.lat_data)).T
        if(self.int8): coords, min, max = self.to8bit_intprecision(coords)
        else: coords, min, max = self.DT(coords), None, None

        self.save_array_to_hdf5('coordinates', coords, (self.coord1,self.coord2,'LST','latitude'), spf=self.det_sample_frame, min=min, max=max)

        self.save_array_to_hdf5('frames',(self.startframe, self.numframes), ('start_frame', 'num_frames'))

        timestamps = self.det_timestamps
        if(self.int8): timestamps, min, max = self.to8bit_intprecision(timestamps)
        else: timestamps, min, max = self.DT(timestamps), None, None
        self.save_array_to_hdf5('timestamps', timestamps, 'synchronized_timestamps', spf=self.det_sample_frame, min=min, max=max)

        #self.save_array_to_hdf5('parameters', 0, self.P)

        return 0
    
    def save_array_to_hdf5(self, grp_name, data, list_names, spf=None, min=None, max=None):

            
        '''
        Save an array in .hdf5
        Parameters
        grp_name: str
            name of the group in which to save the array
        data: 2d array
            array to be saved
        list_names: list
            list that describes the rows of the data array
        spf: int
            sample per frame of the array if applicable
        min: float
            minimum negative value <0  to reconstruct an array saved in 8bits
        max: float
            maximum positive to reconstruct an array saved in 8bits
        ----------
        Returns
        -------
        '''

        temp_filename = self.tods_path + ".tmp"

        try:
            # Step 1 — Copy the existing file to a temporary one
            shutil.copy2(self.tods_path, temp_filename)

            # Step 2 — Open the temporary file in append mode and modify it
            with h5py.File(temp_filename, "a") as H:

                if grp_name not in H: f = H.create_group(grp_name)
                else:                f = H[grp_name]

                if 'data' in f: del f['data']  # deletes group or dataset safely
                f.create_dataset('data', data=data, compression='gzip', compression_opts=9)  # example element

                if 'list_names' in f: del f['list_names']  
                dt = h5py.string_dtype(encoding='utf-8')
                f.create_dataset('list_names', data=np.array(list_names, dtype=dt))

                if 'min' in f: del f['min'] 
                if(min is not None): f.create_dataset('min', data=min) 

                if 'max' in f: del f['max'] 
                if(max is not None): f.create_dataset('max', data=max) 

                if('spf' in f): del f['spf'] 
                if(spf is not None):f.create_dataset('spf', data=spf) 

            # Step 3 — Replace the original only after successful write
            os.replace(temp_filename, self.tods_path)

        except Exception as e:
            print("Error occurred:", e)
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

        return 0
    
    def to8bit_intprecision(array): 

        '''
        function to save an array in 8bits precision. 
        Parameters
        array: ndarray
            array to be downsized
        ----------
        Returns
        -------
        downsampled_array: ndarray
            Array in 8bits precision
        min: float
            the minimum negative value of the array
        max: float
            The maximum value of the array. 
        '''        

        # Rescale data to float range 0.0-1.0

        min = float_array.min()
        if(min<0): float_array -= min
        else: min=None

        max = np.abs(float_array).max()
        float_array /= max

        # 1. Scale the values to the 0-255 range
        scaled_array = float_array * 255

        # 2. Clip the values to ensure they are within the 8-bit range (0-255)
        clipped_array = np.clip(scaled_array, 0, 255)

        # 3. Convert to unsigned 8-bit integer type
        downsampled_array = clipped_array.astype(np.uint8)

        return downsampled_array, np.float16(min), np.float16(max)

class frame_zoom_sync():

    '''
    This class is designed to extract the frames of interest from the complete timestream and 
    sync detector and coordinates timestream given a different sampling of the two
    Parameters
    ----------
    Returns
    -------
    '''

    def __init__(self, det_path, det_data, det_sample_frame,\
                 coord1_data, coord2_data, coord_sample_frame, \
                 startframe, numframes, lst_data, lat_data, lstlat_sample_frame, \
                 DT, IT, freq_target=100):
        
        '''
        Create an instance of the class designed to extract the frames of interest from the complete timestream and 
        sync detector and coordinates timestream given a different sampling of the two
        Parameters
        ----------

        det_path: str
            Path of the detector dirfile
        det_data: list
            Detector data timestream
        det_sample_frame: int
            Number of samples in each frame of the data timestreams
        coord1_data: 1d array
            Coordinate 1 data timestream
        coord2_data: 1d array
            Coordinate 2 data timestream
        coord_sample_frame: int
            Number of samples in each frame of the coordinate timestreams
        startframe1: int
            Start frame
        numframes: int
            Number of frames
        lst_data: 1d array
            LST timestream (if correction is required and coordinates are RA-DEC)
        lat_data: 1d array
            LAT timestream (if correction is required and coordinates are RA-DEC)
        lstlat_sample_frame: int
            LST-LAT samples per frame (if correction is required and coordinates are RA-DEC)
        DT: type
            Float precision required 
        IT: type
            Int precision required 
        freq_target: float
            Frequency to downsample the data to. 
        Returns
        -------
        '''

        self.det_path = det_path                                 #Path of the detector dirfile
        self.det_data = det_data                                 #Detector data timestream
        self.det_sample_frame = int(float(det_sample_frame))     #Detector samples in each frame of the timestream
        self.coord1_data = coord1_data                           #Coordinate 1 data timestream
        self.coord_sample_frame = int(float(coord_sample_frame)) #Coordinates samples in each frame of the time stream
        self.coord2_data = coord2_data                           #Coordinate 2 data timestream
        self.startframe = int(float(startframe))                 #Start frame
        self.numframes = int(float(numframes))                   #Number of frames
        self.lst_data = lst_data                                 #LST timestream (if correction is required and coordinates are RA-DEC)
        self.lat_data = lat_data                                 #LAT timestream (if correction is required and coordinates are RA-DEC)
        self.lstlat_sample_frame = lstlat_sample_frame           #LST-LAT samples per frame (if correction is required and coordinates are RA-DEC)
        self.DT = DT                                             #Float precision required 
        self.IT = IT                                             #Int precision required 
        self.freq_target = freq_target                           #Frequency to downsample the data to. 

        if self.startframe < 100:
            self.bufferframe = int(100)  #Buffer frames to be loaded before and after the starting and ending frame
        else:
            self.bufferframe = int(0)
  
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
        DT: type
            Float precision required
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
        interper = PchipInterpolator(np.arange(0, len(X)), X)

        # New sample points
        new_points = np.arange(0, len(X), ratio)

        # Interpolated values
        x = interper(new_points)

        return x.astype(DT)

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
        print('ok ok ok')
        coord1_int = interp1d(time_acs, coord1, kind='linear',bounds_error=False,fill_value="extrapolate")
        coord2_int = interp1d(time_acs, coord2, kind= 'linear',bounds_error=False,fill_value="extrapolate")

        return coord1_int(time_det), coord2_int(time_det)

    def sync_data(self):

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
    
        num = self.numframes+self.bufferframe
        first_frame = self.startframe+self.bufferframe
        #---------------------------------------------------------------
        # Load the timestamps associated with the coordinates, latitude and lst. 
        # PPS is the pulse per second (pps). It indicates when, more precisely at which second, an element has been recorded. 
        pps = data_value.loaddata(self.det_path, f'coords_pps',self.IT, num, first_frame) 
        #pps -= pps.min()
        subsec = data_value.loaddata(self.det_path, f'coords_subsecond_ps',self.DT, num, first_frame) 
        ctime  = pps.astype(self.DT)+subsec
        turnaround_flags = data_value.loaddata(self.det_path, f'turnaround_flags', self.DT, num, first_frame) 
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
        ctime= self.resampling(ctime, spf_ctime, self.freq_target, self.DT)
        self.coord1_data = self.resampling(self.coord1_data, spf_ctime, self.freq_target, self.DT)
        self.coord2_data = self.resampling(self.coord2_data, spf_ctime, self.freq_target, self.DT)
        self.lst_data = self.resampling(self.lst_data, spf_ctime, self.freq_target, self.DT)
        self.lat_data = self.resampling(self.lat_data, spf_ctime, self.freq_target, self.DT)
        turnaround_flags = np.round(self.resampling(turnaround_flags, spf_ctime, self.freq_target, self.DT)).astype(self.IT)
        #---------------------------------------------------------------

        #--------------------------------------------------------------
        #Load the timestamps and pulse per second of the data. 
        spf_time = data_value.loadspf(self.det_path,  f'data_time', self.DT)
        pps = data_value.loaddata(self.det_path, f'data_pps', self.DT,num, first_frame) 
        subsec = data_value.loaddata(self.det_path, f'data_subsecond_ps',self.DT, num, first_frame) 
        dettime = pps+subsec
        #--------------------------------------------------------------

        #---------------------------------------------------------------
        #Cut the edge frames that dont have all their samples, and put the right number of samples in the other frames
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

        for i in range(len(self.det_data)):
            self.det_data[i] = self.resampling(self.det_data[i], spf_time, self.freq_target, self.DT)
        dettime = self.resampling(dettime, spf_time, self.freq_target, self.DT)
        #---------------------------------------------------------------

        #---------------------------------------------------------------
        #Get the data samples whose timestamps are shared with the coordinates timestamps

        # Determine common time interval (overlap)
        start_time = max(ctime[0], dettime[0])   # latest starting time
        end_time   = min(ctime[-1], dettime[-1]) # earliest ending time

        # Get indices (right for start, right for end)
        i_c_start = np.searchsorted(ctime, start_time, side='left')
        i_c_end   = np.searchsorted(ctime, end_time, side='left')

        i_d_start = np.searchsorted(dettime, start_time, side='left')
        i_d_end   = np.searchsorted(dettime, end_time, side='left')

        # Trim
        dettime = dettime[i_d_start:i_d_end]
        #Keep only the previous samples
        for i in range(len(self.det_data)):
            self.det_data[i] = self.det_data[i][i_d_start:i_d_end]


        ctime   = ctime[i_c_start:i_c_end]
        self.coord1_data = self.coord1_data[i_c_start:i_c_end]
        self.coord2_data = self.coord2_data[i_c_start:i_c_end]
        self.lst_data = self.lst_data[i_c_start:i_c_end]
        self.lat_data = self.lat_data[i_c_start:i_c_end]
        turnaround_flags = turnaround_flags[i_c_start:i_c_end]
        #---------------------------------------------------------------
        #---------------------------------------------------------------
        #Match the number of coordinates samples (coord1, coord2, lat, lst and the turnaround flags) to data samples.
        self.coord1_data, self.coord2_data = self.coord_int(self.coord1_data, self.coord2_data, ctime, dettime)
        self.lst_data, self.lat_data       = self.coord_int(self.lst_data, self.lat_data, ctime, dettime)
        f = interp1d(ctime, turnaround_flags, kind='linear',bounds_error=False,fill_value="extrapolate")
        turnaround_flags_interp = np.round(f(dettime)).astype(self.IT)
        #---------------------------------------------------------------

        return dettime, self.det_data, self.coord1_data, self.coord2_data, self.lst_data, self.lat_data, turnaround_flags_interp
    