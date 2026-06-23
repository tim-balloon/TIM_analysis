#import pygetdata as gd
import numpy as np
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.signal import resample_poly, resample
import os
import astropy.table as tb
from IPython import embed
import src.detector as det 
import h5py
import matplotlib.pyplot as plt
import shutil
import pygetdata as gd
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
                 coord2_name, startframe, numframes, \
                 despike, sigma, prominence, \
                 downsample, freq_target, DT, IT, P=None):

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
        despike: bool
            if True despikes the data using scipy.signal
        sigma: float
            height in std value to look for spikes 
        prominence: float
                prominence in std value to look for spikes
        downsample: bool
            if True downsample the timestreams by decimation
        freq_target: float
            The frequency in Hz to downsample the timestreams to
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
        self.det_name = det_name                    #Detector name to be analyzed kidnum
        self.coord1_name = coord1_name              #Coordinates 1 name, e.g. RA or AZ
        self.coord2_name = coord2_name              #Coordinates 2 name
        self.startframe = startframe                #Starting frame to be analyzed
        self.numframes = numframes                  #Ending frame to be analyzed
        self.DT=DT                                  #Float precision required 
        self.IT=IT                                  #Int precision required 
        self.freq_target = freq_target              #Frequency in Hz to downsample the data to. 
        self.downsample = downsample                #If True, downsample the data
        self.sigma = sigma                          #height in std value to look for spikes
        self.prominence = prominence                #prominence in std value to look for spikes
        self.despike = despike                      #if True despikes the data 
        self.P = P

        if self.startframe < 100:
            self.bufferframe = int(0)  #Buffer frames to be loaded before and after the starting and ending frame
        else:
            self.bufferframe = int(0)

        self.startframe += self.bufferframe

    def conversion_type(self, file_type):

        '''
        Function to define the different datatype conversions strings for pygetdata
        '''

        if file_type == 'u16':
            gdtype = gd.UINT16
        elif file_type == 'u32':
            gdtype = gd.UINT32
        elif file_type == 's32':
            gdtype = gd.INT32
        elif file_type == 'float':
            gdtype = gd.FLOAT32

        return gdtype 

    def loadspf_hdf5(file, field):
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
    
    def loadspf_dirfile(self, file, field):
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

        d = gd.dirfile(file, gd.RDONLY)
        spf = d.spf(field)

        return spf
    
    def load_acquisition_frequency_hdf5(file, field):
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
        freq: float
            the acquisition frequency of the data in Hz
        """    
        H = h5py.File(file, "a")
        f = H[field]
        if('acquisition frequency' in f.keys()): freq = f['acquisition frequency'][()]
        else: freq = None
        H.close()
        return freq
    
    def loaddata_hdf5(file, field, DT, num_frames=None, first_frame=None):
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
            data = f['data'][int(first_frame*spf):int((first_frame+num_frames)*spf)].astype(DT, copy=False)
        else: 
            data = f['data'][()].astype(DT, copy=False)
        H.close()
        return data
    
    def loaddata_dirfile(self, filepath, file, DT, num=None, first_frame=None, file_type=None):

        '''
        Return the values of the DIRFILE as a numpy array
        
        filepath: path of the DIRFILE to be read
        file: name of the value to be read from the dirfile, e.g. detector name or
              coordinate name
        file_type: data type conversion string for the DIRFILE data
        '''
        d = gd.dirfile(filepath, gd.RDONLY)

        if file_type is not None:  gdtype = self.conversion_type(file_type)
        else:                      gdtype = gd.FLOAT64

        if(num is None): num = d.nframes
        if(first_frame is None): first_frame = 0

        values = d.getdata(file, gdtype, num_frames = num, first_frame=first_frame)

        return np.asarray(values).astype(DT, copy=False)

    def values(self):
        """      
        Load the coordinates and amplitudes timestreams for a given list of detectors
        Now, despiking and downsampling of the data happend right after their are loaded. 
        Parameters
        ----------
        Returns
        -------
        dettime: 1d array
            The detector timestamps
        det_data: list
            list of amplitude timestreams
        ctime: 1d array
            The coordinates timestamps
        coord1_data: array
            Coord 1 timestream
        coord2_data: array
            Coord 2 timestream
        turnaround_flags: 1d array
            indicate if the coordinate sample is taken when the telescope speed is not constant
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

        ##################
        """
        kidutils = det.kidsutils()
        det_data = kidutils.KIDmag(I_data, Q_data)`
        """
        ####################

        #-----------------------------------------------------------------------------------------------
        # Load the sample-per-frame of the detector timestreams, assuming they all have the same spf.
        
        if('.hdf5' in self.det_path):
            spf_data = data_value.loadspf_hdf5(self.det_path,  f'data_time')
            #Load the detector timestamps, assuming the detectors all have the same timestamps. 
            #1st, load the pulse per second, which defines to which second each sample belong to. 
            pps = data_value.loaddata_hdf5(self.det_path, f'data_pps', self.DT, self.numframes, self.startframe,) 
            #2nd, load the sub-second part of the timestamps. 
            subsec = data_value.loaddata_hdf5(self.det_path, f'data_subsecond_ps',self.DT, self.numframes, self.startframe)
        else: 
            spf_data = self.loadspf_dirfile(self.det_path,  f'data_time')
            pps      = self.loaddata_dirfile(self.det_path, f'data_pps', self.DT, self.numframes, self.startframe,) 
            subsec   = self.loaddata_dirfile(self.det_path, f'data_subsecond_ps',self.DT, self.numframes, self.startframe)
       
        #Get the final timestamps.
        dettime = pps+subsec
        #-----------------------------------------------------------------------------------------------

        #---------------------
        #Select the edge frames such as they have all their samples
        _, bn = np.unique(pps, return_counts=True)
        pps_bins = bn[bn>0]
        if pps_bins[0]  < spf_data: pps_start = pps_bins[0]
        else: pps_start = 0 
        if pps_bins[-1] < spf_data: pps_end = -pps_bins[-1]
        else: pps_end = None
        #---------------------

        #-----------------------------------------------------------------------------------------------
        #Load the data on a dectector-per-detector basis. 
        #The data are first loaded, then despiked, then high- and low-pass filtered, and finaly decimate to target_frequency. 

        #if downsample is True, define an anti-aliasing filter. 
        if(self.downsample): aaf = det.AntiAliasingFilter( fs_in=spf_data, fs_out=self.freq_target, fc=self.freq_target/2-5, DT=self.DT,window='hann')
        kidutils = det.kidsutils()

        det_data = []

        #For each detector: 
        for kid in self.det_name: 
            '''
            det_I_string = 'kid'+kid+'_I_roachN' #different options in the names here
            det_Q_string = 'kid'+kid+'_Q_roachN'
            I_data = self.load(self.det_path, det_I_string, self.det_file_type)
            Q_data = self.load(self.det_path, det_Q_string, self.det_file_type)
            det_data = kidutils.KIDmag(I_data, Q_data)
            '''
            if('.hdf5' in self.det_path):
                data = data_value.loaddata_hdf5(self.det_path, f'kid_{kid}_roach', self.DT, self.numframes, self.startframe) #kidutils.KIDmag(I_data, Q_data))
            else: 
                data = self.loaddata_dirfile(self.det_path,  f'kid_{kid}_roach', self.DT, self.numframes, self.startframe,) 

            #remove the frames that don't have all their samples: 
            data = data[pps_start:pps_end]

            #Despike the data. 
            if self.despike:
                try:                     
                    desp = det.despike(data)
                    data = desp.replace_peak(hthres=self.sigma, pthres=self.prominence)
                except Exception as e:
                    continue

            #Decimate the data if downsample is True. 
            if(self.downsample): det_data.append( aaf.process(data) )
            else: det_data.append( data )

        #remove the frames that don't have all their samples and decimate the timestamps. 
        dettime = dettime[pps_start:pps_end]
        if(self.downsample): dettime = aaf.downsample(dettime)

        if(self.downsample): spf_data = self.freq_target
        #-----------------------------------------------------------------------------------------------
        
        #-----------------------------------------------------------------------------------------------
        '''
        #For debbuging purpose only
        ras = []
        decs = []
        for kid in kid_num: 
            H = h5py.File(self.det_path, "a")
            f = H[f'kid_{kid}_roach']
            ras.append( f['RA_roach'][int(first_frame*spf_data):int((first_frame+num)*spf_data)] )
            decs.append( f['DEC_roach'][int(first_frame*spf_data):int((first_frame+num)*spf_data)] )
            H.close()
        '''
        #-----------------------------------------------------------------------------------------------
        
        
        print('COORDINATES', self.coord1_name.lower(), self.coord2_name.lower())

        #-----------------------------------------------------------------------------------------------
        # Load the timestamps associated with the coordinates, latitude and lst, assuming they all have the same timestamps. 
        if('.hdf5' in self.det_path): 
            pps = data_value.loaddata_hdf5(self.det_path, f'coords_pps',self.IT, self.numframes, self.startframe)
            subsec = data_value.loaddata_hdf5(self.det_path, f'coords_subsecond_ps',self.DT, self.numframes, self.startframe)
            ctime  = pps.astype(self.DT)+subsec
            #Assumes ctime and coords. have the same spf.
            spf_ctime = data_value.loadspf_hdf5(self.det_path, f'coords_time')
            spf_coord = data_value.loadspf_hdf5(self.det_path, self.coord2_name)
            #Load the turnaround flags 
            turnaround_flags = data_value.loaddata_hdf5(self.det_path, f'turnaround_flags', self.DT, self.numframes, self.startframe)
            #Load the 1st coordinate timestream. 
            coord2_data = data_value.loaddata_hdf5(self.det_path, f'{self.coord2_name}', self.DT, self.numframes, self.startframe)
            if self.coord1_name.lower() == 'xel': 
                coord1_data = data_value.loaddata_hdf5(self.det_path, 'EL', self.DT, self.numframes, self.startframe)
                coord1_data *= np.cos(np.radians(coord2_data)) 
            else: coord1_data = data_value.loaddata_hdf5(self.det_path, f'{self.coord1_name}', self.DT, self.numframes, self.startframe)
            lat = data_value.loaddata_hdf5(self.det_path, 'lat',self.DT, self.numframes, self.startframe)
            lst = data_value.loaddata_hdf5(self.det_path, 'lst',self.DT, self.numframes, self.startframe)
            lst_lat_spf = data_value.loadspf_hdf5(self.det_path, 'lst')

        else:
            pps = self.loaddata_dirfile(self.det_path, f'coords_pps',self.IT, self.numframes, self.startframe)
            subsec = self.loaddata_dirfile(self.det_path, f'coords_subsecond_ps',self.DT, self.numframes, self.startframe)
            ctime  = pps.astype(self.DT)+subsec
            #Assumes ctime and coords. have the same spf.
            spf_ctime = self.loadspf_dirfile(self.det_path, f'coords_time')
            spf_coord = self.loadspf_dirfile(self.det_path, self.coord2_name)
            #Load the turnaround flags 
            turnaround_flags = self.loaddata_dirfile(self.det_path, f'turnaround_flags', self.DT, self.numframes, self.startframe)
            #Load the 1st coordinate timestream. 
            coord2_data = self.loaddata_dirfile(self.det_path, f'{self.coord2_name}', self.DT, self.numframes, self.startframe)
            if self.coord1_name.lower() == 'xel': 
                coord1_data = self.loaddata_dirfile(self.det_path, 'EL', self.DT, self.numframes, self.startframe)
                coord1_data *= np.cos(np.radians(coord2_data)) 
            else: coord1_data = self.loaddata_dirfile(self.det_path, f'{self.coord1_name}', self.DT, self.numframes, self.startframe)
            lat = self.loaddata_dirfile(self.det_path, 'lat',self.DT, self.numframes, self.startframe)
            lst = self.loaddata_dirfile(self.det_path, 'lst',self.DT, self.numframes, self.startframe)
            lst_lat_spf = self.loadspf_dirfile(self.det_path, 'lst')

        #Select the edge frames such as they have all their samples
        _, bn = np.unique(pps, return_counts=True)
        pps_bins = bn[bn>0]
        if pps_bins[0] < spf_coord: pps_start = pps_bins[0]
        else: pps_start = 0 
        if pps_bins[-1] < spf_coord: pps_end = -pps_bins[-1]
        else: pps_end = None

        #remove the frames that don't have all their samples
        ctime = ctime[pps_start:pps_end]
        coord1_data = coord1_data[pps_start:pps_end]
        coord2_data = coord2_data[pps_start:pps_end]
        lat = lat[pps_start:pps_end]
        lst = lst[pps_start:pps_end]
        turnaround_flags = turnaround_flags[pps_start:pps_end]
        #-----------------------------------------------------------------------------------------------

        #-----------------------------------------------------------------------------------------------
        # Decimate the coordinates (and their timestamps) to freq_target.
        if(self.freq_target is not None and spf_ctime > self.freq_target and self.downsample): 
            aaf = det.AntiAliasingFilter( fs_in=spf_ctime, fs_out=self.freq_target, fc=self.freq_target/2-5, DT=self.DT, window='hann')
            ctime= aaf.downsample(ctime)
            self.coord1_data = aaf.downsample(self.coord1_data)
            self.coord2_data = aaf.downsample(self.coord2_data)
            self.lst_data = aaf.downsample(self.lst_data)
            self.lat_data = aaf.downsample(self.lat_data)
            turnaround_flags = aaf.downsample(turnaround_flags)
        #-----------------------------------------------------------------------------------------------

        #-----------------------------------------------------------------------------------------------
        '''
        #For debbuging purpose only
        coord2_data= data_value.loaddata(self.det_path, 'data_'+f'{self.coord2_name}', self.DT, num, first_frame) #!!
        coord1_data = data_value.loaddata(self.det_path, 'data_'+f'{self.coord1_name}', self.DT, num, first_frame) #!!
        spf_coord = data_value.loadspf(self.det_path, 'data_'+self.coord2_name, self.DT)
        #acqfreq_coord = data_value.load_acquisition_frequency(self.det_path, self.coord2_name, )
        #---------------------------------------------------------------------------------
        lat = data_value.loaddata(self.det_path, 'data_lat',self.DT, num, first_frame)
        lst = data_value.loaddata(self.det_path, 'data_lst',self.DT, num, first_frame)
        lst_lat_spf = data_value.loadspf(self.det_path, 'data_lst',self.DT)
        #acqfreq_lstlat = data_value.load_acquisition_frequency(self.det_path, 'lst')
        '''
        #-----------------------------------------------------------------------------------------------


        #-----------------------------------------------------------------------------------------------
        '''
        # For test purpose only
        if(self.P['save_raw_IQ_TODS']):

            tods_compressor = compress_tods(self.P['output_tods'], kid_num, det_data, spf_data, dettime, 
                                            self.coord1_name, self.coord2_name, coord1_data, coord2_data, spf_coord, ctime, 
                                            first_frame, num, lst, lat, self.P, self.DT, self.IT, prefix= 'I_')
            tods_compressor.save_tods()
            tods_compressor = compress_tods(self.P['output_tods'], kid_num, det_data, spf_data, dettime, 
                                            self.coord1_name, self.coord2_name, coord1_data, coord2_data, spf_coord, ctime, 
                                            first_frame, num, lst, lat, self.P, self.DT, self.IT, prefix= 'Q_')
            tods_compressor.save_tods()
        '''
        #-----------------------------------------------------------------------------------------------

        return dettime, det_data, ctime, coord1_data, coord2_data, turnaround_flags, lst, lat, spf_data, spf_coord, lst_lat_spf #ras, decs#, acqfreq_data, acqfreq_coord, acqfreq_lstlat

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

class save_tods():
    
    '''
    Class to save timestreams.
    If the output name ends with '.hdf5', save the timestreams in a hdf5 file. Else save it as dirfile.  
    Parameters
    ----------
    Returns
    -------
    '''

    def __init__(self, tods_path,kid_num, det_data, det_sample_frame, det_timestamps,\
                 coord1, coord2, coord1_data, coord2_data, coords_sample_frame, ctime, startframe, numframes, lst_data, lat_data, P,
                 DT, IT, prefix=''):
        
        '''
        Class to save the detector and coordinates timestreams. 
        Parameters
        ----------
        tods_path: str
            path and name of output file in which to save the timestreams. 
            if the name ends with '.hdf5', the created file is a hdf5 file. Otherwise it is a dirfile. 
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
        coords_sample_frame: int
            sample per frame of the coordinate timestreams
        ctime: 1d array
            the coordinates timestamps.
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
        prefix: str
            the prefix of the key under which to save the detector's timestream 

        Returns
        -------
        '''
        self.tods_path = tods_path                               #Path of the timestreams hdf5
        self.kid_num = kid_num                                   #Dectector name list
        self.det_data = det_data                                 #Detector data timestream
        self.det_sample_frame = det_sample_frame     #Detector samples in each frame of the timestream
        self.det_timestamps = det_timestamps                     #Detector timestamps
        self.coord1 = coord1                                     #Coordinate 1 name  
        self.coord2 = coord2                                     #Coordinate 2 name
        self.coord1_data = coord1_data                           #Coordinate 1 data timestream                        
        self.coord2_data = coord2_data                           #Coordinate 2 data timestream
        self.coords_sample_frame = coords_sample_frame           #Sample per frame of the coordinates
        self.ctime = ctime                                       # Coordinates timestamps.                           
        self.startframe = startframe                             #Start frame
        self.numframes = numframes                               #Number of frames
        self.lst_data = lst_data                                 #LST timestream (if correction is required and coordinates are RA-DEC)
        self.lat_data = lat_data                                 #LAT timestream (if correction is required and coordinates are RA-DEC)
        self.P = P                                               #Parameter dictionary
        self.DT=DT                                               #Float precision required 
        self.IT=IT                                               #Int precision required 
        self.prefix = prefix                                     #the prefix of the key under which to save the detector's timestream 

    def fct_save_tods(self):
        '''
        #Dedeping on the output name, save the TODs in .hdf5 or in dirfile. 
        Parameters
        ----------
        Returns
        -------
        '''

        if('.hdf5' in self.tods_path): 
            self.save_tods_hdf5()
        else: 
            import pygetdata as gd
            self.save_tods_dirfile()

        print('')
        return 0
    
    def save_tods_hdf5(self):

        '''
        Save the timestreams in an .hdf5 
        Parameters
        ----------
        Returns
        -------
        '''

        # Create the file if it doesn't exist, otherwise open it    
              
        '''
        if not os.path.exists(self.tods_path):
            # 'w' creates a new file (overwrites if exists)
            with h5py.File(self.tods_path, "w") as f:
                print(f"Created empty file: {self.tods_path}")
        else:
            # 'a' opens existing file (read/write mode)
            with h5py.File(self.tods_path, "a") as f:
                print(f"Opened existing file: {self.tods_path}")       
        '''
        with h5py.File(self.tods_path, "w") as f:
            print(f"Created file: {self.tods_path}")

        #-----------------------------------------------------------------------------------------------
        data = np.asarray(self.det_data)
        data, min, max = self.to8bit_intprecision(data)

        for d, kid in zip(data, self.kid_num):
            self.save_array_to_hdf5(f"{self.prefix}KID_{kid}", (d,), (kid,), spf=self.det_sample_frame, min=min, max=max)
        self.save_array_to_hdf5('dettime', (self.det_timestamps,), ('dettime',), spf=self.det_sample_frame, min=min, max=max)
        self.save_array_to_hdf5('frames',(self.startframe, self.numframes), ('start_frame', 'num_frames'))
        #-----------------------------------------------------------------------------------------------

        #-----------------------------------------------------------------------------------------------
        #coords = np.vstack((self.coord1_data,self.coord2_data, self.lst_data, self.lat_data, self.ctime)).T
        #if(self.int8): coords, min, max = self.to8bit_intprecision(coords)
        #coords, min, max = self.DT(coords), None, None
        for array, name in zip((self.coord1_data,self.coord2_data, self.lst_data, self.lat_data, self.ctime), (self.coord1,self.coord2,'LST','latitude', 'coords_timestamps')):
            self.save_array_to_hdf5(name, (array,), (name,), spf=self.coords_sample_frame, min=min, max=max)
        #-----------------------------------------------------------------------------------------------

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
    
    def to8bit_intprecision(self, array): 

        '''
        function to save an array in 8bits precision. 
        Parameters
        array: ndarray
            array to be downsized
        ----------
        Returns
        -------
        array: ndarray
            Array in 8bits precision
        min: float
            the minimum negative value of the array
        max: float
            The maximum value of the array. 
        '''        

        # Rescale data to float range 0.0-1.0

        min = array.min()
        if(min<0): array -= min
        else: min=None

        max = np.abs(array).max()
        array /= max

        # 1. Scale the values to the 0-255 range
        scaled_array = array * 255

        # 2. Clip the values to ensure they are within the 8-bit range (0-255)
        clipped_array = np.clip(scaled_array, 0, 255)

        # 3. Convert to unsigned 8-bit integer type
        downsampled_array = clipped_array.astype(np.uint8)

        return downsampled_array, np.float16(min), np.float16(max)

    def save_tods_dirfile(self):

        '''
        Save the timestreams in a dirfile
        Parameters
        ----------
        Returns
        -------
        '''


        if os.path.exists(self.tods_path):
            shutil.rmtree(self.tods_path)
            print(f"Removed existing directory: {self.tods_path}")

        df = gd.dirfile(self.tods_path, gd.RDWR | gd.CREAT | gd.TRUNC)

        data = np.asarray(self.det_data)
        data, min, max = self.to8bit_intprecision(data)

        for d, kid in zip(data, self.kid_num):

            field_name = f"{self.prefix}KID_{kid}"

                    
            entry = gd.entry(
                gd.RAW_ENTRY,
                field_name,
                0,
                parameters={
                    "type": gd.INT8,
                    "spf": int(self.det_sample_frame)
                }
            )
        
            try:
                df.add(entry)
            except gd.DuplicateError:
                df.delete(key)   # remove existing field
                df.add(entry)           # recreate it


            df.putdata(field_name, d)

        minmax = np.asarray((min, max), dtype=np.float64)
        frames = np.asarray((self.startframe, self.numframes), dtype=np.float64)

        for values, field_name in zip(( minmax, frames ), (f"min_max_{self.prefix}KID","first_frame_num_frames")):
                
            entry = gd.entry(
                gd.RAW_ENTRY,
                field_name,
                0,
                parameters={
                    "type": gd.FLOAT64,
                    "spf": 1
                }
            )

            try:
                df.add(entry)
            except gd.DuplicateError:
                df.delete(key)   # remove existing field
                df.add(entry)           # recreate it

            values = np.asarray(values, dtype=np.float64)
            df.putdata(field_name, values)

        for coords, field_name in zip((self.coord1_data,self.coord2_data, self.lst_data, self.lat_data, self.det_timestamps), ((self.coord1,self.coord2,'LST','latitude', 'synch_timestamps'))):
            #if(self.int8): coords, min, max = self.to8bit_intprecision(coords)
            coords, min, max = np.float32(coords), None, None

            entry = gd.entry(
                gd.RAW_ENTRY,
                field_name,
                0,
                parameters={
                    "type": gd.FLOAT32,
                    "spf": int(self.coords_sample_frame)
                }
            )

            try:
                df.add(entry)
            except gd.DuplicateError:
                df.delete(key)   # remove existing field
                df.add(entry)           # recreate it
            df.putdata(field_name, coords)
            #-----------------------------------------------------------------------------------------------

        for key in self.P:
            entry = gd.entry(
                gd.STRING_ENTRY,
                'param_'+key,
                0,
                parameters={key: f"{self.P[key]}" }
            )


            try:
                df.add(entry)
            except gd.DuplicateError:
                df.delete(key)   # remove existing field
                df.add(entry)           # recreate it

        df.close()


        if('zip' in self.tods_path):

            shutil.make_archive(
                base_name=self.tods_path,   # name of the zip file (no .zip)
                format="zip",
                root_dir=self.tods_path
            )
        
        return 0

    
class frame_zoom_sync():

    '''
    This class is designed to sync detector and coordinates timestream given a different sampling of the two
    Parameters
    ----------
    Returns
    -------
    '''

    def __init__(self, dettime, det_data, det_sample_frame,\
                 ctime, coord1_data, coord2_data, coord_sample_frame, \
                 turnaround_flags, lst_data, lat_data, lstlat_sample_frame, \
                 DT, IT):
        
        '''
        Create an instance of the class designed sync detector and coordinates timestream given a different sampling of the two
        Parameters
        ----------
        dettime: 1d array
             detector data timestamps
        det_data: list
            Detector data timestream
        det_sample_frame: int
            Number of samples in each frame of the data timestreams
        ctime: 1d array
            coordinates timestamps. 
        coord1_data: 1d array
            Coordinate 1 data timestream
        coord2_data: 1d array
            Coordinate 2 data timestream
        coord_sample_frame: int
            Number of samples in each frame of the coordinate timestreams
        turnaround_flags:
            if True, the coordinate sample is taken when the telescope speed is not constant
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

        self.dettime = dettime                                   #Detector data timestamps
        self.det_data = det_data                                 #Detector data timestreams
        self.det_sample_frame = int(float(det_sample_frame))     #Detector samples in each frame of the timestream
        self.ctime = ctime                                       #Coordinates timestamps
        self.coord1_data = coord1_data                           #Coordinate 1 data timestream
        self.coord_sample_frame = int(float(coord_sample_frame)) #Coordinates samples in each frame of the time stream
        self.coord2_data = coord2_data                           #Coordinate 2 data timestream
        self.turnaround_flags = turnaround_flags                 #Flags for not-constant telescope speed.
        self.lst_data = lst_data                                 #LST timestream (if correction is required and coordinates are RA-DEC)
        self.lat_data = lat_data                                 #LAT timestream (if correction is required and coordinates are RA-DEC)
        self.lstlat_sample_frame = lstlat_sample_frame           #LST-LAT samples per frame (if correction is required and coordinates are RA-DEC)
        self.DT = DT                                             #Float precision required 
        self.IT = IT                                             #Int precision required 
  
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

        coord1_int = interp1d(time_acs, coord1, kind='linear',bounds_error=False,fill_value="extrapolate")
        coord2_int = interp1d(time_acs, coord2, kind= 'linear',bounds_error=False,fill_value="extrapolate")

        return coord1_int(time_det), coord2_int(time_det)

    def sync_data(self):

        '''
        Wrapper for the previous functions to return the slices of the detector and coordinates TODs,  
        and the associated time.

        Parameters
        ----------
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

        #---------------------------------------------------------------
        #Get the data samples whose timestamps are shared with the coordinates timestamps

        # Determine common time interval (overlap)
        dettime, ctime = self.dettime, self.ctime
        start_time = max(ctime[0], dettime[0]  )   # latest starting time
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
        self.turnaround_flags = self.turnaround_flags[i_c_start:i_c_end]
        #---------------------------------------------------------------
        
        #---------------------------------------------------------------
        #Match the number of coordinates samples (coord1, coord2, lat, lst and the turnaround flags) to data samples.
        if(len(ctime) != len(dettime)):
            self.coord1_data, self.coord2_data = self.coord_int(self.coord1_data, self.coord2_data, ctime, dettime)
            self.lst_data, self.lat_data       = self.coord_int(self.lst_data, self.lat_data, ctime, dettime)
            f = interp1d(ctime, self.turnaround_flags, kind='linear',bounds_error=False,fill_value="extrapolate")
            self.turnaround_flags= np.round(f(dettime))
        #---------------------------------------------------------------

        return dettime, self.det_data, self.coord1_data, self.coord2_data, self.lst_data, self.lat_data, self.turnaround_flags
