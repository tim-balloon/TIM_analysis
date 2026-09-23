#import pygetdata as gd
import numpy as np
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.signal import resample_poly, resample
import os, h5py, shutil
import astropy.table as tb
from IPython import embed
import src.detector as det 
import matplotlib.pyplot as plt
import pygetdata as gd

def load_params(path):
    """
    Returns as a dictionary the parameters stored in a .par file
    Parameters
    ----------
    path: str
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
    Load and process detector and telescope-coordinate timestreams.

    This class provides methods to load detector data, timestamps, and
    telescope coordinates from HDF5 files or DIRFILEs. Detector
    timestreams can optionally be despiked and downsampled. Coordinate
    timestreams, local sidereal time, and telescope latitude can also be
    loaded and downsampled to match the detector data.


    Parameters
    ----------

    Returns
    -------
    '''

    def __init__(self, det_path, det_name, coord1_name, \
                 coord2_name, startframe, numframes, \
                 despike, sigma, prominence, \
                 downsample, freq_target, fc, DT, IT, 
                 remove_turnarounds=False, bufferframe=0):

        """
        Create an instance for loading and processing timestream data.

        Parameters
        ----------
        det_path : str
            Path to the HDF5 file or DIRFILE containing the data.
        det_name : list
            List of detector names to be analyzed.
        coord1_name : str
            Name of the first coordinate timestream.
        coord2_name : str
            Name of the second coordinate timestream.
        startframe : int
            First frame to analyze.
        numframes : int
            Number of frames to analyze.
        despike : bool
            If True, despike the detector timestreams.
        sigma : float
            Detection threshold for spikes in units of the standard
            deviation.
        prominence : float
            Minimum spike prominence in units of the standard deviation.
        downsample : bool
            If True, downsample the detector and coordinate
            timestreams.
        freq_target : float
            Target sampling frequency in Hz.
        DT : type
            Floating-point data type used for the loaded data.
        IT : type
            Integer data type used for integer-valued data.
        remove_turnarounds : bool, optional
            If True, load and return turnaround flags. Defaults to False.
        Returns
        -------
        """    
        self.det_path = det_path                    #Path of the detector dirfile
        self.det_name = det_name                    #Detector name to be analyzed kidnum
        self.coord1_name = coord1_name              #Coordinates 1 name, e.g. RA or AZ
        self.coord2_name = coord2_name              #Coordinates 2 name
        self.startframe = startframe                #Starting frame to be analyzed
        self.numframes = numframes                  #Ending frame to be analyzed
        self.DT=DT                                  #Float precision required 
        self.IT=IT                                  #Int precision required 
        self.freq_target = freq_target              #Frequency in Hz to downsample the data to
        self.fc = fc                                #Frequency in Hz for anti-aliasing filter 
        self.downsample = downsample                #If True, downsample the data
        self.sigma = sigma                          #height in std value to look for spikes
        self.prominence = prominence                #prominence in std value to look for spikes
        self.despike = despike                      #if True despikes the data 
        self.remove_turnarounds = remove_turnarounds
        self.bufferframe = bufferframe
        self.startframe += self.bufferframe

    def conversion_type(self, file_type):
        """
        Convert a data type string to the corresponding pygetdata type.

        Parameters
        ----------
        file_type : str
            Data type identifier used by the input file, such as 'u16', 'u32', 's32', or 'float'.

        Returns
        -------
        gdtype : gdtype
            pygetdata data type corresponding to file_type.
        """

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
        Load the number of samples per frame for an HDF5 field.

        Parameters
        ----------
        file : str
            Path to the HDF5 file.
        field : str
            Name of the field for which to retrieve the samples
            per frame.

        Returns
        -------
        spf : int or None
            Number of samples per frame. Returns None if the spf field is not present.
        """
        H = h5py.File(file, "a")
        f = H[field]
        if('spf' in f.keys()): spf = f['spf'][()]
        else: spf = None
        H.close()
        return spf
    
    def loadspf_dirfile(self, file, field):
        """
        Load the number of samples per frame for an HDF5 field.

        Parameters
        ----------
        file : str
            Path to the HDF5 file.
        field : str
            Name of the field for which to retrieve the samples
            per frame.

        Returns
        -------
        spf : int or None
            Number of samples per frame. Returns None if the spf field is not present.
        """

        d = gd.dirfile(file, gd.RDONLY)
        spf = d.spf(field)

        return spf
        
    def loaddata_hdf5(file, field, DT, num_frames=None, first_frame=None):
        """
        Load data from an HDF5 field.

        If num_frames and first_frame are provided and the field
        contains a samples-per-frame value, only the requested range of
        frames is loaded. Otherwise, the complete timestream is loaded.

        Parameters
        ----------
        file : str
            Path to the HDF5 file.
        field : str
            Name of the field to load.
        DT : type
            Data type used for the returned array.
        num_frames : int, optional
            Number of frames to load.
        first_frame : int, optional
            First frame to load.

        Returns
        -------
        data : numpy.ndarray
            Values stored in the requested field. When frame selection
            is used, the returned data span samples from first_frame * spf to (first_frame + num_frames) * spf.
        """
        if os.path.isfile(file): H = h5py.File(file, "a")
        else: print('no file')
        f = H[field]
        if(('spf' in f.keys()) and (num_frames is not None) and (first_frame is not None)):
            spf = f['spf'][()]
            data = f['data'][int(first_frame*spf):int((first_frame+num_frames)*spf)].astype(DT, copy=False)
        else: 
            data = f['data'][()].astype(DT, copy=False)
        H.close()
        return data
    
    def loaddata_dirfile(self, filepath, file, DT, num=None, first_frame=None, file_type=None):

        """
        Load a field from a DIRFILE as a NumPy array.

        Parameters
        ----------
        filepath : str
            Path to the DIRFILE.
        file : str
            Name of the field to load, such as a detector or
            coordinate name.
        DT : type
            Data type used for the returned array.
        num : int, optional
            Number of frames to load. If None, load all available
            frames.
        first_frame : int, optional
            First frame to load. Defaults to the first frame.
        file_type : str, optional
            Input data type identifier passed to conversion_type.

        Returns
        -------
        values : numpy.ndarray
            Values loaded from the DIRFILE.
        """
        d = gd.dirfile(filepath, gd.RDONLY)

        if file_type is not None:  gdtype = self.conversion_type(file_type)
        else:                      gdtype = gd.FLOAT64

        if(num is None): num = d.nframes
        if(first_frame is None): first_frame = 0

        values = d.getdata(file, gdtype, num_frames = num, first_frame=first_frame)
        values = np.asarray(values, dtype=DT)

        return values

    def values(self):
        """
        Load and process detector and coordinate timestreams.

        Detector timestreams are loaded for the requested list of
        detectors. Depending on the class configuration, the detector
        data are despiked and downsampled. The corresponding coordinate
        timestreams, timestamps, local sidereal time, latitude, and
        turnaround flags are also loaded and optionally downsampled.

        Parameters
        ----------

        Returns
        -------
        dettime : numpy.ndarray
            Detector timestamp timestream.
        det_data : list
            List of detector amplitude timestreams.
        ctime : numpy.ndarray
            Coordinate timestamp timestream.
        coord1_data : numpy.ndarray
            First coordinate timestream.
        coord2_data : numpy.ndarray
            Second coordinate timestream.
        turnaround_flags : numpy.ndarray or None
            Turnaround flags indicating samples taken while the
            telescope speed is not constant. Returns None if remove_turnarounds is False.
        lst : numpy.ndarray
            Local sidereal time timestream.
        lat : numpy.ndarray
            Telescope latitude timestream.
        spf_data : int
            Number of samples per frame of the detector timestreams.
        spf_coord : int
            Number of samples per frame of the coordinate timestreams.
        lst_lat_spf : int
            Number of samples per frame of the latitude and local
            sidereal time timestreams.
        """
        #-----------------------------------------------------------------------------------------------

        # Load the sample-per-frame of the detector timestreams (assuming they all have the same spf). 
        if('.hdf5' in self.det_path):
            spf_data = self.loadspf_hdf5(self.det_path,  f'data_time')
            #Load the detector timestamps, assuming the detectors all have the same timestamps. 
            #1st, load the pulse per second, which defines to which second each sample belong to. 
            pps = self.loaddata_hdf5(self.det_path, f'data_pps', self.DT, self.numframes, self.startframe,) 
            #2nd, load the sub-second part of the timestamps. 
            subsec = self.loaddata_hdf5(self.det_path, f'data_subsecond_ps',self.DT, self.numframes, self.startframe)
        else: 
            spf_data = self.loadspf_dirfile(self.det_path,  f'data_time')
            pps      = self.loaddata_dirfile(self.det_path, f'data_pps', self.DT, self.numframes, self.startframe,) 
            subsec   = self.loaddata_dirfile(self.det_path, f'data_subsecond_ps',self.DT, self.numframes, self.startframe)
       
        #Get the final timestamps.
        dettime = pps+subsec


        #-----------------------------------------------------------------------------------------------

        
        #Select the edge frames such as they have all their samples
        _, bn = np.unique(pps, return_counts=True)
        pps_bins = bn[bn>0]
        if pps_bins[0]  < spf_data: pps_start = pps_bins[0]
        else: pps_start = 0 
        if pps_bins[-1] < spf_data: pps_end = -pps_bins[-1]
        else: pps_end = None
        

        #-----------------------------------------------------------------------------------------------

        
        #if downsample is True, define an anti-aliasing filter. 
        if(self.downsample): aaf = det.AntiAliasingFilter( fs_in=spf_data, fs_out=self.freq_target, fc=self.fc, DT=self.DT,window='hann')
        kidutils = det.kidsutils()


        #-----------------------------------------------------------------------------------------------

        #Load the data on a dectector-per-detector basis. 
        #The data are first loaded, then despiked, then high- and low-pass filtered, and finaly decimate to target_frequency. 

        kidutils = det.kidsutils()
        det_data = []

        #For each detector: 
        for kid in self.det_name: 
            
            det_I_string = f'kid_{kid}_I_roach' #different options in the names here
            det_Q_string = f'kid_{kid}_Q_roach'
            
            if('.hdf5' in self.det_path):
                I_data = self.loaddata_dirfile(self.det_path, det_I_string, self.DT, self.numframes, self.startframe,)
                Q_data = self.loaddata_dirfile(self.det_path, det_Q_string, self.DT, self.numframes, self.startframe,)
            else:     
                #data = self.loaddata_dirfile(self.det_path,  f'kid_{kid}_roach', self.DT, self.numframes, self.startframe,) 
                I_data = self.loaddata_dirfile(self.det_path, det_I_string, self.DT, self.numframes, self.startframe,)
                Q_data = self.loaddata_dirfile(self.det_path, det_Q_string, self.DT, self.numframes, self.startframe,)

            data = kidutils.KIDmag(I_data, Q_data)
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
            if(self.downsample):  
                det_data.append( aaf.process(data) )
            else: 
                det_data.append( data )
        

        #remove the frames that don't have all their samples and decimate the timestamps. 
        dettime = dettime[pps_start:pps_end]
        if(self.downsample): dettime = aaf.downsample(dettime)

        if(self.downsample): spf_data = self.freq_target
        

        #-----------------------------------------------------------------------------------------------


        print('COORDINATES', self.coord1_name.lower(), self.coord2_name.lower())
        # Load the timestamps associated with the coordinates, latitude and lst, assuming they all have the same timestamps. 
        if('.hdf5' in self.det_path): 
            pps = data_value.loaddata_hdf5(self.det_path, f'coords_pps',self.IT, self.numframes, self.startframe)
            subsec = data_value.loaddata_hdf5(self.det_path, f'coords_subsecond_ps',self.DT, self.numframes, self.startframe)
            ctime  = pps.astype(self.DT)+subsec
            #Assumes ctime and coords. have the same spf.
            spf_ctime = data_value.loadspf_hdf5(self.det_path, f'coords_time')
            spf_coord = data_value.loadspf_hdf5(self.det_path, self.coord2_name)
            #Load the turnaround flags 
            if(self.remove_turnarounds): turnaround_flags = data_value.loaddata_hdf5(self.det_path, f'turnaround_flags', self.DT, self.numframes, self.startframe)
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
            if(self.remove_turnarounds): turnaround_flags = self.loaddata_dirfile(self.det_path, f'turnaround_flags', self.DT, self.numframes, self.startframe)
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
        if(self.remove_turnarounds): turnaround_flags = turnaround_flags[pps_start:pps_end]


        #-----------------------------------------------------------------------------------------------

        
        # Decimate the coordinates (and their timestamps) to freq_target.
        if(self.freq_target is not None and spf_ctime > self.freq_target and self.downsample): 
            aaf = det.AntiAliasingFilter( fs_in=spf_ctime, fs_out=self.freq_target, fc=self.fc, DT=self.DT, window='hann')
            ctime= aaf.downsample(ctime)
            self.coord1_data = aaf.downsample(self.coord1_data)
            self.coord2_data = aaf.downsample(self.coord2_data)
            self.lst_data = aaf.downsample(self.lst_data)
            self.lat_data = aaf.downsample(self.lat_data)
            if(self.remove_turnarounds): turnaround_flags = aaf.downsample(turnaround_flags)


        #-----------------------------------------------------------------------------------------------

        if(not self.remove_turnarounds): turnaround_flags= None
        
        return dettime, det_data, ctime, coord1_data, coord2_data, turnaround_flags, lst, lat, spf_data, spf_coord, lst_lat_spf
        
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
        frame1 : int
            Starting frame
        frame2 : int
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
        Function to read a star camera offset file and return the coordinates offset

        Parameters
        ----------

        Returns
        -------
        xsc_1 : numpy.ndarray
            timestream of the star camera coordinate 1 
        xsc_2 : numpy.ndarray
            timestream of the star camera coordinate 
        '''

        path = os.getcwd()+'/xsc_'+str(int(self.xsc))+'.txt'

        xsc_file = np.loadtxt(path, skiprows = 2)

        index, = np.where((xsc_file[0]>=float(self.frame1)) & (xsc_file[1]<float(self.frame2)))

        if np.size(index) > 1:
            index = index[0]

        xsc_1 = xsc_file[2]
        xsc_2 = xsc_file[3]

        return xsc_1, xsc_2

class det_table():
    '''
    Load detector properties from a detector table.

    The detector table contains the boresight offsets, white-noise
    levels, and detector responses for the selected detectors.

    Parameters
    ----------

    Returns
    -------
    '''

    def __init__(self, dets, pathtable):
        '''
        Create an instance for loading detector properties.

        Parameters
        ----------
        dets : list
            List of detector names for which to load the properties.
        pathtable : str
            Path to the file containing the detector property table.
        
        Returns
        -------
        '''

        self.name = dets
        self.pathtable = pathtable

    def loadtable(self):
        """
        Load detector properties from the detector table.

        The detector table is read and the properties corresponding to
        the selected detectors are extracted.

        Parameters
        ----------

        Returns
        -------
        det_off : numpy.ndarray
            Array of shape ``(N, 2)`` containing the detector angular
            offsets from the center of the array. The first column
            contains the cross-elevation (XEL) offsets and the second
            column contains the elevation (EL) offsets.
        noise : numpy.ndarray
            Array containing the white-noise level of each detector.
        resp : numpy.ndarray
            Array containing the response of each detector.
        """

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
    Class to save detector and telescope-coordinate timestreams.

    The output format is determined by the extension of tods_path:
    files ending in .hdf5 are saved as HDF5 files, while other outputs
    are saved as dirfiles.

    Parameters
    ----------

    Returns
    -------
    '''

    def __init__(self, tods_path,kid_num, det_data, det_sample_frame, det_timestamps,\
                 coord1, coord2, coord1_data, coord2_data, coords_sample_frame, ctime, 
                 startframe, numframes, lst_data, lat_data, P,
                 DT, IT, prefix=''):
        
        '''
        Create an instance of the class to save detector and coordinate
        timestreams.

        Parameters
        ----------
        tods_path : str
            Path and name of the output file or directory.
        kid_num : list
            List of detector names.
        det_data : list
            List of cleaned detector timestreams, ordered according to
            ``kid_num``.
        det_sample_frame : float
            Number of detector samples per frame.
        det_timestamps : numpy.ndarray
            Timestamps associated with the detector timestreams.
        coord1 : str
            Name of the first coordinate.
        coord2 : str
            Name of the second coordinate.
        coord1_data : numpy.ndarray
            Timestream of the first coordinate.
        coord2_data : numpy.ndarray
            Timestream of the second coordinate.
        coords_sample_frame : int
            Number of samples per frame for the coordinate timestreams.
        ctime : numpy.ndarray
            Timestamps associated with the coordinate timestreams.
        startframe : int
            Index of the first loaded frame.
        numframes : int
            Number of loaded frames.
        lst_data : numpy.ndarray
            Local Sidereal Time timestream.
        lat_data : numpy.ndarray
            Latitude timestream.
        P : dict
            Dictionary of parameters to save as metadata.
        DT : type
            Floating-point data type or precision required for the output.
        IT : type
            Integer data type or precision required for the output.
        prefix : str, optional
            Prefix used for the detector timestream field names.

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
        Save the timestreams using the output format specified by
        tods_path. HDF5 is used when the output name contains
        .hdf5; otherwise a dirfile is created.

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

    
    def save_tods_hdf5(self):

        '''
        Save the detector and coordinate timestreams to an HDF5 file.

        Parameters
        ----------

        Returns
        -------
        '''

        # Creats the file if it doesn't exist, otherwise open it    
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

        
        for array, name in zip((self.coord1_data,self.coord2_data, self.lst_data, self.lat_data, self.ctime), (self.coord1,self.coord2,'LST','latitude', 'coords_timestamps')):
            self.save_array_to_hdf5(name, (array,), (name,), spf=self.coords_sample_frame, min=min, max=max)


        #-----------------------------------------------------------------------------------------------

    
    def save_array_to_hdf5(self, grp_name, data, list_names, spf=None, min=None, max=None):
        '''
        Save an array and its associated metadata to an HDF5 group.

        Parameters
        ----------
        grp_name : str
            Name of the HDF5 group in which to save the array.
        data : tuple
            Data arrays to save.
        list_names : tuple
            Names describing the data arrays.
        spf : int, optional
            Number of samples per frame, if applicable.
        min : float, optional
            Minimum value used when reconstructing data stored in 8-bit
            precision.
        max : float, optional
            Maximum value used when reconstructing data stored in 8-bit
            precision.

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

    
    def save_tods_dirfile(self):

        '''
        Save the detector and coordinate timestreams to a dirfile.

        Detector timestreams are stored as 8-bit integer data, while
        coordinate timestreams are stored as 32-bit floating-point data.
        The parameter dictionary is stored as metadata fields.

        Parameters
        ----------

        Returns
        -------
        '''

        if os.path.exists(self.tods_path):
            shutil.rmtree(self.tods_path)
            print(f"Removed existing directory: {self.tods_path}")
        print('dirfile is:' , self.tods_path)

        df = gd.dirfile(self.tods_path, gd.RDWR | gd.CREAT | gd.TRUNC)

        data = np.asarray(self.det_data)
        data, min, max = self.to8bit_intprecision(data)

        for d, kid in zip(data, self.kid_num):

            field_name = f"{self.prefix}KID_{kid}"

            entry = gd.entry(gd.RAW_ENTRY,field_name,0,parameters={"type": gd.INT8,"spf": int(self.det_sample_frame)})
        
            try:
                df.add(entry)
            except gd.DuplicateError:
                df.delete(key)   # remove existing field
                df.add(entry)           # recreate it

            df.putdata(field_name, d.tolist())

        minmax = np.asarray((min, max), dtype=np.float64)
        frames = np.asarray((self.startframe, self.numframes), dtype=np.float64)


        #-----------------------------------------------------------------------------------------------


        for values, field_name in zip(( minmax, frames ), (f"min_max_{self.prefix}KID","first_frame_num_frames")):
                
            entry = gd.entry(gd.RAW_ENTRY,field_name,0,parameters={"type": gd.FLOAT64,"spf": 1})

            try:
                df.add(entry)
            except gd.DuplicateError:
                df.delete(key)   # remove existing field
                df.add(entry)           # recreate it

            values = np.asarray(values, dtype=np.float64)
            df.putdata(field_name, values.tolist())

        for coords, field_name in zip((self.coord1_data,self.coord2_data, self.lst_data, self.lat_data, self.det_timestamps), ((self.coord1,self.coord2,'LST','latitude', 'synch_timestamps'))):
            #if(self.int8): coords, min, max = self.to8bit_intprecision(coords)
            coords, min, max = np.float32(coords), None, None

            entry = gd.entry(gd.RAW_ENTRY,field_name,0,parameters={"type": gd.FLOAT32,"spf": int(self.coords_sample_frame)})

            try:
                df.add(entry)
            except gd.DuplicateError:
                df.delete(key)   # remove existing field
                df.add(entry)           # recreate it
            df.putdata(field_name, coords.tolist())


        #-----------------------------------------------------------------------------------------------


        for key in self.P:
            entry = gd.entry(gd.STRING_ENTRY,'param_'+key,0,parameters={key: f"{self.P[key]}" })


            try:
                df.add(entry)
            except gd.DuplicateError:
                df.delete(key)   # remove existing field
                df.add(entry)           # recreate it

        df.close()


        if('zip' in self.tods_path):

            print('make the zip')
            shutil.make_archive(base_name=self.tods_path, format="zip",root_dir=self.tods_path)
        
        #-----------------------------------------------------------------------------------------------

    
    def to8bit_intprecision(self, array): 

        '''
        Convert an array to unsigned 8-bit integer precision.

        The array is rescaled to the range 0--255. If the input contains
        negative values, the minimum value is subtracted before scaling.
        The corresponding minimum and maximum values are returned so that
        the original scaling can be reconstructed.

        Parameters
        ----------
        array : numpy.ndarray
            Array to convert to 8-bit precision.

        Returns
        -------
        array : numpy.ndarray
            Array converted to unsigned 8-bit integers.
        min : float
            Minimum value of the input array, or ``None`` if the input
            contains no negative values.
        max : float
            Maximum absolute value used for the rescaling.
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
    
class frame_zoom_sync():
    '''
    Synchronize detector and coordinate timestreams with different sampling
    frequencies.

    The coordinate timestreams are interpolated to the detector timestamps
    when the two timestreams have different numbers of samples.

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
        Create an instance of the class to synchronize detector and
        coordinate timestreams with different sampling frequencies.

        Parameters
        ----------
        dettime : numpy.ndarray
            Timestamps of the detector timestream.
        det_data : list
            List of detector data timestreams.
        det_sample_frame : int
            Number of detector samples per frame.
        ctime : numpy.ndarray
            Timestamps of the coordinate timestreams.
        coord1_data : numpy.ndarray
            First coordinate timestream.
        coord2_data : numpy.ndarray
            Second coordinate timestream.
        coord_sample_frame : int
            Number of coordinate samples per frame.
        turnaround_flags : numpy.ndarray or None
            Flags indicating samples acquired while the telescope speed is
            not constant. If ``None``, turnaround flags are not synchronized.
        lst_data : numpy.ndarray
            Local Sidereal Time timestream.
        lat_data : numpy.ndarray
            Latitude timestream.
        lstlat_sample_frame : int
            Number of LST and latitude samples per frame.
        DT : type
            Floating-point data type or precision required.
        IT : type
            Integer data type or precision required.

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
        if(self.turnaround_flags == None): self.remove_turnarounds = False
        else: self.remove_turnarounds = True
        self.lst_data = lst_data                                 #LST timestream (if correction is required and coordinates are RA-DEC)
        self.lat_data = lat_data                                 #LAT timestream (if correction is required and coordinates are RA-DEC)
        self.lstlat_sample_frame = lstlat_sample_frame           #LST-LAT samples per frame (if correction is required and coordinates are RA-DEC)
        self.DT = DT                                             #Float precision required 
        self.IT = IT                                             #Int precision required 
  
    def coord_int(self, coord1, coord2, time_acs, time_det):
        '''
        Interpolate coordinate timestreams to the detector timestamps.

        Parameters
        ----------
        coord1 : numpy.ndarray
            First coordinate timestream to interpolate.
        coord2 : numpy.ndarray
            Second coordinate timestream to interpolate.
        time_acs : numpy.ndarray
            Timestamps associated with the coordinate timestreams.
        time_det : numpy.ndarray
            Detector timestamps to which the coordinate timestreams are
            interpolated.

        Returns
        -------
        coord1_int : numpy.ndarray
            First coordinate interpolated at the detector timestamps.
        coord2_int : numpy.ndarray
            Second coordinate interpolated at the detector timestamps.
        '''

        coord1_int = interp1d(time_acs, coord1, kind='linear',bounds_error=False,fill_value="extrapolate")
        coord2_int = interp1d(time_acs, coord2, kind= 'linear',bounds_error=False,fill_value="extrapolate")

        return coord1_int(time_det), coord2_int(time_det)

    def sync_data(self):

        '''
        Synchronize detector and coordinate timestreams.

        The timestreams are first restricted to their common time interval.
        If the detector and coordinate timestreams contain different numbers
        of samples, the coordinate, LST, and latitude timestreams are linearly
        interpolated to the detector timestamps. Turnaround flags, when
        provided, are also interpolated and rounded to integer values.

        Returns
        -------
        dettime : numpy.ndarray
            Synchronized detector timestamps.
        det_data : list
            List of synchronized detector timestreams.
        coord1_data : numpy.ndarray
            First coordinate timestream synchronized to the detector
            timestamps.
        coord2_data : numpy.ndarray
            Second coordinate timestream synchronized to the detector
            timestamps.
        lst_data : numpy.ndarray
            Local Sidereal Time timestream synchronized to the detector
            timestamps.
        lat_data : numpy.ndarray
            Latitude timestream synchronized to the detector timestamps.
        turnaround_flags : numpy.ndarray or None
            Synchronized turnaround flags, or None if no turnaround
            flags were provided.
        '''
        #-----------------------------------------------------------------------------------------------


        # Get the data samples whose timestamps are shared with the coordinates timestamps
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

        ctime = ctime[i_c_start:i_c_end]
        self.coord1_data = self.coord1_data[i_c_start:i_c_end]
        self.coord2_data = self.coord2_data[i_c_start:i_c_end]
        self.lst_data = self.lst_data[i_c_start:i_c_end]
        self.lat_data = self.lat_data[i_c_start:i_c_end]
        if(self.remove_turnarounds): self.turnaround_flags = self.turnaround_flags[i_c_start:i_c_end]
        

        #-----------------------------------------------------------------------------------------------


        #Match the number of coordinates samples (coord1, coord2, lat, lst and the turnaround flags) to data samples.
        if(len(ctime) != len(dettime)):
            self.coord1_data, self.coord2_data = self.coord_int(self.coord1_data, self.coord2_data, ctime, dettime)
            self.lst_data, self.lat_data       = self.coord_int(self.lst_data, self.lat_data, ctime, dettime)
            if(self.remove_turnarounds):
                f = interp1d(ctime, self.turnaround_flags, kind='linear',bounds_error=False,fill_value="extrapolate")
                self.turnaround_flags= np.round(f(dettime))

        
        #-----------------------------------------------------------------------------------------------


        
        if(not self.remove_turnarounds): self.turnaround_flags = None

        return dettime, self.det_data, self.coord1_data, self.coord2_data, self.lst_data, self.lat_data, self.turnaround_flags
