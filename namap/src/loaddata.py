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
        #---------------------------------------------------------------------------------
        
        coord1_data = data_value.loaddata(self.det_path, f'{self.coord1_name}', num, first_frame) 
        coord2_data = data_value.loaddata(self.det_path, f'{self.coord2_name}', num, first_frame) 
        spf_coord = data_value.loadspf(self.det_path, self.coord2_name, )

        #---------------------------------------------------------------------------------
        lat = data_value.loaddata(self.det_path, 'lat',num, first_frame)
        lst = data_value.loaddata(self.det_path, 'lst',num, first_frame)
        lst_lat_spf = data_value.loadspf(self.det_path, 'lst')

        return det_data, coord1_data, coord2_data, lst, lat, spf_data, spf_coord, lst_lat_spf

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

 