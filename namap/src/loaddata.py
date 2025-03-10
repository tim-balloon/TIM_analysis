import pygetdata as gd
import numpy as np
from scipy.interpolate import interp1d
import os
import astropy.table as tb
from IPython import embed
import src.detector as det 
import h5py


def loaddata(file, field, num_frames=None, first_frame=None):
    '''
    Equivalent to d.getdata()
    file : the name of the hdf5 file
    field: the field to be loaded
    num_frame: the number of frames to load, with N=spf samples in each frame.
    first_frame: the first frame to load. 
    ''' 
    H = h5py.File(file, "a")
    f = H[field]
    if(('spf' in f.keys()) and (num_frames is not None) and (first_frame is not None)):
      spf = f['spf'][()]
      data = f['data'][first_frame*spf:(first_frame+num_frames)*spf]
    else: 
      data = f['data'][()]
    H.close()
    return data

def loadspf(file, field):
    '''
    Equivalent to d.getdata()
    file : the name of the hdf5 file
    field: the field to be loaded
    num_frame: the number of frames to load, with N=spf samples in each frame.
    first_frame: the first frame to load. 
    ''' 
    H = h5py.File(file, "a")
    f = H[field]
    if('spf' in f.keys()): spf = f['spf'][()]
    else: spf = None
    H.close()
    return spf

class data_value():
    
    '''
    Class for reading the values of the TODs (detectors and coordinates) from a DIRFILE
    '''

    def __init__(self, det_path, det_name, coord_path, coord1_name, \
                 coord2_name, experiment, lst_file_type, lat_file_type, hwp_file_type,
                 startframe, numframes, roach_number=None, telemetry=False):

        '''
        For BLAST-TNG the detector name is given as kid_# where # is 1,2,3,4,5
        The number is then converted to the equivalent letters that are coming from 
        the telemetry name
        '''
        self.det_path = det_path                    #Path of the detector dirfile
        self.det_name = det_name                    #Detector name to be analyzed
        self.coord_path = coord_path                #Path of the coordinates dirfile
        self.coord1_name = coord1_name              #Coordinates 1 name, e.g. RA or AZ
        self.coord2_name = coord2_name              #Coordinates 2 name
        self.experiment = experiment                #Experiment to be analyzed

        self.lst_file_type = lst_file_type          #LST DIRFILE datatype
        self.lat_file_type = lat_file_type          #LAT DIRFILE datatype

        self.hwp_file_type = hwp_file_type          #HWP DIRFILE datatype

        self.startframe = startframe        #Starting frame to be analyzed
        self.numframes = numframes           #Ending frame to be analyzed

        if self.startframe < 100:
            self.bufferframe = int(0)  #Buffer frames to be loaded before and after the starting and ending frame
        else:
            self.bufferframe = int(100)

        self.telemetry = telemetry
        self.roach_number = roach_number

    def gdload(self, filepath, file, file_type):

        '''
        Return the values of the DIRFILE as a numpy array
        
        filepath: path of the DIRFILE to be read
        file: name of the value to be read from the dirfile, e.g. detector name or
              coordinate name
        file_type: data type conversion string for the DIRFILE data
        '''
        if np.size(file) == 1: 
            d = gd.dirfile(filepath, gd.RDONLY)
            if file_type is not None:
                gdtype = self.conversion_type(file_type)
            else:
                gdtype = gd.FLOAT64
            if self.experiment.lower()=='blast-tng':
                num = self.endframe-self.startframe+2*self.bufferframe
                first_frame = self.startframe-self.bufferframe
            else:
                H = h5py.File(filepath, "a")
                num = H['nframes'][()]
                H.close()
                first_frame = 0
            if isinstance(file, str):
                value = d.getdata(file, gdtype, num_frames = num, first_frame=first_frame)

            else:
                value = d.getdata(file[0], gdtype, num_frames = num, first_frame=first_frame)

            return np.asarray(values)
        else:
            #d = gd.dirfile(filepath, gd.RDONLY)
            gdtype = self.conversion_type(file_type)
            values = np.array([])

            for i in range(len(file)):
                if i == 0:
                    values = d.getdata(file[i], gdtype,num_frames = d.nframes)
                else:
                    values = np.vstack((values, d.getdata(file[i], gdtype,num_frames = d.nframes)))
            return values

    def load(filepath, fields, num_frames=None, first_frame=None):

        '''
        Return the values of the DIRFILE as a numpy array
        
        filepath: path of the DIRFILE to be read
        file: name of the value to be read from the dirfile, e.g. detector name or
              coordinate name
        file_type: data type conversion string for the DIRFILE data
        ##Already suppose that all data have same frequency sample... so return the mean spf
        '''
        spf = []
        for i,field in enumerate(fields):
            if i == 0: values = loaddata(filepath, field, num_frames, first_frame)
            else:      values = np.vstack((values, loaddata(filepath, field, num_frames, first_frame)))
            spf.append(loadspf(filepath, field, ))    

        return np.asarray(values), (np.asarray(spf)).mean()

    def values(self):

        '''
        Function to return the timestreams for detector and coordinates
        '''
        num = self.numframes#+2*self.bufferframe
        first_frame = self.startframe
        kid_num  = self.det_name

        #---------------------------------------------------------------------------------
        if( self.experiment.lower() == 'blast-tng'):

            if self.telemetry:
                list_conv = [['A', 'B'], ['D', 'E'], ['G', 'H'], ['K', 'I'], ['M', 'N']]
                det_I_string_list = ['kid' + list_conv[kid - 1][0] + '_roachN' for kid in kid_num]
                det_Q_string_list = ['kid' + list_conv[kid - 1][1] + '_roachN' for kid in kid_num]
            else:
                det_I_string_list = ['I_kid' + f'{kid}' + '_roach' for kid in kid_num] 
                det_Q_string_list = ['Q_kid' + f'{kid}' + '_roach' for kid in kid_num] 

            I_data, spf_data = data_value.load(self.det_path, det_I_string_list, num, first_frame)
            Q_data, spf_data = data_value.load(self.det_path, det_Q_string_list, num, first_frame)
            kidutils = det.kidsutils()
            det_data = kidutils.KIDmag(I_data, Q_data)
        #---------------------------------------------------------------------------------
        else:
            det_data = loaddata(self.det_path, self.det_name, num, first_frame)
            spf_data = loaddata(self.det_path, self.det_name)
        #---------------------------------------------------------------------------------
        if self.coord2_name.lower() == 'dec':
            if self.experiment.lower()=='blast-tng': coord2 = 'DEC'
            else: coord2 = 'dec'
        
        elif self.coord2_name.lower() == 'y': coord2 = 'y_stage'
        else:
            if self.experiment.lower()=='blast-tng': coord2 = 'EL'
            else: coord2 = 'el'

        coord2_data = loaddata(self.coord_path, coord2, num, first_frame)
        spf_coord = loadspf(self.coord_path, coord2, )
        #---------------------------------------------------------------------------------
        if self.coord1_name.lower() == 'ra':
            if self.experiment.lower()=='blast-tng': coord1 = 'RA'
            else: coord1 = 'ra'
        elif self.coord1_name.lower() == 'x': coord1 = 'x_stage'
        else:
            if self.experiment.lower()=='blast-tng': coord1 = 'AZ'
            else:  coord1 = 'az'
        
        coord1_data = loaddata(self.coord_path, coord1, num, first_frame)
        #---------------------------------------------------------------------------------
        if self.hwp_file_type is not None:
            hwp_data = loaddata(self.coord_path, 'pot_hwpr')
            hwp_spf = loadspf(self.coord_path, 'pot_hwpr')
        else:
            hwp_data = 0.
            hwp_spf = 0
        #---------------------------------------------------------------------------------
        if self.lat_file_type is not None and self.lst_file_type is not None:
            
            lat = loaddata(self.coord_path, 'lat',num, first_frame)
            lst = loaddata(self.coord_path, 'lst',num, first_frame)
            lat_spf = loadspf(self.coord_path, 'lst')

            return det_data, coord1_data, coord2_data, hwp_data, lst, lat, spf_data, spf_coord,hwp_spf,lat_spf
        
        else:
            return det_data, coord1_data, coord2_data, hwp_data, None, None, spf_data, spf_coord, hwp_spf, 0

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

    '''
    This class is designed to extract the frames of interest from the complete timestream and 
    sync detector and coordinates timestream given a different sampling of the two
    '''

    def __init__(self, det_data, det_sample_frame, det_fs, coord1_data, coord2_data, 
                 coord_fs, coord_sample_frame,
                 startframe, endframe, experiment, 
                 lst_data, lat_data, lstlatfreq, lstlat_sample_frame, offset =None, 
                 roach_number=None, roach_pps_path=None, 
                 hwp_data=0., hwp_fs=None, hwp_sample_frame=None, xystage=False):
        
        self.det_data = det_data                                #Detector data timestream
        self.det_fs = det_fs                                    #Detector frequency sampling
        self.det_sample_frame = det_sample_frame                #Detector samples in each frame of the timestream
        self.coord1_data = coord1_data                          #Coordinate 1 data timestream
        self.coord_fs = coord_fs                                #Coordinates frequency sampling
        self.coord_sample_frame = coord_sample_frame            #Coordinates samples in each frame of the time stream
        self.coord2_data = coord2_data                          #Coordinate 2 data timestream
        self.startframe = startframe                            #Start frame
        self.endframe = endframe                                #End frame
        self.experiment = experiment                            #Experiment to be analyzed, right now BLASTPol or BLAST-TNG
        self.lst_data = lst_data                                #LST timestream (if correction is required and coordinates are RA-DEC)
        self.lat_data = lat_data                                #LAT timestream (if correction is required and coordinates are RA-DEC)
        self.lstlatfreq = lstlatfreq                            #LST-LAT sampling frequency (if correction is required and coordinates are RA-DEC)
        self.lstlat_sample_frame = lstlat_sample_frame          #LST-LAT samples per frame (if correction is required and coordinates are RA-DEC)
        if roach_number is not None:
            self.roach_number = roach_number                    #If BLAST-TNG is the experiment, this gives the number of the roach used to read the detector
        else:
            self.roach_number = roach_number
        self.roach_pps_path = roach_pps_path                    #Pulse per second of the roach used to sync the data
        self.offset = offset                                    #Time offset between detector data and coordinates
        self.hwp_data = hwp_data
        if hwp_fs is not None:
            self.hwp_fs = hwp_fs
        else:
            self.hwp_fs = hwp_fs
        if hwp_sample_frame is not None:
            self.hwp_sample_frame = hwp_sample_frame
        else:
            self.hwp_sample_frame = hwp_sample_frame

        self.xystage=xystage                                   #Flag to check if the coordinates data are coming from an xy stage scan                       #Flag to check if the coordinates data are coming from an xy stage scan
        
        if self.startframe < 100:
            self.bufferframe = int(0)
        else:
            self.bufferframe = int(100)
        

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

        end_det_frame = self.endframe#+self.bufferframe
        first_frame = self.startframe#-self.bufferframe
        interval = self.endframe-self.startframe

        if self.experiment.lower() == 'blast-tng':
            #d = gd.dirfile(self.roach_pps_path)
            
            ctime_mcp = loaddata(self.roach_pps_path, 'time', first_frame=first_frame, num_frames=interval) 
            ctime_usec = loaddata(self.roach_pps_path, 'time_usec', first_frame=first_frame, num_frames=interval) 

            if self.xystage is True: sample_ctime = 100
            else: sample_ctime = self.coord_sample_frame
            ctime_start = ctime_mcp+ctime_usec/1e6+0.2

            if self.offset is not None: ctime_mcp += self.offset/1000.

            ctime_start = ctime_mcp[0]
            ctime_end = ctime_mcp[-1]
            
            coord1 = self.coord1_data #[int(self.bufferframe*self.coord_sample_frame):int(self.bufferframe*self.coord_sample_frame+interval*self.coord_sample_frame)]
            coord2 = self.coord2_data
                                    
            if self.xystage is True:
                freq_array = np.append(0, np.cumsum(np.repeat(1/self.coord_sample_frame, self.coord_sample_frame*interval-1)))
                coord1time = ctime_start+freq_array
                coord2time = coord1time.copy()
            else:
                if self.coord_sample_frame != 100:
                    freq_array = np.append(0, np.cumsum(np.repeat(1/self.coord_sample_frame, self.coord_sample_frame*interval-1)))
                    coord1time = ctime_start+freq_array
                    coord2time = coord1time.copy()
                else:
                    coord1time = ctime_mcp.copy()
                    coord2time = ctime_mcp.copy()

            if telemetry:

                kidutils = det.kidsutils()
                frames = np.array([first_frame, end_det_frame], dtype='int')
                dettime, pps_bins = kidutils.det_time(self.roach_pps_path, self.roach_number, frames, \
                                                      ctime_start, ctime_mcp[-1], self.det_fs)
                
                coord1int = interp1d(coord1time, coord1, kind='linear')
                coord2int = interp1d(coord2time, coord2, kind= 'linear')

                idx_roach_start = np.argmin(np.abs(dettime - ctime_start), axis=1)
                idx_roach_end =   np.argmin(np.abs(dettime - ctime_end),   axis=1)
                data_resampled = []

                if self.det_data.ndim == 1:  # If it's a 1D array
                    a = kidutils.interpolation_roach(self.det_data, pps_bins[pps_bins>350], self.det_fs)
                    data_resampled = a[idx_roach_start[0]:idx_roach_end[0]]
                else:
                    for i in range(len(self.det_data)): 
                        #self.det_data[i] = kidutils.interpolation_roach(self.det_data[i], pps_bins[i][pps_bins[i]>350], self.det_fs)
                        a = kidutils.interpolation_roach(self.det_data[i], pps_bins[i][pps_bins[i]>350], self.det_fs)
                        b = a[idx_roach_start[0]:idx_roach_end[0]]
                        data_resampled.append(b)

                self.det_data = np.asarray(data_resampled)
                dettime = dettime[:,idx_roach_start[0]:idx_roach_end[0]]
                
            else:
                if len(np.shape(self.det_data)) == 1: dettime = ctime_start+np.append(0, np.cumsum(np.repeat(1/self.det_fs, len(self.det_data))))
                else:                                 dettime = ctime_start+np.append(0, np.cumsum(np.repeat(1/self.det_fs, len(self.det_data[0]))))

            index1_list = np.argmin(np.abs(dettime - coord1time[0]), axis=1)
            index2_list = np.argmin(np.abs(dettime - coord1time[-1]), axis=1)

            coord1_inter_list = []
            coord2_inter_list = []
            dettime_list = []

            for index1, index2, detT in zip(index1_list, index2_list, dettime):
                T = detT[index1+200:index2-200]
                coord1_inter_list.append( coord1int(T) )
                coord2_inter_list.append( coord2int(T) )
                dettime_list.append(T)

            Data = []
            for i, (index1, index2) in enumerate(zip(index1_list, index2_list)):
                    
                if self.det_data.ndim == 1: data = self.det_data
                else: data = self.det_data[i,:]
                Data.append( data[index1+200:index2-200] )

            self.det_data = Data
        
        if isinstance(self.hwp_data, np.ndarray):
            hwp = self.hwp_data[self.bufferframe*self.coord_sample_frame:self.bufferframe*self.coord_sample_frame+\
                                interval*self.coord_sample_frame]
            freq_array = np.append(0, np.cumsum(np.repeat(1/self.hwp_sample_frame, self.hwp_sample_frame*interval-1)))
            hwptime = ctime_start+freq_array
            hwp_interpolation = interp1d(hwptime, hwp, kind='linear')
            hwp_inter = hwp_interpolation(dettime)
            del hwptime
            del hwp

        else:

            hwp_inter =  [[0] * len(sublist) for sublist in coord1_inter_list] #np.zeros_like(coord1_inter)

        del coord1time
        del coord2time
        del coord1
        del coord2

        if self.lat_data is not None and self.lat_data is not None:
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
            #if np.size(np.shape(self.det_data)) > 1:
            #    return (dettime, self.det_data, coord1_inter, coord2_inter, hwp_inter, lst_inter, lat_inter)
            #else:
            return (dettime, self.det_data, coord1_inter_list, coord2_inter_list,  hwp_inter, lst_inter, lat_inter)
        else:     return (dettime, self.det_data, coord1_inter_list, coord2_inter_list, hwp_inter, None, None)
            #if np.size(np.shape(self.det_data)) > 1:
            #else: return (dettime, self.det_data, coord1_inter, coord2_inter, hwp_inter, None, None)

class xsc_offset():
    
    '''
    class to read star camera offset files
    '''

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
\
        return xsc_file[2], xsc_file[3]

class det_table():

    '''
    Class to read detector tables. For BLASTPol can convert also detector names using another table
    '''

    def __init__(self, name, experiment, pathtable):

        self.name = name
        self.experiment = experiment
        self.pathtable = pathtable

    def loadtable(self):
        det_off = np.zeros((np.size(self.name), 2))
        noise = np.ones(np.size(self.name))
        grid_angle = np.zeros(np.size(self.name))
        pol_angle_offset = np.zeros(np.size(self.name))
        resp = np.zeros(np.size(self.name))

        if self.experiment.lower() == 'blastpol':

            for i in range(np.shape(det_off)[0]):
                if self.name[i][0].lower() == 'n':            
                    path = self.pathtable+'bolo_names.txt'
                    name_table = np.loadtxt(path, skiprows = 1, dtype = str)

                    index, = np.where(self.name[i].upper() == name_table[:,1])
                    real_name = name_table[index, 0]
                else:
                    real_name = self.name

                path = self.pathtable+'bolotable.tsv'
                btable = tb.Table.read(path, format='ascii.basic')
                index, = np.where(btable['Name'] == real_name[0].upper())
                det_off[i, 1] = btable['EL'][index]/3600.       #Conversion from arcsec to degrees
                det_off[i, 0] = btable['XEL'][index]/3600.     #Conversion from arcsec to degrees          

                noise[i] = btable['WhiteNoise'][index]
                grid_angle[i] = btable['Angle'][index]
                pol_angle_offset[i] = btable['Chi'][index]
                resp[i] = btable['Resp.'][index]*-1.


            return det_off, noise, grid_angle, pol_angle_offset, resp

