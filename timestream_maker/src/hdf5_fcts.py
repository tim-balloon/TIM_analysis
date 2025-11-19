import h5py
import pandas as pd
import os
import numpy as np
from IPython import embed

def to8bit_intprecision(array): 

    float_array = np.array(array, dtype=np.float32)
    # Assuming the original float range is 0.0-1.0

    min = float_array.min()
    if(min<0): float_array -= min

    max = np.abs(float_array).max()
    float_array /= max

    # 1. Scale the values to the 0-255 range
    scaled_array = float_array * 255

    # 2. Clip the values to ensure they are within the 8-bit range (0-255)
    clipped_array = np.clip(scaled_array, 0, 255)

    # 3. Convert to unsigned 8-bit integer type
    downsampled_array = clipped_array.astype(np.uint8)

    return downsampled_array, np.float16(min), np.float16(max)

def save_scan_path(tod_file, scan_path, spf, acquisition_frequency, keys,save="64bytes", compression='gzip'):
    """
    Save the scan path in the .hdf5 format. 

    Parameters
    ----------
    tod_file: string 
        name of the output hdf5 file  
    scan_path_sky: 2d array
        (ra, dec) coordinates timestreams of the center pixel
    spf: int
        the number of samples per frame
    keys: list
        list of names under which the two coordinates are saved
    
    Returns
    -------
    """ 
    H = h5py.File(tod_file, "a")
    for i, (name, coord) in enumerate(zip(keys, (scan_path[:,0],scan_path[:,1]))):

        if name in H: del H[name]   # WARNING: deletes entire detector group
        grp = H.create_group(name)

        if(save=="32bytes"): coord = (coord).astype(np.float32)
        elif(save=="16bytes"): coord = (coord).astype(np.float16)

        if(compression is not None): grp.create_dataset('data', data=coord, compression='gzip', compression_opts=9)
        else: grp.create_dataset('data', data=coord, compression=compression)
        grp.create_dataset('spf', data=spf)
        grp.create_dataset('acquisition frequency', data=acquisition_frequency)
    H.close() 

def save_timestamps(tod_file, T, spf, acquisition_frequency, key, compression='gzip',save="64bytes"):
    '''
    Save the time tod in the .hdf5 format. 

    Parameters
    ----------
    tod_file: string 
        name of the output hdf5 file  
    T: array
        time timestreams
    spf: int
        the number of samples per frame
    key: string
        name under which the timestamps are saved
    Returns
    -------
    '''
    H = h5py.File(tod_file, "a")

    if key in H: del H[key]   # WARNING: deletes entire detector group
    grp = H.create_group(key)

    if(save=="32bytes"): T = (T).astype(np.float32)
    elif(save=="16bytes"): T = (T).astype(np.float16)
    if(compression is not None): grp.create_dataset('data', data=T, compression=compression, compression_opts=9)
    else: grp.create_dataset('data', data=T, compression=compression)

    grp.create_dataset('spf', data=spf)
    grp.create_dataset('acquisition frequency', data=acquisition_frequency)

    H.close()

        
def save_tod_in_hdf5(tod_file, det_names, samples, pixel_offset, pixel_shift, dect_file, F, spf, acquisition_frequency, compression='gzip',save="64bytes"):
    """
    Save the tod for one array of TIM detectors in the .hdf5 format. 

    Parameters
    ----------
    tod_file: string 
        name of the output hdf5 file   
    det_names: list
        list of names for the detectors, same lenght as pixel_offset
    samples: list
        list of amplitude timestreams.  
    pixel_offset: array
        vertical position of each pixel on the array with respect to the center 
    pixel_shift array
        horizontal position of each pixel on the array with respect to the center 
    dect_file: string
        the name of the .csv where the info on each pixel is stored. This function add the frequency band info to the .csv file. 
    F: float
        the frequency band seen by the detectors [GHz]
    spf: int
        the number of samples per frame
    acquisition_frequency: float
        the acquisition frequency of the detectors [Hz]


    Returns
    -------
    """ 
    
    H = h5py.File(tod_file, "a")

    for detector, (offset, shift, name) in enumerate(zip(pixel_offset, pixel_shift, det_names)):
            
        namegrp = f'kid_{name}_roach'
        if namegrp in H: del H[namegrp]   # WARNING: deletes entire detector group
        grp = H.create_group(namegrp)

        grp.create_dataset('spf', data=np.float16(spf))
        grp.create_dataset('pixel_offset_y', data=np.float16(offset))
        grp.create_dataset('pixel_offset_x', data=np.float16(shift))
        grp.create_dataset('acquisition frequency', data=np.float16(acquisition_frequency))
        
        if(save=="32bytes"):
            sample = (samples[detector,:]).astype(np.float32)
            if(compression is not None): 
                grp.create_dataset('data', data=sample, compression=compression, compression_opts=9)
                grp.create_dataset('data_Q', data=sample, compression=compression, compression_opts=9)
            else: 
                grp.create_dataset('data', data=sample, compression=compression)
                grp.create_dataset('data_Q', data=sample, compression=compression)
            

        elif(save=="16bytes"):

            sample = (samples[detector,:]).astype(np.float16)
            if(compression is not None): 
                grp.create_dataset('data', data=sample, compression=compression, compression_opts=9)
                grp.create_dataset('data_Q', data=sample, compression=compression,compression_opts=9)
            else: 
                grp.create_dataset('data', data=sample, compression=compression)
                grp.create_dataset('data_Q', data=sample, compression=compression)


        else: 
            sample = samples[detector,:]
            if(compression is not None): 
                grp.create_dataset('data', data=sample, compression=compression, compression_opts=9)
                grp.create_dataset('data_Q', data=sample, compression=compression,compression_opts=9)
            else: 
                grp.create_dataset('data', data=sample, compression=compression)
                grp.create_dataset('data_Q', data=sample, compression=compression)

    H.close()

    if( not os.path.isfile(dect_file) ):

        with open(dect_file, 'w') as f:
            f.write("Name\tEL\tXEL\tFrequency\n")  # Column headers
            for name in det_names:
                f.write(f"{name}\t\t\t\t\t\t\n")  # Tab-separated values
    #---------------------------

    if(True):
        #Finally, update the detectors file with the central frequency of the detectors
        det_names_dict = pd.read_csv(dect_file, sep='\t')
        mask = det_names_dict["Name"].isin(det_names)
        det_names_dict.loc[mask, 'Frequency'] = F
        det_names_dict.to_csv(dect_file, sep='\t', index=False)
