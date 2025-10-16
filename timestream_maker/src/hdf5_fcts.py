import h5py
import pandas as pd
import os
import numpy as np

def save_scan_path(tod_file, scan_path, spf, acquisition_frequency, keys,save="64bit"):
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
        namegrp = name
        if namegrp not in H: grp = H.create_group(namegrp)
        else:                grp = H[namegrp]
        if('data' in grp): del grp['data'] 
        if('spf' in grp): del grp['spf'] 
        if('acquisition frequency' in grp): del grp['acquisition frequency'] 

        if(save=="32bit"): coord = (coord).astype(np.float32)
        elif(save=="16bit"): coord = (coord).astype(np.float16)

        grp.create_dataset('data', data=coord, compression='gzip', compression_opts=9)
        grp.create_dataset('spf', data=spf)
        grp.create_dataset('acquisition frequency', data=acquisition_frequency)
    H.close() 

def save_timestamps(tod_file, T, spf, acquisition_frequency, key, save="64bit"):
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
    namegrp = key
    if namegrp not in H: grp = H.create_group(namegrp)
    else:                grp = H[namegrp]
    if('data' in grp): del grp['data'] 
    if('spf' in grp): del grp['spf'] 
    if('acquisition frequency' in grp): del grp['acquisition frequency'] 

    if(save=="32bit"): T = (T).astype(np.float32)
    elif(save=="16bit"): T = (T).astype(np.float16)
    
    grp.create_dataset('data', data=T, compression='gzip', compression_opts=9)
    grp.create_dataset('spf', data=spf)
    grp.create_dataset('acquisition frequency', data=acquisition_frequency)

    H.close()

def save_tod_in_hdf5(tod_file, det_names, samples, pixel_offset, pixel_shift, dect_file, F, spf, acquisition_frequency, save="64bit"):
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
        if namegrp not in H: grp = H.create_group(namegrp)
        else:                grp = H[namegrp]
        if('data' in grp): del grp['data'] 
        if('spf' in grp): del grp['spf'] 
        if('pixel_offset_y' in grp): del grp['pixel_offset_y'] 
        if('pixel_offset_x' in grp): del grp['pixel_offset_x'] 
        if('frequency' in grp): del grp['frequency'] 
        if('acquisition frequency' in grp): del grp['acquisition frequency'] 
        if(save=="32bit"):
            sample = (samples[detector,:]).astype(np.float32)
            grp.create_dataset('data', data=sample, compression='gzip', compression_opts=9)
            grp.create_dataset('spf',   data=np.float32(spf))                  
            grp.create_dataset('pixel_offset_y', data=np.float32(offset))
            grp.create_dataset('pixel_offset_x', data=np.float32(shift))
            grp.create_dataset('acquisition frequency', data=np.float32(acquisition_frequency))
        elif(save=="16bit"):
            sample = (samples[detector,:]).astype(np.float16)
            grp.create_dataset('data', data=sample, compression='gzip', compression_opts=9)
            grp.create_dataset('spf', data=np.float16(spf))
            grp.create_dataset('pixel_offset_y', data=np.float16(offset))
            grp.create_dataset('pixel_offset_x', data=np.float16(shift))
            grp.create_dataset('acquisition frequency', data=np.float16(acquisition_frequency))
        else: 
            sample = samples[detector,:]
            grp.create_dataset('data', data=sample, compression='gzip', compression_opts=9)
            grp.create_dataset('spf', data=spf)
            grp.create_dataset('pixel_offset_y', data=offset)
            grp.create_dataset('pixel_offset_x', data=shift)
            grp.create_dataset('acquisition frequency', data=acquisition_frequency)
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
