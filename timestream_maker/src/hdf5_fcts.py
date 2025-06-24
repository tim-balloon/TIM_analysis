import h5py
import pandas as pd

def save_scan_path(tod_file, scan_path, spf, names):
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
    lower_spf: bool
        if save the rad dec with an spf lower than the spf of the data, change the name of the group. 
    Returns
    -------
    """ 
    H = h5py.File(tod_file, "a")
    for i, (name, coord) in enumerate(zip(names, (scan_path[:,0],scan_path[:,1]))):
        namegrp = name
        if namegrp not in H: grp = H.create_group(namegrp)
        else:                grp = H[namegrp]
        if('data' in grp): del grp['data'] 
        if('spf' in grp): del grp['spf'] 
        grp.create_dataset('data', data=coord, compression='gzip', compression_opts=9)
        grp.create_dataset('spf', data=spf)
    H.close() 

def save_time_tod(tod_file, T, spf):
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
    Returns
    -------
    '''
    H = h5py.File(tod_file, "a")
    namegrp = f'time'
    if namegrp not in H: grp = H.create_group(namegrp)
    else:                grp = H[namegrp]
    if('data' in grp): del grp['data'] 
    if('spf' in grp): del grp['spf'] 
    grp.create_dataset('data', data=T, compression='gzip', compression_opts=9)
    grp.create_dataset('spf', data=spf)

    H.close()

def save_tod_in_hdf5(tod_file, det_names, samples, pixel_offset, pixel_shift, dect_file, F, spf):
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
    spf: int
        the number of samples per frame
    F: astropy quantity
        the frequency band seen by the detectors

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
        grp.create_dataset('data', data=samples[detector,:],
                            compression='gzip', compression_opts=9)
        grp.create_dataset('spf', data=spf)
        grp.create_dataset('pixel_offset_y', data=offset)
        grp.create_dataset('pixel_offset_x', data=shift)
        grp.create_dataset('frequency', data=F)

    H.close()

    #Finally, update the detectors file with the central frequency of the detectors
    det_names_dict = pd.read_csv(dect_file, sep='\t')
    mask = det_names_dict["Name"].isin(det_names)
    det_names_dict.loc[mask, 'Frequency'] = F
    det_names_dict.to_csv(dect_file, sep='\t', index=False)

def save_lst_lat(tod_file, lst, lat, spf):
    """
    Save the scan path in the .hdf5 format. 

    Parameters
    ----------
    tod_file: string 
        name of the output hdf5 file  
    lst: array
        the local sideral time timestream, for the center of the array. 
    lat: array 
        the latitudetimestream, for the center of the array. 
    spf: int
        the number of samples per frame
    Returns
    -------
    """ 

    H = h5py.File(tod_file, "a")
    for i, (name, coord) in enumerate(zip(('lst', 'lat'), (lst,lat))):
        namegrp = name
        if namegrp not in H: grp = H.create_group(namegrp)
        else:                grp = H[namegrp]
        if('data' in grp): del grp['data'] 
        if('spf' in grp): del grp['spf'] 
        grp.create_dataset('data', data=coord, compression='gzip', compression_opts=9)
        grp.create_dataset('spf', data=spf)
    H.close() 

def save_az_el(tod_file, azimuths, elevations, spf):
    """
    Save the scan path in the .hdf5 format. 

    Parameters
    ----------
    tod_file: string 
        name of the output hdf5 file  
    azimuths: array
        the azimuth timestream of the centre of the array
    elevations: array 
        the latitude timestream of the centre of the array
    spf: int
        the number of samples per frame
    Returns
    -------
    """ 
    H = h5py.File(tod_file, "a")
    for i, (name, coord) in enumerate(zip(('AZ', 'EL'), (azimuths,elevations))):
        namegrp = name
        if namegrp not in H: grp = H.create_group(namegrp)
        else:                grp = H[namegrp]
        if('data' in grp): del grp['data'] 
        if('spf' in grp): del grp['spf'] 
        grp.create_dataset('data', data=coord, compression='gzip', compression_opts=9)
        grp.create_dataset('spf', data=spf)
    H.close() 

def save_telescope_coord(tod_file, x_tel, y_tel, spf):
    """
    Save the scan path in the .hdf5 format. 

    Parameters
    ----------
    tod_file: string 
        name of the output hdf5 file  
    azimuths: array
    elevations: array 
    spf: int
        the number of samples per frame
    Returns
    -------
    """ 
    H = h5py.File(tod_file, "a")
    for i, (name, coord) in enumerate(zip(('xEL', 'EL'), (x_tel,y_tel))):
        namegrp = name
        if namegrp not in H: grp = H.create_group(namegrp)
        else:                grp = H[namegrp]
        if('data' in grp): del grp['data'] 
        if('spf' in grp): del grp['spf'] 
        grp.create_dataset('data', data=coord, compression='gzip', compression_opts=9)
        grp.create_dataset('spf', data=spf)
    H.close() 

def save_PA(tod_file, PA, spf):
    """
    Save the parallactic angle in the .hdf5 format. 

    Parameters
    ----------
    tod_file: string 
        name of the output hdf5 file  
    PA: array
        the parallactic angle timestream 
    spf: int
        the number of samples per frame
    Returns
    -------
    """ 
    H = h5py.File(tod_file, "a")
    for i, (name, coord) in enumerate(zip(('PA',), (PA,))):
        namegrp = name
        if namegrp not in H: grp = H.create_group(namegrp)
        else:                grp = H[namegrp]
        if('data' in grp): del grp['data'] 
        if('spf' in grp): del grp['spf'] 
        grp.create_dataset('data', data=coord, compression='gzip', compression_opts=9)
        grp.create_dataset('spf', data=spf)
    H.close() 
