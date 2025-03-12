import pygetdata as gd
import numpy as np
from IPython import embed
import src.detector as det
import src.loaddata as ld
import src.detector as tod
import src.mapmaker as mp
import src.pointing as pt  
import copy
from astropy import wcs 
import os
from src.gui import MapPlotsGroup
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from astropy.nddata import Cutout2D
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import h5py
import argparse
from scipy.interpolate import interp1d
import scipy.signal as sgn

import tracemalloc

def load_params(path, force_pysides_path = ''):

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

def map2d(data=None, coord1=None, coord2=None, crval=None, ctype=None, pixnum=None, telcoord=False, cdelt=None, \
          crpix=None, projection=None, xystage=False, det_name=None, idx=None):


    """
    Plot the map out of the data timestreams.     
    Parameters
    ----------   
    Returns
    -------
    """    

    intervals = 3   

    if telcoord is False:        
        if xystage is False:
            position = SkyCoord(crval[0], crval[1], unit='deg', frame='icrs')

            size = (pixnum[1], pixnum[0])     # pixels

            cutout = Cutout2D(data, position, size, wcs=projection)
            proj = cutout.wcs
            mapdata = cutout.data
        else:
            masked = np.ma.array(data, mask=(np.abs(data)<1))
            mapdata = masked
            proj = 'rectilinear'

    else:
        idx_xmin = crval[0]-cdelt*pixnum[0]/2   
        idx_xmax = crval[0]+cdelt*pixnum[0]/2
        idx_ymin = crval[1]-cdelt*pixnum[1]/2
        idx_ymax = crval[1]+cdelt*pixnum[1]/2

        proj = None

        idx_xmin = np.amax(np.array([np.ceil(crpix[0]-1-pixnum[0]/2), 0.], dtype=int))
        idx_xmax = np.amin(np.array([np.ceil(crpix[0]-1+pixnum[0]/2), np.shape(data)[1]], dtype=int))

        if np.abs(idx_xmax-idx_xmin) != pixnum[0]:
            if idx_xmin != 0 and idx_xmax == np.shape(data)[1]:
                idx_xmin = np.amax(np.array([0., np.shape(data)[1]-pixnum[0]], dtype=int))
            if idx_xmin == 0 and idx_xmax != np.shape(data)[1]:
                idx_xmax = np.amin(np.array([pixnum[0], np.shape(data)[1]], dtype=int))

        idx_ymin = np.amax(np.array([np.ceil(crpix[1]-1-pixnum[1]/2), 0.], dtype=int))
        idx_ymax = np.amin(np.array([np.ceil(crpix[1]-1+pixnum[1]/2), np.shape(data)[0]], dtype=int))

        if np.abs(idx_ymax-idx_ymin) != pixnum[1]:
            if idx_ymin != 0 and idx_ymax == np.shape(data)[0]:
                idx_ymin = np.amax(np.array([0., np.shape(data)[0]-pixnum[1]], dtype=int))
            if idx_ymin == 0 and idx_ymax != np.shape(data)[0]:
                idx_ymax = np.amin(np.array([pixnum[1], np.shape(data)[0]], dtype=int))

        mapdata = data[idx_ymin:idx_ymax, idx_xmin:idx_xmax]
        crpix[0] -= idx_xmin
        crpix[1] -= idx_ymin

        w = wcs.WCS(naxis=2)
        w.wcs.crpix = crpix
        w.wcs.cdelt = cdelt
        w.wcs.crval = crval
        w.wcs.ctype = ["TLON-TAN", "TLAT-TAN"]
        proj = w

    levels = np.linspace(0.5, 1, intervals)*np.amax(mapdata)
    
    axis = plt.subplot(111, projection=proj)

    if telcoord is False:
        if ctype == 'RA and DEC':
            ra = axis.coords[0]
            dec = axis.coords[1]
            ra.set_axislabel('RA (deg)')
            dec.set_axislabel('Dec (deg)')
            dec.set_major_formatter('d.ddd')
            ra.set_major_formatter('d.ddd')
        
        elif ctype == 'AZ and EL':
            az = axis.coords[0]
            el = axis.coords[1]
            az.set_axislabel('AZ (deg)')
            el.set_axislabel('EL (deg)')
            az.set_major_formatter('d.ddd')
            el.set_major_formatter('d.ddd')
        
        elif ctype == 'CROSS-EL and EL':
            xel = axis.coords[0]
            el = axis.coords[1]
            xel.set_axislabel('xEL (deg)')
            el.set_axislabel('EL (deg)')
            xel.set_major_formatter('d.ddd')
            el.set_major_formatter('d.ddd')

        elif ctype == 'XY Stage':
            axis.set_title('XY Stage')
            axis.set_xlabel('X')
            axis.set_ylabel('Y')

    else:
        ra_tel = axis.coords[0]
        dec_tel = axis.coords[1]
        ra_tel.set_axislabel('YAW (deg)')
        dec_tel.set_axislabel('PITCH (deg)')
        ra_tel.set_major_formatter('d.ddd')
        dec_tel.set_major_formatter('d.ddd')

    if telcoord is False:
        im = axis.imshow(mapdata, origin='lower', cmap=plt.cm.viridis)
        #axis.contour(mapdata, levels=levels, colors='white', alpha=0.5)
    else:
        im = axis.imshow(mapdata, origin='lower', cmap=plt.cm.viridis)
        #axis.contour(mapdata, levels=levels, colors='white', alpha=0.5)
    plt.colorbar(im, ax=axis)
    title = 'Map '+ idx + ' det'+f'{det_name}'     
    plt.title(title)

    path = os.getcwd()+'/plot/'+f'{det_name}'+'_'+idx+'.png'

    plt.savefig(path)

if __name__ == "__main__":

    '''
    Download the .hdf5: scp yournetid@cc-login.campuscluster.illinois.edu:/projects/ncsa/caps/TIM_analysis/timestreams/master.hdf5 .

    To run: python namap_main.py params_namap.par

    Left to be done:
        Load the detectors and the detectors offests --> xsc_offset and det_table
        Explain the time system in ld.sync_data ?
        Do we need the telemetry option ? 
        hwp ? 
        buffer frame ?
        Implement TIM settings
        Do we need 'CROSS-EL and EL' ? 
        Implement pointing offset correction
        Implement respons correction
        parallactic angle ? pol angle ?
        Implement noise detectors
        Implement spectral axis 
    '''

    #-------------------------------------------------------------------------------------------------------------------------
    #Iinitialization
    parser = argparse.ArgumentParser(description='NaMap. A naive mapmaker written in Python for BLASTPol and BLAST-TNG', \
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    #options
    parser.add_argument('params', help=".par file with params", default = None)
    args = parser.parse_args()

    P = load_params(args.params)

    if P['ctype'] == 'RA and DEC':
        coord1 = str('RA')
        coord2 = str('DEC')
        xystage = False
    elif P['ctype'] == 'AZ and EL':
        coord1 = str('AZ')
        coord2 = str('EL')
        xystage = False
    elif P['ctype'] == 'CROSS-EL and EL':
        coord1 = str('xEL')
        coord2 = str('EL')
        xystage = False
    elif P['ctype'] == 'XY Stage':
        coord1 = str('X')
        coord2 = str('Y')
        xystage = True

    experiment = P['experiment']
    if experiment.lower() == 'blastpol': xystage=False
    filepath = P['hdf5_file']
    kid_num = P['kid_num']
    num_frames , first_frame = P['num_frames'], P['first_frame']
    hwp, lat, lst = P['hwp'], P['lat'], P['lst']
    offset = P['time_offset']
    telemetry = P['telemetry']
    highpassfreq = P['highpassfreq']
    polynomialorder = P['polynomialorder']
    despike_bool = P['despike']
    sigma,prominence = P['sigma'],P['prominence']
    convolution, std = P['gaussian_convolution'], P['std']
    #-------------------------------------------------------------------------------------------------------------------------

    if(not P['coadd']):

        if(P['det_offset'] is None): det_off   = np.zeros((np.size(kid_num),2))
        if(not P['noise_det']):      noise_det = np.ones(np.size(kid_num))
        grid_angle = np.zeros(np.size(kid_num))
        resp = np.ones(np.size(kid_num))

        for id, dect in enumerate(kid_num):

            dect = (dect,) #Work with list comprehension

            #-------------------
            #Load the data
            dataload = ld.data_value(filepath, dect, filepath, coord1, coord2, experiment, lst, lat, hwp, first_frame, num_frames, xystage, telemetry)
            det_data, coord1_data, coord2_data, hwp_data, lst, lat, spf_data, spf_coord, hwp_spf, lat_spf = dataload.values()
            #--------------------
            #Synchronise the data
            zoomsyncdata = ld.frame_zoom_sync(det_data, spf_data, spf_data, coord1_data, coord2_data, spf_coord, spf_coord, 
                                first_frame, num_frames+first_frame, experiment, lst,lat,lat_spf, lat_spf,
                                offset, dect, filepath, hwp_data, hwp_spf, hwp_spf, xystage)
            timemap, detslice, coord1slice, coord2slice, hwpslice, lstslice, latslice = zoomsyncdata.sync_data()
            #--------------------

            if coord1.lower() == 'xel': coord1slice = coord1slice*np.cos(np.radians(coord2slice)) #Needs to be modify !

            if(hwp is not None and experiment.lower() == 'blastpol'): hwpslice = (hwpslice-0.451)*(-360.) #Needs to be modify !

            if P['detector_table'] is not None:
                #--------------------
                #Needs to be modify !
                dettable = ld.det_table(dect, experiment, P['detector_table']) 
                det_off, noise_det, grid_angle, pol_angle_offset, resp = dettable.loadtable()
                #--------------------
            else: 
                det_off = np.zeros((np.size(dect),2))
                noise_det = np.ones(np.size(dect))
                grid_angle = np.zeros(np.size(dect))
                pol_angle_offset = np.zeros(np.size(dect))
                resp = np.ones(np.size(dect))

            if(P['correction']): 
                #--------------------
                #Needs to be modify !
                if(P['pointing_table'] is not None):               
                    xsc_file = ld.xsc_offset(P['pointing_table'], first_frame, num_frames+first_frame)
                    xsc_offset = xsc_file.read_file()
                else: xsc_offset = np.zeros(2)
                corr = pt.apply_offset(coord1slice, coord2slice, datatype_coord, \
                                    xsc_offset, det_offset = det_off, lst = lstslice, \
                                    lat = latslice)
                coord1slice, coord2slice = corr.correction()
                #--------------------

            #elif(coord1.lower() == 'ra'): coord1slice = coord1slice*15. #Conversion between hours to degree ##!!!!
            
            if P['telescope_coordinate'] or P['I_only'] is False:
                #--------------------
                #Needs to be modify !
                parallactic = np.zeros_like(coord1slice)
                if np.size(np.shape(detslice)) == 1:
                    tel = pt.utils(coord1slice/15., coord2slice, lstslice, latslice)
                    parallactic = tel.parallactic_angle()
                else:
                    if np.size(np.shape(coord1slice)) == 1:
                        tel = pt.utils(coord1slice/15., coord2slice, lstslice, latslice)
                        parallactic = tel.parallactic_angle()
                    else:
                        for j in range(np.size(np.shape(detslice))):
                            tel = pt.utils(coord1slice[j]/15., coord2slice[j], \
                                        lstslice, latslice)
                            parallactic[j,:] = tel.parallactic_angle()
                #--------------------
            else:
                if np.size(np.shape(detslice)) == 1:
                    parallactic = 0.
                else:
                    if np.size(np.shape(coord1slice)) == 1:
                        parallactic = 0.
                    else:
                        parallactic = np.zeros_like(detslice)


            #---------------------------------
            #Clean the TOD
            det_tod = tod.data_cleaned(detslice, spf_data,highpassfreq, dect,
                                    polynomialorder, despike_bool, sigma, prominence)
            cleaned_data = det_tod.data_clean()
            #--------------------
            #Needs to be modify ! How to implement the respons ? 
            if np.size(resp) > 1:
            
                if experiment.lower() == 'blast-tng':
                    cleaned_data = np.multiply(cleaned_data, np.reshape(1/resp, (np.size(1/resp), 1)))
                else:
                    cleaned_data = np.multiply(cleaned_data, np.reshape(resp, (np.size(resp), 1)))
            else:
                if experiment.lower() == 'blast-tng':
                    cleaned_data /= resp
                else:
                    cleaned_data *= resp
            #--------------------

            #--------------------
            #Needs to be modify
            pol_angle = np.radians(parallactic+2*hwpslice+(grid_angle-2*pol_angle_offset)) 
            #if np.size(np.shape(coord1slice)) != 1: pol_angle = np.reshape(pol_angle, np.size(pol_angle))
            pol_angle = np.zeros_like(cleaned_data)
            #--------------------

            #Create the maps
            maps = mp.maps(P['ctype'], P['crpix'], P['cdelt'], P['crval'], (cleaned_data,), coord1slice, coord2slice, \
                        convolution, std, P['I_only'], pol_angle=pol_angle, noise=noise_det, \
                        telcoord = P['telescope_coordinate'], parang=parallactic)

            maps.wcs_proj()
            proj = maps.proj
            map_value = maps.map2d()
            
            w = maps.w
            x_min_map = np.floor(np.amin(w[:,0]))
            y_min_map = np.floor(np.amin(w[:,1]))
            index1, = np.where(w[:,0]<0)
            index2, = np.where(w[:,1]<0)
            if np.size(index1) > 1: crpix1_new  = (P['crpix'][0]-x_min_map)
            else: crpix1_new = copy.copy(P['crpix'][0])
            if np.size(index2) > 1: crpix2_new  = (P['crpix'][1]-y_min_map)
            else: crpix2_new = copy.copy(P['crpix'][1])
            
            crpix_new = np.array([crpix1_new, crpix2_new])

            #---------------------------------
            #Plot the maps

            if P['I_only']:
                map2d(data = map_value, coord1=coord1slice, coord2=coord1slice, crval=P['crval'], ctype=P['ctype'], pixnum=P['pixnum'], \
                    telcoord=P['telescope_coordinate'], crpix=crpix_new, cdelt=P['cdelt'], projection=proj, xystage=xystage, \
                    det_name=dect, idx='I')
            else:
                idx_list = ['I', 'Q', 'U']

                for i in range(len(idx_list)):
                    map2d(map_value, coord1=coord1slice, coord2=coord1slice,  crval=P['crval'], ctype=P['ctype'], pixnum=P['pixnum'], \
                        telcoord=P['telescope_coordinate'], crpix=crpix_new, cdelt=P['cdelt'], projection=proj, xystage=xystage, \
                        det_name=dect, idx=idx_list[i])

            plt.show()
            #---------------------------------

    #-------------------------------------------------------------------------------------------------------------------------
    else: 
        dataload = ld.data_value(filepath, kid_num, filepath, coord1, coord2, experiment, lst, lat, hwp, first_frame, num_frames, kid_num, telemetry)
        det_data, coord1_data, coord2_data, hwp_data, lst, lat, spf_data, spf_coord, hwp_spf, lat_spf = dataload.values()
        
        if experiment.lower() == 'blastpol': roach_number=None
        else: roach_number = kid_num
        
        zoomsyncdata = ld.frame_zoom_sync(det_data, spf_data, spf_data, coord1_data, coord2_data, spf_coord, spf_coord,
                                        first_frame, num_frames+first_frame,  experiment,
                                        lst,lat,lat_spf, lat_spf, offset, roach_number, filepath, 
                                        hwp_data, hwp_spf, hwp_spf, xystage)
        
        timemap, detslice, coord1slice, coord2slice, hwpslice, lstslice, latslice = zoomsyncdata.sync_data()

        if coord1.lower() == 'xel': coord1slice = [x * np.cos(np.radians(y)) for x,y in zip(coord1slice, coord2slice)] 

        if(hwp is not None and experiment.lower() == 'blastpol'): hwpslice = (hwpslice-0.451)*(-360.)

        if P['detector_table'] is not None:
            dettable = ld.det_table(P['detector_table'], experiment, P['detector_table'])
            det_off, noise_det, grid_angle, pol_angle_offset, resp = dettable.loadtable()
        else:
        
            det_off = np.zeros((np.size(kid_num),2))
            noise_det = np.ones(np.size(kid_num))
            grid_angle = np.zeros(np.size(kid_num))
            pol_angle_offset = np.zeros(np.size(kid_num))
            resp = np.ones(np.size(kid_num))

        parallactic = [np.zeros_like(slice) for slice in detslice]

        det_tod = tod.data_cleaned(detslice, spf_data,highpassfreq, kid_num,
                                polynomialorder, despike_bool, sigma, prominence)
        cleaned_data = det_tod.data_clean()
        
        pol_angle = [np.zeros_like(slice) for slice in cleaned_data]

        maps = mp.maps(P['ctype'], P['crpix'], P['cdelt'], P['crval'], cleaned_data, coord1slice, coord2slice, \
                    convolution, std, P['I_only'], pol_angle=pol_angle, noise=noise_det, \
                    telcoord = P['telescope_coordinate'], parang=parallactic)

        maps.wcs_proj()
        proj = maps.proj
        map_value = maps.map2d()

        w = maps.w

        crpix_new = P['crpix'] #np.array([crpix1_new, crpix2_new])

        if P['I_only']:
            map2d(data = map_value, coord1=coord1slice, coord2=coord1slice, crval=P['crval'], ctype=P['ctype'], pixnum=P['pixnum'], \
                telcoord=P['telescope_coordinate'], crpix=crpix_new, cdelt=P['cdelt'], projection=proj, xystage=xystage, \
                det_name=0, idx='I')
        plt.show()