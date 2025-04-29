import sys
sys.path.append('../simulations')
import argparse
from strategy import load_params
import numpy as np
from pylab import *
from progress.bar import Bar
import pandas as pd
from itertools import islice
from IPython import embed
import h5py
import os
import astropy.units as u 
from scipy import interpolate
from astropy.wcs import WCS
from astropy.io import fits
import pickle
from progress.bar import Bar
import time




if __name__ == "__main__":
    '''
    Generate noise TOD. 
    The noise TOD are frequency, detector, and time independant.
    The noise TODs are gaussian. 
    The noise TODs for pixels seeing the same beam (at different frequency bands) are correlated. 
    '''
    #------------------------------------------------------------------------------------------
    #load the .par file parameters
    parser = argparse.ArgumentParser(description="strategy parameters",
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    #options
    parser.add_argument('params', help=".par file with params", default = None)
    args = parser.parse_args()
    P = load_params(args.params)
    #------------------------------------------------------------------------------------------

    det_names_dict = pd.read_csv(P['detectors_name_file'], sep='\t')
    #Each pixel with the same offset sees the same beam, but in different frequency band. 
    same_offset_groups = det_names_dict.groupby(['Frequency'])['Name'].apply(list).reset_index()

    # #Load the WCS
    dwcs = pickle.load( open(P['wcs_dict'], 'rb'))
    wcs = dwcs['wcs']
    xbins = np.arange(-0.5, wcs.pixel_shape[0]+0.5, 1)
    ybins = np.arange(-0.5, wcs.pixel_shape[1]+0.5, 1)
    embed()

    tod_file=os.getcwd()+'/'+P['path']+'TOD_'+P['file'][:-5]+'.hdf5'

    for key in ('data', 'noisy_data'):
        
        cube_per_pix = []

        for group in range(len(same_offset_groups)):

            #------------------------------------------------------------------
            xpix_list = []
            ypix_list = []
            samples = []
            #Load the sky timestreams (from strategy.py)
            H = h5py.File(tod_file, "a")
            for id, d in enumerate(same_offset_groups.iloc[group]['Name']): 
                f = H[f'kid_{d}_roach']
                samples.append(f[key][()]) 
                f = H[f'kid_{d}_RA']
                ra = f['data'][()]
                f = H[f'kid_{d}_DEC']
                dec = f['data'][()]
                y_pixel_coords, x_pixel_coords = wcs.world_to_pixel_values(ra,dec)    
                xpix_list.append(x_pixel_coords)
                ypix_list.append(y_pixel_coords)
            H.close()

            xpix_list = np.asarray(xpix_list)
            ypix_list = np.asarray(ypix_list)
            samples =np.asarray( samples )
            #------------------------------------------------------------------

            norm, edges = np.histogramdd(sample=(xpix_list.ravel(), ypix_list.ravel()), bins=(xbins,ybins),  )
            hist, edges = np.histogramdd(sample=(xpix_list.ravel(), ypix_list.ravel()), bins=(xbins,ybins), weights=samples.ravel())    
            cube_per_pix.append(hist/norm) #Jy/pix

        pixel_sr = (wcs.wcs.cdelt[0]* np.pi/180/3600 )**2 #solid angle of the pixel in sr 
        cube_per_sr =  np.asarray(cube_per_pix) / pixel_sr * 1.e-6 #The 1.e-6 is used to go from Jy to My, final result in MJy/sr
        

        filename = P["output_path"] + '/' +  P["run_name"] + '_' + key + '.fits'
        print('Write '+filename+'...')
        
        if os.path.exists(P['output_path']) == False:
            print('Create '+P['output_path'])
            os.makedirs(P['output_path'])
        
        f = fits.PrimaryHDU(cube_per_sr, header=wcs.to_header())
        hdu = fits.HDUList([f])
        hdr = hdu[0].header
        hdr.set("cube")
        hdr.set("Datas")
        hdr["BITPIX"] = ("64", "array data type")
        hdr["BUNIT"] = 'Jy/sr'
        hdr["DATE"] = (str(datetime.datetime.now()), "date of the creation")
        hdu.writeto(filename, overwrite=True)
        hdu.close()

    
