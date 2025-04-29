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

    tod_file=os.getcwd()+'/'+P['path']+'TOD_'+P['file'][:-5]+'.hdf5'

    for group in range(len(same_offset_groups)):

        #------------------------------------------------------------------
        xpix_list = []
        ypix_list = []
        samples = []
        #Load the sky timestreams (from strategy.py)
        H = h5py.File(tod_file, "a")
        for id, d in enumerate(same_offset_groups.iloc[group]['Name']): 
            f = H[f'kid_{d}_roach']
            samples.append(f['noisy_data'][()]) 
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
        plt.imshow(hist/norm, origin='lower')
        plt.savefig(f'noise_grp{group}.png')
        plt.show()
    embed()