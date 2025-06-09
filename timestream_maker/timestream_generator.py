import argparse
from src.load_params import *
from src.hdf5_fcts import * 
from src.scan_fcts import *
from src.astrometry_fcts import *
from astropy.io import fits
import astropy.units as u 
from astropy.coordinates import SkyCoord
from progress.bar import Bar
import pandas as pd
import h5py 
import matplotlib.patches as mpatches
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
from astropy.wcs import WCS

from IPython import embed

def gen_tod(wcs, Map, ybins, xbins, pointing_paths):

    """
    Generate the tod for one array of TIM detectors. 

    Parameters
    ----------
    wcs: astropy.wcs.wcs.WCS
        The wcs used to generate the TODs
    Map: 2d array
        the 2d angular map of the sky in a frequency channel
    ybins: array
        edges of the pixels
    xbins: array
        edges of the pixels
    pointing_paths: list of 2d array
        coordinates of the sky scan path of each pixel, for pixels seeing the same frequency band
    Returns
    -------
    wcs: astropy.wcs.wcs.WCS
        The wcs used to generate the TODs
    map: 2d array
        the sky map used to generate the amplitude TODs
    hist: 2d array
        the reconstructed sky map given the pointing paths 
    norm: 2d array
        the hitmap
    samples: list
        list of the amplitude timestreams of each detectors
    positions_x: list
        list of RA coordinates timestreams of each detectors
    positions_y: list
        list of DEC coordinates timestreams of each detectors 
    """ 

    
    positions_x = np.zeros((len(pointing_paths), len(pointing_paths[0][:,0])))
    positions_y = np.zeros((len(pointing_paths), len(pointing_paths[0][:,0])))
    samples = np.zeros((len(pointing_paths), len(pointing_paths[0][:,0])))
    
    for detector, path in enumerate(pointing_paths):
        
        y_pixel_coords, x_pixel_coords = wcs.world_to_pixel_values(pointing_paths[detector][:,0], pointing_paths[detector][:,1])    
        # Round the positions and convert to integer indices
        x_pixel_coords_rounded = np.round(x_pixel_coords).astype(int)
        y_pixel_coords_rounded = np.round(y_pixel_coords).astype(int)
        # Create a mask for positions within bounds
        valid_mask = (
            (x_pixel_coords_rounded >= 0) & (x_pixel_coords_rounded < hdr['NAXIS1'] - 1) &  # x within bounds
            (y_pixel_coords_rounded >= 0) & (y_pixel_coords_rounded < hdr['NAXIS2'] - 1) )   # y within bounds
        # Initialize the output array with zeros
        values = np.zeros_like(x_pixel_coords_rounded, dtype=float)
        # Assign values from the map for valid positions
        values[valid_mask] = Map[x_pixel_coords_rounded[valid_mask], y_pixel_coords_rounded[valid_mask]]
        samples[detector,:] = np.asarray(values.astype(float))
        positions_x[detector,:] = x_pixel_coords
        positions_y[detector,:] = y_pixel_coords

    norm, edges = np.histogramdd(sample=(positions_x.ravel(), positions_y.ravel()), bins=(xbins,ybins),  )
    hist, edges = np.histogramdd(sample=(positions_x.ravel(), positions_y.ravel()), bins=(xbins,ybins), weights=samples.ravel())

    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", RuntimeWarning)  # Catch all runtime warnings
        hist /= norm  # Perform the division

    return hist, norm, samples, positions_x, positions_y

if __name__ == "__main__":

    '''
    PAR_files/params_strategy.par is a file containing all the modifiable parameters. 
    To generate your TODs: 

    Step 1/3: Generate your observation scan path with python hitmap_1detector.py.py PAR_files/params_strategy.par
    Step 2/3: Generate your detector array with python gen_detectors_arrays.py PAR_files/params_strategy.par
    Step 3/3: Sample the TODs for your detector array following your observation scan path from a simulation with python hitmap_array.py PAR_files/params_strategy.par
    '''
    #------------------------------------------------------------------------------------------
    #load the .par file parameters
    parser = argparse.ArgumentParser(description="strategy parameters",
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    #options
    parser.add_argument('params', help=".par file with params", default = None)
    parser.add_argument('--non_iteractive', help = "deactivate matplotlib", action="store_true")

    args = parser.parse_args()

    if(args.non_iteractive): 
        import matplotlib
        matplotlib.use("Agg")

    P = load_params(args.params)
    #------------------------------------------------------------------------------------------

    #-----------------------------
    #Initiate the parameters

    #The coordinates of the field
    name=P['name_field']
    c=SkyCoord.from_name(name)
    ra = 0 
    rafield = c.ra.value
    dec = c.dec.value

    #load the observer position
    lat = P['latittude']

    #Load the resolution. 
    #if not in params, load it from the map used to generate the TOD. 
    res = P['res']
    if(res is None):
        hdr = fits.getheader(P['path']+P['file'])
        res = (hdr['CDELT1'] * u.Unit(hdr['CUNIT1'])).to(u.deg).value
    
    dt = P['dt']*np.pi/3.14 #Make the timestep non rational to avoid some stripes in the hitmap. 
    spf = int(1/np.round(dt*3600,3)) #sample per frame defined here as the acquisition rate in Hz.

    tod_file=P['path']+f"TOD_{format_duration(P['T_duration'])}.hdf5" #os.getcwd()+'/'+'+P['file'][:-5]+'
    H = h5py.File(tod_file, "a")
    T = H['time']['data'][()]
    LST = H['lst']['data'][()]
    RA_path = H['RA_path']['data'][()]
    DEC_path = H['DEC_path']['data'][()]
    scan_path = np.asarray((RA_path, DEC_path)).T
    H.close()
    #-----------------------------

    #-------------------------------
    #Load the sky simulation from which to generate the TODs from
    simu_sky_path = P['path']+P['file'] #os.getcwd()
    hdr  = fits.getheader(simu_sky_path)
    pix_size = ((hdr['CDELT1']*u.Unit(hdr['CUNIT1']))**2).to(u.sr).value
    hdr['CRVAL1'] = ra 
    hdr['CRVAL2'] = dec
    hdr['CRPIX1'] = hdr['NAXIS1']//2
    hdr['CRPIX2'] = hdr['NAXIS2']//2
    wcs = WCS(hdr, naxis=2) 
    #Create the list of frequency channels of the simulated cube. 
    freqs =( np.arange(hdr['CRVAL3'], hdr['CRVAL3']+hdr['NAXIS3']*hdr['CDELT3'], hdr['CDELT3'])*u.Unit(hdr['CUNIT3']) ).to(u.GHz)
    #Create the binning of the map in pixel coordinates. 
    xbins = np.arange(-0.5, hdr['NAXIS1']+0.5, 1)
    ybins = np.arange(-0.5, hdr['NAXIS2']+0.5, 1)
    #load the angular spectral cube. 
    
    cube = fits.getdata(simu_sky_path)
    #Remove the mean in each map, to wich we are not sensitive. 
    cubemean = np.mean(cube, axis=(1,2)) 
    cube -= cubemean[:, None, None]
    cube *= 1e6*pix_size #conversion MJy/sr to Jy/beam
    #-----------------------------
    det_names_dict = pd.read_csv(P['detectors_name_file'], sep='\t')

    LW = det_names_dict[det_names_dict['XEL'] > 0]
    HW = det_names_dict[det_names_dict['XEL'] < 0]

    for array_name, array, freqs_array in zip( ('HW', 'LW'), (HW, LW),
                                   (freqs[:P['nb_channels_per_array']], freqs[ P['nb_channels_per_array']:P['nb_channels_per_array']*2 ])):

        #------------------------------------------------------------------
        # Group by (XEL, EL), keeping both the group keys and names
        same_offset_groups = array.groupby(['XEL', 'EL'])['Name'].apply(list)

        # Extract (XEL, EL) as a MultiIndex and convert to list
        xel_el_keys = same_offset_groups.index.tolist()

        # Transpose the list of lists of Names
        grouped_lists = same_offset_groups.tolist()
        transposed_groups = list(zip(*grouped_lists))  # One element from each group

        # Combined detectors per group of electromagnetic frequency
        frequency_groups = pd.DataFrame(transposed_groups, columns=pd.MultiIndex.from_tuples(xel_el_keys, names=["XEL", "EL"]))
        #------------------------------------------------------------------

        #------------------------------------------------------------------
        #Load the pointing path for a group of pixel seeing the same electromagnetic frequency
        group = frequency_groups.iloc[0]
        names = group.values
        # Extract XEL and EL from the MultiIndex of the row
        xel = group.index.get_level_values('XEL')
        el = group.index.get_level_values('EL')

        #-------------------------------
        #Generate the scan path of each pixel, as a function of their offset to the center of the arrays. 
        pixel_paths  = genPixelPath(scan_path, el, xel, P['theta'])
        #Generate the pointing on the sky of each pixel. 
        pointing_paths_to_save = [genPointingPath(T, pixel_path, LST, lat, dec, ra) for pixel_path in pixel_paths]
        #-------------------------------

        #------------------------------------------------------------------
        bar = Bar(f'Generate the TODs of the {array_name} array', max=len(frequency_groups))
        #for each frequency,
        for f in range(len(frequency_groups)):
            #----------------------------------------
            #select the detectors
            group = frequency_groups.iloc[f]
            names = group.values
            #----------------------------------------

            #----------------------------------------
            #Select the electromagnetic frequency channel out of which the TODs will be sampled. 
            F = f
            if(array_name=='LW'): F += P['nb_channels_per_array'] 
            Map = cube[F,:,:]
            #----------------------------------------

            #----------------------------------------
            hist, norm, samples, positions_x, positions_y = gen_tod(wcs, Map, ybins, xbins, pointing_paths_to_save)
            #----------------------------------------

            #----------------------------------------
            fig, axs = plt.subplots(1,3, figsize=(12,4), dpi = 200,subplot_kw={'projection': wcs}, sharex=True, sharey=True )
            imgdec = axs[0].imshow(hist, interpolation='nearest', origin='lower', vmin=Map.min(), vmax=Map.max(), cmap='cividis' )
            img = axs[1].imshow(Map, interpolation='nearest', origin='lower', vmin=Map.min(), vmax=Map.max(), cmap='cividis' )
            count = axs[2].imshow(norm, interpolation='nearest', origin='lower', cmap='binary' )
            for ax in (axs[0], axs[1], axs[2]):
                lon = ax.coords[0]
                LAT = ax.coords[1]
                LAT.set_major_formatter('d.d')
                lon.set_major_formatter('d.d')
                lon.set_axislabel('RA')
                LAT.set_axislabel('Dec')
                if(ax is not axs[0]): ax.tick_params(axis='y', labelleft=False)
            plt.subplots_adjust(wspace=0, hspace=0)
            plt.savefig('plot/'+f'freq{freqs[F].value:.0f}GHz_channel_{P["scan"]}_summary_plot.png')
            plt.close()
            #----------------------------------------

            save_tod_in_hdf5(tod_file, names, samples, el, xel, P['detectors_name_file'], F, spf)
            
            bar.next()
        #------------------------------------------------------------------
        bar.finish
        print('')
        
            
    
