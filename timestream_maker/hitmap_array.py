import argparse
from src.load_params import load_params, format_duration
from src.scan_fcts import *
from src.astrometry_fcts import *
from src.hdf5_fcts import *
from astropy.io import fits 
import os
import matplotlib.patches as mpatches
from matplotlib.pyplot import cm
from astropy.wcs import WCS
import pickle
import datetime
import time
import astropy.table as tb
from progress.bar import Bar

if __name__ == "__main__":
    '''
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
    #The contour of the field
    contours = P['contours']
    x_cen, y_cen = np.mean(contours[:, 1]), np.mean(contours[:, 0])

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

    #-----------------------------
    #Generate the offset of the pixels with respect to the center of the two arrays, in degrees. 
    pixel_offset_HW, pixel_shift_HW = pixelOffset(P['nb_pixel_HW'], P['offset_HW'], -P['arrays_separation']/2)
    pixel_offset_LW, pixel_shift_LW = pixelOffset(P['nb_pixel_LW'], P['offset_LW'], P['arrays_separation']/2) 
    pixel_offset = np.concatenate((pixel_offset_HW, pixel_offset_LW))
    pixel_shift = np.concatenate((pixel_shift_HW, pixel_shift_LW))
    #-------------------------------

    #-------------------------------
    #Generate the scan path of each pixel, as a function of their offset to the center of the arrays. 
    pixel_paths  = genPixelPath(scan_path, pixel_offset, pixel_shift, P['theta'])
    #Generate the pointing on the sky of each pixel. 
    pointing_paths = [genPointingPath(T, pixel_path, LST, lat, dec, ra) for pixel_path in pixel_paths]
    #Generate the hitmap, using all the detectors. 
    xedges,yedges,hit_map = binMap(pointing_paths,res=res,f_range=P['f_range'],dec=dec,ra=ra) 
    #-------------------------------

    #----------------------------------------
    #Plot a scan route
    BS = 8; plt.rc('font', size=BS); plt.rc('axes', titlesize=BS); plt.rc('axes', labelsize=BS); mk = 3; lw=1
    fig, axs = plt.subplots(1,3,figsize=(9,3), dpi=160,)# sharey=True, sharex=True)
    #---
    axradec, axp, axpix = axs[0], axs[1], axs[2]
    axradec.plot(RA_path-RA_path.max()/2,DEC_path-DEC_path.max()/2,'b')
    axradec.set_xlabel('Az [deg]')
    axradec.set_ylabel('El [deg]')
    #axradec.set_aspect(aspect=1)
    #---
    img = axp.imshow((hit_map), extent=[x_cen-P['f_range'], x_cen+P['f_range'],y_cen-P['f_range'], y_cen+P['f_range'],], 
                    interpolation='nearest', origin='lower', vmin=0, vmax=np.max(hit_map), cmap='binary' )
    if(contours is not None): axp.plot(contours[:, 1], contours[:, 0], c= 'g')
    fig.colorbar(img, ax=axp, label='Counts')
    #CS = axp.contour(np.roll(np.sqrt(hit_map*np.roll(hit_map, 4, axis=1)),-2,axis=1), levels=[0.5*np.max(hit_map)], lw=0.5, origin='lower', linewidths=0.6,
    #extent=[x_cen-f_range, x_cen+f_range,y_cen-f_range, y_cen+f_range,], colors='magenta')
    #axp.clabel(CS, inline=1, fontsize=8, )
    axp.set_xlabel('RA [deg]')
    axp.set_ylabel('Dec [deg]')
    #---
    patchs = []
    axpix.set_title('Pointing Path per pixel')
    idx = np.arange(len(pixel_paths))[::15]
    n = 11
    for i,c in zip(idx, cm.rainbow(np.linspace(0.,1,len(idx)))):
        axpix.scatter(pointing_paths[i][::n,0], pointing_paths[i][::n,1], s=0.1,c=c)
        patch = mpatches.Patch(color=c, label= 'pixel %d'%i); patchs.append(patch); 
    axpix.set_aspect(aspect=1)
    axpix.legend(handles=patchs,frameon=False, bbox_to_anchor=(1,1))
    axpix.set_xlabel('RA [deg]')
    axpix.set_ylabel('Dec [deg]')
    fig.tight_layout()
    plt.savefig(os.getcwd()+'/plot/'+f"scan_route_{P['scan']}_{format_duration(P['T_duration'])}_for_array.png")
    plt.close()
    #----------------------------------------

    #-------------------------------
    wcs = WCS(naxis=2)
    #Define pixel-to-world transformation 
    wcs.wcs.crpix = [hit_map.shape[1] // 2, hit_map.shape[0] // 2]      # Reference pixel
    wcs.wcs.cdelt = [res, res] # Pixel scale in degrees/pixel (RA, Dec)
    wcs.wcs.crval = [ra, dec]          # Reference coordinates (RA, Dec)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]  # Projection type
    d = {'wcs':wcs}
    pickle.dump(d, open(P['wcs_dict'], 'wb'))

    f = fits.PrimaryHDU(hit_map, header=wcs.to_header())
    hdu = fits.HDUList([f])
    hdr = hdu[0].header
    hdr.set("map")
    hdr.set("Datas")
    hdr["BITPIX"] = ("64", "array data type")
    hdr["BUNIT"] = 'counts'
    hdr["DATE"] = (str(datetime.datetime.now()), "date of creation")
    hdu.writeto( f'fits_and_hdf5/hit_map_array_{format_duration(P["T_duration"])}.fits', overwrite=True)
    hdu.close()
    #-------------------------------

    #-------------------------------
    if(False): 
        start = time.time()
        #Load the names of the detectors.
        det_names_dict = pd.read_csv(P['detectors_name_file'], sep='\t')
        same_offset_groups = det_names_dict.groupby(['XEL', 'EL'])['Name'].apply(list).reset_index()
            
        bar = Bar(f'Save the detector poiting paths', max=len(same_offset_groups))
        for i, (pointing_path, offset, shift) in enumerate(zip(pointing_paths, pixel_offset, pixel_shift )):
            names = same_offset_groups.iloc[i]['Name']
            for name in names: 
                save_scan_path(tod_file, pointing_path, spf, (f'RA_{name}', f'DEC_{name}'))
            bar.next()
        bar.finish
        print('')
        print(f"Saved the detector pointing paths in {np.round((time.time() - start),2)}"+'s')
    #-------------------------------