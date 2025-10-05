import argparse
from src.load_params import load_params, format_duration
from src.scan_fcts import *
from src.astrometry_fcts import *
from src.hdf5_fcts import *
from astropy.io import fits 
import os
from astropy.wcs import WCS
from gen_timestreams import gen_tod

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
    lat = P['latitude']

    #Plot parameter
    f_range = P['f_range']

    #Load the resolution. 
    #if not in params, load it from the map used to generate the TOD. 
    res = P['res']
    if(res is None):
        hdr = fits.getheader(P['path']+P['file'])
        res = (hdr['CDELT1'] * u.Unit(hdr['CUNIT1'])).to(u.deg).value

    #Angle of the rotation to apply to the detector array. 
    theta = np.radians(P['theta'])

    #Load the scan duration and generate the time coordinates with the desired acquisition rate. 
    T_duration = P['T_duration'] 
    T_integration= P['T_integration']
    acquisition_frequency = P['acquisition_frequency']  #sample per frame defined here as the acquisition rate in Hz. 
    dt = 1/acquisition_frequency/3600*np.pi/3.14 #Make the timestep non rational to avoid some stripes in the hitmap. 
    spf = np.round(acquisition_frequency).astype(int)


    #local sideral time
    
    #LST = []
    #times = np.arange(-T_duration/2,T_duration/2,6/60)
    #for t in times: LST = np.concatenate((LST, np.arange(t-T_integration/2,t+T_integration/2,dt)))
    LST = np.arange(-T_duration/2,T_duration/2,dt) #hours
    
    #---
    T = LST*3600 #s
    pps = np.floor(T).astype(int)
    subsecond_ps = T-pps

    tod_file=P['output_path']+f'TOD_{format_duration(T_duration)}.hdf5' #os.getcwd()+'/'+'+P['file'][:-5]+'
    #------------------------------------------------------------------------------------------

    #------------------------------------------------------------------------------------------    
    #Generate the scan path for the center of the arrays. 
    if(P['scan']=='loop'):   az, alt, flag = genLocalPath(az_size=P['az_size'], alt_size=P['alt_size'], alt_step=P['alt_step'], acc=P['acc'], scan_v=P['scan_v'], dt=np.round(dt*3600,3))
    if(P['scan']=='raster'): az, alt, flag = genLocalPath_cst_el_scan(az_size=P['az_size'], alt_size=P['alt_size'], alt_step=P['alt_step'], acc=P['acc'], scan_v=P['scan_v'], dt=np.round(dt*3600,3))
    if(P['scan']=='crisscross'): az, alt, flag = genLocalPath_cst_el_scan_crisscross(az_size=P['az_size'], alt_size=P['alt_size'], alt_step=P['alt_step'], acc=P['acc'], scan_v=P['scan_v'], dt=np.round(dt*3600,3))

    scan_path, scan_flag = genScanPath(T, alt, az, flag)
    #scan_path = scan_path[scan_flag==1] #Use the scan flag to keep only the constant scan speed part of the pointing. 
    #T = T[scan_flag==1]
    #LST = LST[scan_flag==1]

    #Generate the pointing on the sky for the center of the arrays
    scan_path_sky, azel = genPointingPath(T, scan_path, LST, lat, dec, azel=True) 
    #----------------------------------------

    #----------------------------------------
    #Plot a scan route
    BS = 10; plt.rc('font', size=BS); plt.rc('axes', titlesize=BS); plt.rc('axes', labelsize=BS)
    fig, axs = plt.subplots(2,2,figsize=(7,7), dpi=160,)# sharey=True, sharex=True)
    axradec, ax, axr, axc = axs[0,0], axs[1,1], axs[0,1], axs[1,0]
    #---
    axc.scatter(scan_path_sky[:,0], scan_path_sky[:,1], s=0.1,c='r')
    axc.set_xlabel('RA [deg]')
    axc.set_ylabel('Dec [deg]')
    axc.set_aspect('auto')
    #---
    axradec.plot(az-az.max()/2,alt-alt.max()/2,'cyan')
    axradec.set_xlabel('RA [deg]')
    axradec.set_ylabel('Dec [deg]')
    #---
    hitmap, xedges, yedges = np.histogram2d(scan_path_sky[:,0], scan_path_sky[:,1], bins=int(2/res))
    im = ax.imshow(hitmap.T, origin='lower', cmap='binary',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    cbar = fig.colorbar(im, ax=ax, label='1 detector counts')
    ax.set_xlabel('RA [deg]')
    ax.set_ylabel('Dec [deg]')
    ax.set_aspect('auto')
    if(contours is not None):
        ax.plot(contours[:, 1]-rafield, contours[:, 0], c='g' )
        axradec.plot(contours[:, 1]-rafield, contours[:, 0]-dec, c='g' )
        axc.plot(contours[:, 1]-rafield, contours[:, 0], c='g' )
    #---
    az_unwrapped = (np.radians(azel[:,0]) + np.pi) % (2 * np.pi) - np.pi
    axr.plot(np.degrees(az_unwrapped),azel[:,1],'b')
    axr.set_xlabel('Az [deg]')
    axr.set_ylabel('El [deg]')
    #-----
    patchs = []
    fig.tight_layout()
    plt.savefig(os.getcwd()+'/plot/'+f"scan_route_1det_{P['scan']}_{format_duration(T_duration)}.png")
    plt.close()
    #----------------------------------------

    #latitude  timestream
    lat = np.ones(len(LST)) * lat
    #Generate the telescope coordinates and parallactic angle. 

    #RA and Dec
    '''
    coord1 = np.radians(scan_path_sky[:,0])
    coord2 = np.radians(scan_path_sky[:,1])

    cos_lat = np.cos(np.radians(lat))
    sin_lat = np.sin(np.radians(lat))

    hour_angle = (LST - ra / 15)*np.pi/12
    index, = np.where(hour_angle<0)
    hour_angle[index] += 2*np.pi
    
    #Parallactic angle
    y_pa = cos_lat*np.sin(hour_angle)
    x_pa = sin_lat*np.cos(coord2)-np.cos(hour_angle)*cos_lat*np.sin(coord2)
    pa = np.arctan2(y_pa, x_pa)

    #Telescope coordinates
    x_tel = coord1*np.cos(pa)-coord2*np.sin(pa)
    y_tel = coord2*np.cos(pa)+coord1*np.sin(pa)
    '''
    #----------------------------------------

    #----------------------------------------
    #Save timestreams in a .hdf5 file 
    #save_PA(tod_file, np.degrees(pa), spf)
    #save_telescope_coord(tod_file, np.degrees(x_tel), np.degrees(y_tel), spf)
    save_scan_path(tod_file, np.array((LST, lat)).T, spf,acquisition_frequency, ('data_lst', 'data_lat'))
    save_scan_path(tod_file, azel, spf,acquisition_frequency,('data_AZ', 'data_EL'))
    save_scan_path(tod_file, scan_path_sky, spf, acquisition_frequency,('data_RA', 'data_DEC'))
    save_scan_path(tod_file, scan_path,     spf, acquisition_frequency,('data_RA_path', 'data_DEC_path'))
    save_timestamps(tod_file, T, spf, acquisition_frequency,'data_time')
    save_timestamps(tod_file, pps, spf, acquisition_frequency,'data_pps')  
    save_timestamps(tod_file, subsecond_ps, spf, acquisition_frequency,'data_subsecond_ps')  
    save_timestamps(tod_file, scan_flag, spf, acquisition_frequency,'data_turnaround_flags')
    #-------------------------------------------

    #----------------------------------------
    #Create the coordinates sampled differently, to test the synchronization with data in Namap.
    
    lat = P['latitude']

    acquisition_frequency_prime = P['acquisition_frequency_coords']  #sample per frame defined here as the acquisition rate in Hz. 
    dt_coords = 1/acquisition_frequency_prime/3600*np.pi/3.14 #Make the timestep non rational to avoid some stripes in the hitmap. 
    spf_prime = np.round(acquisition_frequency_prime).astype(int)

    LST_prime = np.arange(-T_duration/2,T_duration/2,dt_coords) #hours
    T_prime = LST_prime * 3600 #s np.arange(0,T_duration,dt_coords)
    pps = np.floor(T_prime).astype(int)
    subsecond_ps = T_prime-pps

    if(P['scan']=='loop'):   az, alt, flag = genLocalPath(az_size=P['az_size'], alt_size=P['alt_size'], alt_step=P['alt_step'], acc=P['acc'], scan_v=P['scan_v'], dt=np.round(3600*dt_coords,3))
    if(P['scan']=='raster'): az, alt, flag = genLocalPath_cst_el_scan(az_size=P['az_size'], alt_size=P['alt_size'], alt_step=P['alt_step'], acc=P['acc'], scan_v=P['scan_v'], dt=np.round(3600*dt_coords,3))
    if(P['scan']=='crisscross'): az, alt, flag = genLocalPath_cst_el_scan_crisscross(az_size=P['az_size'], alt_size=P['alt_size'], alt_step=P['alt_step'], acc=P['acc'], scan_v=P['scan_v'], dt=np.round(3600*dt_coords,3))

    scan_path_prime, scan_flag = genScanPath(T_prime, alt, az, flag)
    scan_path_sky_prime, azel = genPointingPath(T_prime, scan_path_prime, LST_prime, lat, dec, azel=True) 
    lat = np.ones(len(LST_prime)) * lat
    
    #spf_prime = spf
    #acquisition_frequency_prime = acquisition_frequency

    save_scan_path(tod_file, np.array((LST_prime, lat)).T, spf_prime, acquisition_frequency_prime,('lst', 'lat'))
    save_scan_path(tod_file, azel, spf_prime,acquisition_frequency_prime,('AZ', 'EL'))
    save_scan_path(tod_file, scan_path_sky_prime, spf_prime,acquisition_frequency_prime, ('RA', 'DEC'))
    save_scan_path(tod_file, scan_path_prime,     spf_prime,acquisition_frequency_prime, ('RA_path', 'DEC_path'))
    save_timestamps(tod_file, T_prime, spf_prime, acquisition_frequency_prime,'coords_time')
    save_timestamps(tod_file, pps, spf_prime, acquisition_frequency_prime,'coords_pps')  
    save_timestamps(tod_file, subsecond_ps, spf_prime, acquisition_frequency_prime,'coords_subsecond_ps')  
    save_timestamps(tod_file, scan_flag, spf_prime,acquisition_frequency_prime, ('turnaround_flags'))


    #Extra piece of code to measure the TOD file size with data in MB.
    #-------------------------------------------------------
    if(False):
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
        I = 3
        pixel_offset_EL, pixel_offset_xEL= pixelOffset(I, 0.0145 ,0)
        pixel_offsets = pixels_rotations(pixel_offset_EL, pixel_offset_xEL, P['theta'])
        pointing_paths = [genPointingPath(T, scan_path, LST, lat[0], dec, offsets) for offsets in pixel_offsets]

        det_names_dict = pd.read_csv(P['detectors_name_file'], sep='\t')
        # Extract the Name column as a list
        names = det_names_dict["Name"].tolist()
        # Split into chunks of 10
        chunks = [names[i:i+I] for i in range(0, len(names), I+1)]
        # Example: print them
        print('')
        for idx, group in enumerate(chunks):

            if(idx>10): continue
            print(idx, group)
            Map = cube[idx,:,:]
            #----------------------------------------
            hist, norm, samples, positions_x, positions_y = gen_tod(wcs, Map, hdr, ybins, xbins, pointing_paths, T=T)
            #----------------------------------------
            save_tod_in_hdf5(tod_file, group, samples, pixel_offset_EL, pixel_offset_xEL, P['detectors_name_file'], (freqs[idx].value,), spf, acquisition_frequency)
    #-------------------------------------------------------



    