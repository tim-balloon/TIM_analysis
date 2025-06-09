import argparse
from src.load_params import load_params, format_duration
from src.scan_fcts import *
from src.astrometry_fcts import *
from src.hdf5_fcts import *
from astropy.io import fits 
import os

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
    lat = P['latittude']

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
    dt = P['dt']*np.pi/3.14 #Make the timestep non rational to avoid some stripes in the hitmap. 
    spf = int(1/np.round(dt*3600,3)) #sample per frame defined here as the acquisition rate in Hz. 
    T = np.arange(0,T_duration,dt) * 3600 #s
    #local sideral time
    LST = np.arange(-T_duration/2,T_duration/2,dt) #hours

    tod_file=P['path']+f'TOD_{format_duration(T_duration)}.hdf5' #os.getcwd()+'/'+'+P['file'][:-5]+'
    #------------------------------------------------------------------------------------------

    #------------------------------------------------------------------------------------------    
    #Generate the scan path for the center of the arrays. 
    if(P['scan']=='loop'):   az, alt, flag = genLocalPath(az_size=P['az_size'], alt_size=P['alt_size'], alt_step=P['alt_step'], acc=P['acc'], scan_v=P['scan_v'], dt=np.round(dt*3600,3))
    if(P['scan']=='raster'): az, alt, flag = genLocalPath_cst_el_scan(az_size=P['az_size'], alt_size=P['alt_size'], alt_step=P['alt_step'], acc=P['acc'], scan_v=P['scan_v'], dt=np.round(dt*3600,3))
    if(P['scan']=='zigzag'): az, alt, flag = genLocalPath_cst_el_scan_crisscross(az_size=P['az_size'], alt_size=P['alt_size'], alt_step=P['alt_step'], acc=P['acc'], scan_v=P['scan_v'], dt=np.round(dt*3600,3))

    scan_path, scan_flag = genScanPath(T, alt, az, flag)
    scan_path = scan_path[scan_flag==1] #Use the scan flag to keep only the constant scan speed part of the pointing. 
    T_trim = T[scan_flag==1]
    LST_trim = LST[scan_flag==1]

    #Generate the pointing on the sky for the center of the arrays
    scan_path_sky, azel = genPointingPath(T_trim, scan_path, LST_trim, lat, dec, ra, azel=True) 
    #----------------------------------------

    #----------------------------------------
    #Plot a scan route
    BS = 10; plt.rc('font', size=BS); plt.rc('axes', titlesize=BS); plt.rc('axes', labelsize=BS); mk = 3; lw=1
    fig, axs = plt.subplots(1,3,figsize=(9,3), dpi=160,)# sharey=True, sharex=True)
    #---
    axradec, ax, axr = axs[0], axs[2], axs[1]
    axradec.plot(az-az.max()/2,alt-alt.max()/2,'cyan')
    axradec.set_xlabel('RA [deg]')
    axradec.set_ylabel('Dec [deg]')
    #---
    heatmap, xedges, yedges = np.histogram2d(scan_path_sky[:,0], scan_path_sky[:,1], bins=int(2/res))
    im = ax.imshow(heatmap.T, origin='lower', cmap='binary',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    cbar = fig.colorbar(im, ax=ax, label='1 detector counts')
    ax.set_xlabel('RA [deg]')
    ax.set_ylabel('Dec [deg]')
    if(contours is not None): ax.plot(contours[:, 1]-rafield, contours[:, 0], c='g' )
    if(contours is not None): axradec.plot(contours[:, 1]-rafield, contours[:, 0]-dec, c='g' )
    #---
    az_unwrapped = (np.radians(azel[:,0]) + np.pi) % (2 * np.pi) - np.pi
    axr.plot(np.degrees(az_unwrapped),azel[:,1],'b')
    axr.set_xlabel('Az [deg]')
    axr.set_ylabel('El [deg]')
    ##---
    patchs = []
    fig.tight_layout()
    plt.savefig(os.getcwd()+'/plot/'+f"scan_route_1det_{P['scan']}_{format_duration(T_duration)}.png")
    plt.close()
    #----------------------------------------

    #latitude timestream
    lat = np.ones(len(LST_trim)) * lat
    #Generate the telescope coordinates and parallactic angle. 

    #RA and Dec
    coord1 = np.radians(scan_path_sky[:,0])
    coord2 = np.radians(scan_path_sky[:,1])

    cos_lat = np.cos(np.radians(lat))
    sin_lat = np.sin(np.radians(lat))

    hour_angle = (LST_trim - ra / 15)*np.pi/12
    index, = np.where(hour_angle<0)
    hour_angle[index] += 2*np.pi
    
    #Parallactic angle
    y_pa = cos_lat*np.sin(hour_angle)
    x_pa = sin_lat*np.cos(coord2)-np.cos(hour_angle)*cos_lat*np.sin(coord2)
    pa = np.arctan2(y_pa, x_pa)

    #Telescope coordinates
    x_tel = coord1*np.cos(pa)-coord2*np.sin(pa)
    y_tel = coord2*np.cos(pa)+coord1*np.sin(pa)
    #----------------------------------------

    #----------------------------------------
    #Save timestreams in a .hdf5 file 
    save_PA(tod_file, np.degrees(pa), spf)
    save_telescope_coord(tod_file, np.degrees(x_tel), np.degrees(y_tel), spf)
    save_lst_lat(tod_file, LST_trim, lat, spf)
    save_az_el(tod_file, azel[:,0], azel[:,1], spf)
    save_time_tod(tod_file, T_trim, spf)
    save_scan_path(tod_file, scan_path_sky, spf, ('RA', 'DEC'))
    save_scan_path(tod_file, scan_path, spf, ('RA_path', 'DEC_path'))
    #-------------------------------------------