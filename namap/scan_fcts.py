import numpy as np
from TIM_scan_strategy import *
from IPython import embed
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, ICRS


def hitsPerSqdeg(total_hits, area, res):
    """
    hits per sqare degree 
    Parameters
    ----------
    total_hits: 2d array
        hit map   
    area: float
        area in deg2
    res: float
        resolution in same units as area

    Returns
    -------
    hitsPerSqdeg: 2d array
        map of hits per square degree 
    """ 
    return total_hits/(area/res**2)

def timeFractionAbove(hmap, level):
    """
    fraction of time above a level of hits
    Parameters
    ----------
    hmap: 2d array
        hit map   
    level: float
        level of hits
    Returns
    -------
    timeFractionAbove: float
        fraction of the hitmaps above a level of hits
    """ 
    hits = hmap.flatten()
    return np.sum(hits[hits>level])/np.sum(hits)

def genLocalPath_cst_el_scan_zigzag(az_size = 1, alt_size = 1, alt_step=0.02, acc = 0.05, scan_v=0.05, dt= 0.01):

    """
    Function that generates the local scaning pattern.
    Currently can only generate closed loop    
    Parameters
    ----------
    az_size: float
        azimuth angular size, in degrees   
    alt_size: float
        altitude angular size, in degrees
    alt_step: float
        step in altitude angle, in degrees
    acc: float
        acceleration in second angle 
    scan_v: float
        angular speed of the scan, in deg/sec
    dt: float
        time step in second angle 
    Returns
    -------
    az: array
        azimuth scan path coordinates, in degrees
    alt: array
        altitude scan path coordinates, in degrees
    flag: array
        constant scan speed part.
    scan_eff: array
        scan efficiency: the ratio between the constant scan speed part and not constant scan speed part
    t: array
        time during the scan, in second angle
    """ 

    #----
    #Compute Number of Vertical Steps 
    ver_N = int(alt_size//alt_step)
    #Compute Time for Scan and Turns
    scan_time = az_size/scan_v #Time required to cover the full azimuth range at scan_v
    turn_time = 2*scan_v/acc #Time required to perform a turn (deceleration, reversal, acceleration)
    #Generate Azimuth Acceleration Pattern (a):
    #The motion consists of acceleration, constant velocity, and deceleration, forming a symmetric back-and-forth oscillation in azimuth.
    a = np.concatenate((np.ones(int(turn_time/dt))*acc,np.zeros(int(scan_time/dt))))
    a = np.concatenate((a,-1*a))
    #The sequence is repeated for each altitude step (ver_N times).
    a = np.tile(a,ver_N)
    #Generate Altitude Acceleration Pattern
    acc_alt = alt_step/(turn_time/2)**2   
    #The altitude changes slightly during turns, using a small acceleration.
    #A similar acceleration pattern is applied to a2 to control altitude transitions.
    cycles_per_scan = 1
    oscillation = np.tile(
        np.concatenate([
            np.ones(int(turn_time / dt / 2)) * acc_alt,
            np.ones(int(turn_time / dt / 2)) * -acc_alt
        ]), cycles_per_scan
    )
    # Ensure no extra oscillation at the ends of azimuth scan
    a3 = np.concatenate((oscillation, np.zeros(int(scan_time / dt))))
    a3 = np.concatenate((a3, a3))  # No altitude change on the leftward scan
    a3 = np.tile(a3, ver_N)

    #Compute Azimuth (az) and Altitude (alt) Coordinates:
    #Computed by integrating acceleration to get velocity, then integrating velocity to get position.
    v = np.cumsum(a)*dt-scan_v
    az = np.cumsum(v)*dt
    v2 = np.cumsum(a3)*dt
    alt  = np.cumsum(v2)*dt

    flag = np.where(a==0,1,0) #constant scan speed part
    t = np.arange(0,len(a))*dt

    return az,alt,flag  

def genLocalPath_cst_el_scan(az_size = 1, alt_size = 1, alt_step=0.02, acc = 0.05, scan_v=0.05, dt= 0.01):

    """
    Function that generates the local scaning pattern.
    Currently can only generate closed loop    
    Parameters
    ----------
    az_size: float
        azimuth angular size, in degrees   
    alt_size: float
        altitude angular size, in degrees
    alt_step: float
        step in altitude angle, in degrees
    acc: float
        acceleration in second angle 
    scan_v: float
        angular speed of the scan, in deg/sec
    dt: float
        time step in second angle 
    Returns
    -------
    az: array
        azimuth scan path coordinates, in degrees
    alt: array
        altitude scan path coordinates, in degrees
    flag: array
        constant scan speed part.
    scan_eff: array
        scan efficiency: the ratio between the constant scan speed part and not constant scan speed part
    t: array
        time during the scan, in second angle
    """ 

    #----
    #Compute Number of Vertical Steps 
    ver_N = int(alt_size//alt_step)

    #Compute Time for Scan and Turns
    scan_time = az_size/scan_v #Time required to cover the full azimuth range at scan_v
    turn_time = 2*scan_v/acc #Time required to perform a turn (deceleration, reversal, acceleration).

    #Generate Azimuth Acceleration Pattern (a):
    #The motion consists of acceleration, constant velocity, and deceleration, forming a symmetric back-and-forth oscillation in azimuth.
    a = np.concatenate((np.ones(int(turn_time/dt))*acc,np.zeros(int(scan_time/dt))))
    a = np.concatenate((a,-1*a))
    #The sequence is repeated for each altitude step (ver_N times).
    a = np.tile(a,ver_N)
    #Generate Altitude Acceleration Pattern
    acc_alt = alt_step/(turn_time/2)**2   

    #The altitude changes slightly during turns, using a small acceleration.
    #A similar acceleration pattern is applied to a2 to control altitude transitions.
    cycles_per_scan = 1#int(scan_time / (2 * turn_time))  # Number of oscillations per scan
    oscillation = np.tile(
        np.concatenate([
            np.ones(int(turn_time / dt / 2)) * acc_alt,
            np.ones(int(turn_time / dt / 2)) * -acc_alt
        ]), cycles_per_scan
    )
    # Ensure no extra oscillation at the ends of azimuth scan
    a3 = np.concatenate((oscillation, np.zeros(int(scan_time / dt))))
    #a3 = np.concatenate((a3, -1 * a3))  # Repeat for downward scan
    a3 = np.concatenate((a3, np.zeros_like(a3)))  # No altitude change on the leftward scan
    a3 = np.tile(a3, ver_N)
    #a3 = np.tile(a3,ver_N)

    #Compute Azimuth (az) and Altitude (alt) Coordinates:
    #Computed by integrating acceleration to get velocity, then integrating velocity to get position.
    v = np.cumsum(a)*dt-scan_v
    az = np.cumsum(v)*dt
    v2 = np.cumsum(a3)*dt
    alt  = np.cumsum(v2)*dt

    flag = np.where(a==0,1,0) #constant scan speed part
    t = np.arange(0,len(a))*dt
    return az,alt,flag  

def genLocalPath(az_size = 1, alt_size = 1, alt_step=0.02, acc = 0.05, scan_v=0.05, dt= 0.01):
    """
    Function that generates the local scaning pattern.
    Currently can only generate closed loop    
    Parameters
    ----------
    az_size: float
        azimuth angular size, in degrees   
    alt_size: float
        altitude angular size, in degrees
    alt_step: float
        step in altitude angle, in degrees
    acc: float
        acceleration in second angle 
    scan_v: float
        angular speed of the scan, in deg/sec
    dt: float
        time step in second angle 
    Returns
    -------
    az: array
        azimuth scan path coordinates, in degrees
    alt: array
        altitude scan path coordinates, in degrees
    flag: array
        constant scan speed part.
    scan_eff: array
        scan efficiency: the ratio between the constant scan speed part and not constant scan speed part
    t: array
        time during the scan, in second angle
    """ 
    #----
    #Compute Number of Vertical Steps 
    ver_N = int(alt_size//alt_step)

    #Compute Time for Scan and Turns
    scan_time = az_size/scan_v #Time required to cover the full azimuth range at scan_v
    turn_time = 2*scan_v/acc #Time required to perform a turn (deceleration, reversal, acceleration).

    #Generate Azimuth Acceleration Pattern (a):
    #The motion consists of acceleration, constant velocity, and deceleration, forming a symmetric back-and-forth oscillation in azimuth.
    a = np.concatenate((np.ones(int(turn_time/dt))*acc,np.zeros(int(scan_time/dt))))
    a = np.concatenate((a,-1*a))
    #The sequence is repeated for each altitude step (ver_N times).
    a = np.tile(a,ver_N)
    #Generate Altitude Acceleration Pattern
    acc_alt = alt_step/(turn_time/2)**2
    #The altitude changes slightly during turns, using a small acceleration.
    #A similar acceleration pattern is applied to a2 to control altitude transitions.
    a2 = np.concatenate((np.ones(int(turn_time/dt/2))*acc_alt,-1*np.ones(int(turn_time/dt/2))*acc_alt,np.zeros(int(scan_time/dt))))
    a2 = np.concatenate((np.tile(a2,ver_N),np.tile(-1*a2,ver_N)))

    flag = np.where(a==0,1,0) #constant scan speed part

    t = np.arange(0,len(a))*dt

    #Compute Azimuth (az) and Altitude (alt) Coordinates:
    #Computed by integrating acceleration to get velocity, then integrating velocity to get position.
    v = np.cumsum(a)*dt-scan_v
    az = np.cumsum(v)*dt

    v2 = np.cumsum(a2)*dt
    alt  = np.cumsum(v2)*dt

    #Measures the fraction of time spent scanning at constant speed vs. turning.
    scan_eff = np.sum(flag)/len(az)
    
    return az,alt,flag  

def genScanPath(T, alt, az, flag, plot=False):
    """    
    Function that generates the pointing coordinates vs time.

    Parameters
    ----------
    T: array
        time stream
    az: array
        azimuth scan path coordinates, in degrees
    alt: array
        altitude scan path coordinates, in degrees
    flag: array
        constant scan speed part. 
    Returns
    -------
    coor: 2d array
        coordinates in degrees
    flag: array
        constant scan speed part. 
    """ 

    coor = np.zeros((len(T),2))

    idx = np.int_(np.fmod(T,len(alt)/100)*100)
    
    coor[:,0] = az[idx]-np.mean(az)
    coor[:,1] = alt[idx]-np.mean(alt)
    flag      = flag[idx]
    
    return coor,flag

def pixelOffset(pixel_num, pixel_pitch, pixel_array_separation):
    """
    Function that  gernerates the pixel offset vs pointing center
    Parameters
    ----------
    pixel_num: int
        number of spatial pixels
    pixel_pitch: float
        spatial distance between adjacent pixels in degrees
    Returns
    -------
    yoffsets: array
        the pixel offset vs pointing center, in degrees
    """ 
    yoffsets = (np.arange(0,pixel_num)-pixel_num/2)*pixel_pitch
#     offsets = np.vstack((np.zeros(pixel_num),yoffsets)).T
    xoffsets = np.ones(len(yoffsets)) * pixel_array_separation
    
    return yoffsets, xoffsets

def genPixelPath(pointing_path, pixel_offset, pixel_shift, theta):
    """
    Function that gernerates the pointing time stream for each pixel
    Parameters
    ----------
    pointing_path: 2d array
        coordinates timestream of the pointing
    pixel_offset: float
        spatial distance between adjacent pixels in degrees
    theta: float
        angle in degree
    Returns
    -------
    pixel_path: nd array
        the coordinates timestream of the pointing of each pixel, in degrees
    """ 
    pixel_path = []
    for pixel, xpixel in zip(pixel_offset, pixel_shift):  
        pixel_w_time = np.array([xpixel * np.cos(theta) - pixel * np.sin(theta), 
                                 xpixel * np.sin(theta) + pixel * np.cos(theta)])  # Apply rotation
        #pixel_w_time = np.append( pixel*np.sin(theta), pixel*np.cos(theta))
        pixel_path.append(pointing_path+pixel_w_time) 
    return pixel_path

def genPointingPath(T, scan_path, HA, lat, dec,ra, azel=False):

    """
    Function that takes local paths and generates the pointing on sky vs time.
    Parameters
    ----------
    T: array
        coordinates timestream of the pointing
    pixel_offset: float
        spatial distance between adjacent pixels in degrees
    Returns
    -------
    pixel_path: nd array
        the coordinates timestream of the pointing of each pixel, in degrees
    """     
    alt = elevationAngle(dec,lat,HA)+np.radians(scan_path[:,1])
    azi = azimuthAngle(dec,lat,HA)+np.radians(scan_path[:,0]) 
    dec_point = declinationAngle(np.degrees(azi), np.degrees(alt), lat)
    ha_point  = hourAngle(       np.degrees(azi), np.degrees(alt), lat)
    path = np.vstack((np.degrees(ha_point-HA*np.pi/12),np.degrees(dec_point))).T
    path[:,0] += ra
    azel_path = np.vstack((np.degrees(elevationAngle(dec,lat,HA)),np.degrees(azimuthAngle(dec,lat,HA)))).T

    if(azel): return path, azel_path
    else: return path

def genPointingPath_mod(scan_path, HA, lat, dec,ra, azel=False):
    """
    Function that takes local paths and generates the pointing on sky vs time.
    Parameters
    ----------
    T: array
        coordinates timestream of the pointing
    pixel_offset: float
        spatial distance between adjacent pixels in degrees
    Returns
    -------
    pixel_path: nd array
        the coordinates timestream of the pointing of each pixel, in degrees
    """    
    HAr = HA * np.pi / 12 # Hour angle [rad]
    decr = np.radians(dec)  # DEC offset due to scanning [rad]
    latr = np.radians(lat)             # Observer latitude [rad]
    # Precompute trigonometric terms
    sin_dec = np.sin(decr)
    cos_dec = np.cos(decr)
    sin_lat = np.sin(latr)
    cos_lat = np.cos(latr)
    cos_HA = np.cos(HAr)
    sin_HA = np.sin(HAr)

    el_namap = np.arcsin(sin_dec*sin_lat+cos_lat*cos_dec*cos_HA) 
    az_namap = np.arccos((sin_dec-sin_lat*np.sin(el_namap))/(cos_lat*np.cos(el_namap)))
    index, = np.where(sin_HA>0)
    az_namap[index] = 2*np.pi - az_namap[index]
    el_tot = el_namap + np.radians(scan_path[:, 1])
    az_tot = az_namap + np.radians(scan_path[:, 0])

    sin_el_tot = np.sin(el_tot)
    sin_az_tot = np.sin(az_tot)
    cos_el_tot = np.cos(el_tot)
    cos_az_tot = np.cos(az_tot)

    sin_dec_namap = sin_el_tot*sin_lat+cos_lat*cos_el_tot*cos_az_tot
    dec_namap = np.arcsin(sin_dec_namap)
    cos_dec_namap = np.cos(dec_namap)
    hour_angle = np.arccos((sin_el_tot-sin_lat*sin_dec_namap)/(cos_lat*cos_dec_namap))
    index, = np.where(sin_az_tot > 0)
    hour_angle[index] =  - hour_angle[index]
    ra_namap = HA*15 - np.degrees(hour_angle) 
    index, = np.where(ra_namap<0)
    path = np.vstack((ra_namap+ra,np.degrees(dec_namap))).T
    azel_path = np.vstack((np.degrees(az_tot),np.degrees(el_tot))).T

    if(azel): return path, azel_path
    else: return path

    '''
    times = np.arange(0,len(T),300)
    for t in times: 
        fig, axs = plt.subplots(1,2,figsize=(8,5), dpi=160,)
        axs[0].plot(azi[:t], alt[:t])
        axs[0].set_xlabel('az');axs[0].set_ylabel('el')
        axs[0].set_xlim(-2.1,2.1); axs[0].set_ylim(0.39, 0.7)
        axs[0].set_title(f'HA={HA[t]:2f}deg')

        axs[1].set_xlabel('az pattern');axs[1].set_ylabel('el pattern')
        axs[1].plot(path[:t,0], path[:t,1])
        axs[1].set_xlim(52.7,53.5); axs[1].set_ylim(-27.69,-27.9)
        
        fig.tight_layout();fig.savefig(f'plot/b_frame_t{t:2f}.png')
        plt.close()
    '''
    '''
    fig, axs = plt.subplots(1,2,figsize=(5,2.5), dpi=160,)
    axs[0].plot(az, el)
    axs[0].set_xlabel('az');axs[0].set_ylabel('el')
    axs[1].set_xlabel('az pattern');axs[1].set_ylabel('el pattern')
    axs[0].set_ylim(0.69,0.7)
    axs[0].set_xlim(-0.16, 0.16)
    axs[1].set_xlim(-0.35, 0.35)
    axs[1].set_ylim(-0.05, 0.031) 
    axs[1].plot(scan_path[:,0], scan_path[:,1])
    plt.show()
    times = np.arange(0,len(T),30000)
    for t in times: 
        fig, axs = plt.subplots(1,2,figsize=(8,5), dpi=160,)
        axs[0].plot(az[:t], el[:t])
        axs[0].set_xlabel('az');axs[0].set_ylabel('el')
        axs[1].set_xlabel('az pattern');axs[1].set_ylabel('el pattern')
        axs[0].set_title(f'HA={HA[t]:2f}deg')
        axs[0].set_ylim(0.69,0.7)
        axs[0].set_xlim(-0.16, 0.16)
        axs[1].set_xlim(-0.35, 0.35)
        axs[1].set_ylim(-0.05, 0.031) 
        axs[1].plot(scan_path[:t,0], scan_path[:t,1])
        fig.tight_layout();fig.savefig(f'plot/a_frame_t{t:2f}.png')
        plt.close()
    '''

def binMap(pointing_paths, res=0.02, f_range=1,dec=0, ra=0):
    
    """
    Binning the pointing into 2d array
    Parameters
    ----------
    pointing_paths: 2d array
        coordinates timestream of the pointing
    res: float
        spatial resolution of the map
    f_range: float
        range
    Returns
    -------
    xedges: array
        the x edges of the binned hitmap
    yedges: array
        the y edges of the binned hitmap
    hit_map: 2d array
        2d histogram of hit on the sky

    """ 
    x_range = f_range
    y_range = x_range

    x_res = res
    y_res = x_res

    xedges = ra+np.arange(-x_range, x_range+x_res, x_res)
    yedges = dec+np.arange(-y_range, y_range+y_res, y_res)

    pointings = np.concatenate([pixel for pixel in pointing_paths])
    hit_map   = bining(xedges,yedges, pointings)
    return xedges,yedges,hit_map

def bining(xedges,yedges,pointings):
    """
    Binning the pointing into 2d array
    Parameters
    ----------
    xedges: array
        the x edges of the binned hitmap
    yedges: array
        the y edges of the binned hitmap
    pointing: 2d array
        coordinates timestream of the pointing
    Returns
    -------
    H: 2d array
        2d histogram of hit on the sky
    """ 

    H, xedges, yedges = np.histogram2d(pointings[:,0], pointings[:,1], bins=(xedges, yedges))
    return H.T

