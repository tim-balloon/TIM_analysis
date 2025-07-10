import numpy as np
from src.astrometry_fcts import *
from IPython import embed

def hitsPerSqdeg(total_hits, area):
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
    return np.sum(total_hits)/area

def timeFractionAbove(hmap, threshold):
    """
    fraction of time above a level of hits
    Parameters
    ----------
    hmap: 2d array
        hit map   
    treshold: int
        treshold of hits
    Returns
    -------
    timeFractionAbove: float
        fraction of the hitmaps above a level of hits
    """ 
    hits = hmap.flatten()
    return np.sum(hits[hits>threshold])/np.sum(hits)

def genLocalPath_cst_el_scan_crisscross(az_size = 1, alt_size = 1, alt_step=0.02, acc = 0.05, scan_v=0.05, dt= 0.01):

    """
    Function that generates the local scanning pattern.
    This function generate a constant elevation scan, that steps in elevation after each turn-around. 
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
    vertical_steps = int(alt_size//alt_step)
    #Compute Time for Scan and Turns
    scan_time = az_size/scan_v #Time required to cover the full azimuth range at scan_v
    turn_time = 2*scan_v/acc #Time required to perform a turn (deceleration, reversal, acceleration)
    #Generate Azimuth Acceleration Pattern (az_acc):
    #The motion consists of acceleration, constant velocity, and deceleration, forming a symmetric back-and-forth oscillation in azimuth.
    az_acc = np.concatenate((np.ones(int(turn_time/dt))*acc,np.zeros(int(scan_time/dt))))
    az_acc = np.concatenate((az_acc,-1*az_acc))
    #The sequence is repeated for each altitude step (vertical_steps times).
    az_acc = np.tile(az_acc,vertical_steps)

    #Compute the Altitude Acceleration
    acc_alt_value =  alt_step/(turn_time/2)**2   

    #Generate Altitude Acceleration Pattern (acc_alt)
    #The altitude changes slightly during turns, using a small acceleration.
    oscillation = np.tile(
        np.concatenate([
            np.ones(int(turn_time / dt / 2)) * acc_alt_value,
            np.ones(int(turn_time / dt / 2)) * -acc_alt_value
        ]), 1
    )
    # Ensure no extra oscillation at the ends of azimuth scan
    acc_alt = np.concatenate((oscillation, np.zeros(int(scan_time / dt))))
    acc_alt = np.concatenate((acc_alt, acc_alt))  # No altitude change on the leftward scan
    acc_alt = np.tile(acc_alt, vertical_steps)

    #Compute Azimuth (az) and Altitude (alt) Coordinates:
    #Computed by integrating acceleration to get velocity, then integrating velocity to get position.
    az_v = np.cumsum(az_acc)*dt-scan_v
    az = np.cumsum(az_v)*dt
    alt_v = np.cumsum(acc_alt)*dt
    alt  = np.cumsum(alt_v)*dt

    flag = np.where(az_acc==0,1,0) #constant scan speed part
    return az,alt,flag  

def genLocalPath_cst_el_scan(az_size = 1, alt_size = 1, alt_step=0.02, acc = 0.05, scan_v=0.05, dt= 0.01):

    """
    Function that generates the local scanning pattern.
    This function generate a constant elevation scan, that steps in elevation at every other turn-around.
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
    vertical_steps = 1 #int(alt_size//alt_step) 

    #Compute Time for Scan and Turns
    scan_time = az_size/scan_v #Time required to cover the full azimuth range at scan_v
    turn_time = 2*scan_v/acc #Time required to perform a turn (deceleration, reversal, acceleration).

    #Generate Azimuth Acceleration Pattern (az_acc):
    #The motion consists of acceleration, constant velocity, and deceleration, forming a symmetric back-and-forth oscillation in azimuth.
    az_acc = np.concatenate((np.ones(int(turn_time/dt))*acc,np.zeros(int(scan_time/dt))))
    az_acc = np.concatenate((az_acc,-1*az_acc))
    #The sequence is repeated for each altitude step (vertical_steps times).
    az_acc = np.tile(az_acc,vertical_steps)

    #Compute the Altitude Acceleration
    acc_alt_value =  alt_step/(turn_time/2)**2   

    #Generate Altitude Acceleration Pattern (acc_alt)
    #The altitude changes slightly during turns, using a small acceleration.
    oscillation = np.tile(
        np.concatenate([
            np.ones(int(turn_time / dt / 2)) * acc_alt_value,
            np.ones(int(turn_time / dt / 2)) * -acc_alt_value
        ]), 1
    )
    # Ensure no extra oscillation at the ends of azimuth scan
    acc_alt = np.concatenate((oscillation, np.zeros(int(scan_time / dt))))
    #acc_alt = np.concatenate((acc_alt, -1 * acc_alt))  # Repeat for downward scan
    acc_alt = np.concatenate((acc_alt, np.zeros_like(acc_alt)))  # No altitude change on the leftward scan
    acc_alt = np.tile(acc_alt, vertical_steps)

    #Compute Azimuth (az) and Altitude (alt) Coordinates:
    #Computed by integrating acceleration to get velocity, then integrating velocity to get position.
    az_v = np.cumsum(az_acc)*dt-scan_v
    az = np.cumsum(az_v)*dt
    alt_v = np.cumsum(acc_alt)*dt
    alt  = np.cumsum(alt_v)*dt

    flag = np.where(az_acc==0,1,0) #constant scan speed part

    #t = np.arange(len(alt)*dt,dt)
    #v = np.vstack((az_v,alt_v)).T

    return az,alt,flag #,v

def genLocalPath(az_size = 1, alt_size = 1, alt_step=0.02, acc = 0.05, scan_v=0.05, dt= 0.01):
    """
    Function that generates the local scanning pattern.
    This function a generate closed loop, that steps in elevation at every turn around.
    Then, it come back to the starting point by stepping down in elevation ar    Parameters
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
    vertical_steps = int(alt_size//alt_step)

    #Compute Time for Scan and Turns
    scan_time = az_size/scan_v #Time required to cover the full azimuth range at scan_v
    turn_time = 2*scan_v/acc #Time required to perform a turn (deceleration, reversal, acceleration).

    #Generate Azimuth Acceleration Pattern (az_acc):
    #The motion consists of acceleration, constant velocity, and deceleration, forming a symmetric back-and-forth oscillation in azimuth.
    az_acc = np.concatenate((np.ones(int(turn_time/dt))*acc,np.zeros(int(scan_time/dt))))
    az_acc = np.concatenate((az_acc,-1*az_acc))
    #The sequence is repeated for each altitude step (vertical_steps times).
    az_acc = np.tile(az_acc,vertical_steps)

    #Compute the Altitude Acceleration
    acc_alt_value =  alt_step/(turn_time/2)**2   

    #Generate Altitude Acceleration Pattern (acc_alt)
    #The altitude changes slightly during turns, using a small acceleration.
    acc_alt = np.concatenate((np.ones(int(turn_time/dt/2))*acc_alt_value,-1*np.ones(int(turn_time/dt/2))*acc_alt_value,np.zeros(int(scan_time/dt))))
    acc_alt = np.concatenate((np.tile(acc_alt,vertical_steps),np.tile(-1*acc_alt,vertical_steps)))

    flag = np.where(az_acc==0,1,0) #constant scan speed part

    #Compute Azimuth (az) and Altitude (alt) Coordinates:
    #Computed by integrating acceleration to get velocity, then integrating velocity to get position.
    az_v = np.cumsum(az_acc)*dt-scan_v
    az = np.cumsum(az_v)*dt

    alt_v = np.cumsum(acc_alt)*dt
    alt  = np.cumsum(alt_v)*dt

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
    #v_list = np.zeros((len(T),2))


    idx = np.int_(np.fmod(T,len(alt)/100)*100)
    
    coor[:,0] = az[idx]-np.mean(az)
    coor[:,1] = alt[idx]-np.mean(alt)

    #v_list[:,0] = v[idx,0]
    #v_list[:,1] = v[idx,1]

    flag = flag[idx]
    
    return coor , flag #,v_list,flag

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

def pixels_rotations(pixel_offset, pixel_shift, theta):
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
    rotated_pixel = []
    for pixel, xpixel in zip(pixel_offset, pixel_shift):  
        pixel_w_time = np.array([xpixel * np.cos(theta) - pixel * np.sin(theta), 
                                 xpixel * np.sin(theta) + pixel * np.cos(theta)])  # Apply rotation
        #pixel_w_time = np.append( pixel*np.sin(theta), pixel*np.cos(theta))
        rotated_pixel.append(pixel_w_time) 
    return np.asarray(rotated_pixel)

def genPointingPath(T, scan_path, HA, lat, dec,ra, offsets = np.zeros(2), azel=False):

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

    x_el = azi * np.cos(alt) - np.radians(offsets[0])
    azi = x_el / np.cos(alt)
    alt += +np.radians(offsets[1])

    dec_point = declinationAngle(np.degrees(azi), np.degrees(alt), lat)
    ha_point  = hourAngle(       np.degrees(azi), np.degrees(alt), lat)

    ra = (HA*np.pi/12-ha_point)
    ra_unwrapped = ( ra + np.pi) % (2 * np.pi) - np.pi

    path = np.vstack((np.degrees(ra_unwrapped),np.degrees(dec_point))).T
    azel_path = np.vstack((np.degrees(azi),np.degrees(alt))).T

    if(azel): return path, azel_path
    else: return path

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
    hit_map   = binning(xedges,yedges, pointings)
    return xedges,yedges,hit_map

def binning(xedges,yedges,pointings):
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

