import numpy as np
from TIM_scan_strategy import *
from IPython import embed

def hitsPerSqdeg(total_hits, area, res):
    return total_hits/(area/res**2)

def timeFractionAbove(hmap, level):
    hits = hmap.flatten()
    return np.sum(hits[hits>level])/np.sum(hits)

def genLocalPath(az_size = 1, alt_size = 1, alt_step=0.2, acc = 0.005, scan_v=0.05, dt= 0.01):
    '''
    Function that generates the local scaning pattern.
    Currently can only generate closed loop
    '''
    ver_N = int(alt_size//alt_step)

    scan_time = az_size/scan_v
    turn_time = 2*scan_v/acc

    a = np.concatenate((np.ones(int(turn_time/dt))*acc,np.zeros(int(scan_time/dt))))
    a = np.concatenate((a,-1*a))
    a = np.tile(a,ver_N) 

    acc_alt = alt_step/(turn_time/2)**2
    a2 = np.concatenate((np.ones(int(turn_time/dt/2))*acc_alt,-1*np.ones(int(turn_time/dt/2))*acc_alt,np.zeros(int(scan_time/dt))))
    a2 = np.concatenate((np.tile(a2,ver_N),np.tile(-1*a2,ver_N)))

    flag = np.where(a==0,1,0) #constant scan speed part

    t = np.arange(0,len(a))*dt

    v = np.cumsum(a)*dt-scan_v
    az = np.cumsum(v)*dt

    v2 = np.cumsum(a2)*dt
    alt  = np.cumsum(v2)*dt

    scan_eff = np.sum(flag)/len(az)
    
    return az,alt,flag, scan_eff ,t 

def genScanPath(T, alt, az, flag, plot=False):
    '''
    Function that generates the pointing coordinates vs time.
    T： time stream
    '''

    coor = np.zeros((len(T),2))

    idx = np.int_(np.fmod(T,len(alt)/100)*100)
    
    coor[:,0] = az[idx]-np.mean(az)
    coor[:,1] = alt[idx]-np.mean(alt)

    if(plot):
        plt.figure()
        plt.plot(az-np.mean(az), alt-np.mean(alt), 'or')
        plt.plot(coor[:,0],coor[:,1],'ob')
        plt.show()
    #flag      = flag[idx]

    return coor,  flag

def pixelOffset(pixel_num, pixel_pitch):
    '''
    Function that  gernerates the pixel offset vs pointing center
    pixel_num: number of spatial pixels
    pixel_pitch: spatial distance between adjacent pixels in degrees
    '''
    yoffsets = (np.arange(0,pixel_num)-pixel_num/2)*pixel_pitch
#     offsets = np.vstack((np.zeros(pixel_num),yoffsets)).T
    
    return yoffsets

def genPixelPath(pointing_path, pixel_offset, theta):
    '''
    Function that gernerates the pointing time stream for each pixel
    '''
    pixel_path = []
    for pixel in pixel_offset:   
        pixel_w_time = np.append( pixel*np.sin(theta), pixel*np.cos(theta),)
#         print(pixel_w_time)
        pixel_path.append(pointing_path+pixel_w_time) 
        
    return pixel_path

def genPointingPath(T, scan_path, HA, lat, dec):
    '''
    Function that takes local paths and generates the pointing on sky vs time.
    '''
    alt = elevationAngle(dec,lat,HA)+np.radians(scan_path[:,1])
    azi = azimuthAngle(dec,lat,HA)+np.radians(scan_path[:,0])
    
    dec_point = declinationAngle(np.degrees(azi), np.degrees(alt), lat)
    ha_point  = hourAngle(np.degrees(azi), np.degrees(alt), lat)
    
    return np.vstack((np.degrees(ha_point-HA*np.pi/12),np.degrees(dec_point))).T

def binMap(pointing_paths, res=0.02, f_range=1,dec=0):
    '''
    Binning the pointing into 2d array
    '''
    x_range = f_range
    y_range = x_range

    x_res = res
    y_res = x_res

    xedges = np.arange(-x_range, x_range+x_res, x_res)
    yedges = dec+np.arange(-y_range, y_range+y_res, y_res)

    pointings = np.concatenate([pixel for pixel in pointing_paths])
    hit_map   = bining(xedges,yedges, pointings)
    return xedges,yedges,hit_map

def bining(xedges,yedges,pointings):
    H, xedges, yedges = np.histogram2d(pointings[:,0], pointings[:,1], bins=(xedges, yedges))
    return H.T

def az_scan_custom(stripe_size, step_y, num_steps, xoffset=0, yoffset=0, plot=False):
    """
    Generate an azimuthal scan path based on input parameters.
    
    Parameters:
        stripe_size (float): Size of the horizontal stripe in radians (extent in x-direction).
        step_y (float): Step size in the y-direction in radians.
        num_steps (int): Number of steps (rows) in the y-direction.
    
    Returns:
        x_series, y_series: The x and y coordinates of the scan path.
    """
    # Define the field size
    X = stripe_size / 2  # Half the field size (square bounds)

    # Adjust the starting point and stripe size
    x0 = -stripe_size / 2
    
    # Calculate the y0 offset to center the middle row at y = 0
    if num_steps % 2 == 1:  # Odd number of rows
        y0 = -((num_steps // 2) * step_y)
    else:  # Even number of rows
        y0 = -((num_steps // 2 - 0.5) * step_y)

    # Calculate the total number of samples required per row
    dx = stripe_size / 200  # Set a reasonable resolution for the x-direction
    lx = stripe_size  # Horizontal extent of the stripe
    samples_per_row = int(lx / dx)

    # Initialize arrays for x and y coordinates
    x_series = []
    y_series = []

    # Generate the scan path row by row
    for step in range(num_steps):
        # Calculate y position for this step
        y_current = y0 + step * step_y

        # Generate x values for this row
        x_row = x0 + (np.arange(samples_per_row) * dx) % lx
        # Flip every other row for the zig-zag pattern
        if step % 2 == 1:
            x_row = -x_row

        # Append the current row's x and y values
        x_series.append(x_row)
        y_series.append(np.full_like(x_row, y_current))

    # Flatten the arrays
    x_series = np.concatenate(x_series)+xoffset-x0
    y_series = np.concatenate(y_series)+yoffset-y0
    if(plot):
        # Plot the scan path
        plt.plot(x_series, y_series, c='b', alpha=0.8)
        plt.xlabel('X (rad)')
        plt.ylabel('Y (rad)')
        plt.title('Custom Azimuthal Scan Path')
        plt.show()

    return x_series, y_series