import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
import astropy.units as u

def genLocalPath_cst_el_scan(az_size = 1, alt_size = 1, alt_step=0.02, acc = 0.05, scan_v=0.05, dt= 0.01):

    """
    Function that generates the local scanning pattern.
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

#The coordinates of the field
ra = 53.11667
dec = -27.80833

#load the observer position
lat = -77.83

#Load the scan duration and generate the time coordinates with the desired acquisition rate. 
dt = 2.7777777777777e-06*np.pi/3.14 #Make the timestep non rational to avoid some stripes in the hitmap.
T_duration = 1 #hours
T = np.arange(0,T_duration,dt) * 3600
HA = np.arange(-T_duration/2,T_duration/2,dt) #hour
HAr = HA*np.pi/12
index, = np.where(HAr<0)
HAr[index] += 2*np.pi
HA = HAr*12/np.pi
#----------------------------------------
#Generate the scan path for the center of the arrays. 
az, alt, flag = genLocalPath_cst_el_scan(az_size=0.3, alt_size=0.04, alt_step=0.02, acc=0.05, scan_v=0.05, dt=np.round(dt*3600,3))
scan_path, scan_flag = genScanPath(T, alt, az, flag)

plt.plot(T/3600, HA); plt.xlabel('t_obs [hour]'); plt.ylabel('HA [hour]')
plt.title('it is midnight at Zenith'); 
plt.figure()
plt.plot(scan_path[:,0],scan_path[:,1] )
#----------------------------------------------
# Constants
decr = np.radians(dec)  # DEC offset due to scanning [rad]
latr = np.radians(lat)             # Observer latitude [rad]
# Hour angle [rad]
HAr = HA * np.pi / 12  # Convert from hour to radians
index, = np.where(np.sin(HAr)<0)

# Unified transformation from Equatorial (RA/DEC) to Horizontal (Az/Alt)
# Using consistent formulas based on arctan2 and arcsin

# Precompute trigonometric terms
sin_dec = np.sin(decr)
cos_dec = np.cos(decr)
sin_lat = np.sin(latr)
cos_lat = np.cos(latr)
cos_HA = np.cos(HAr)
sin_HA = np.sin(HAr)

# Elevation (altitude)
el = np.arcsin(sin_dec * sin_lat + cos_dec * cos_lat * cos_HA)

# Azimuth (from North, increasing towards East)
sin_az = sin_HA * cos_dec
cos_az = (sin_dec - sin_lat * np.sin(el)) / (cos_lat * np.cos(el))
az = np.arctan2(sin_az, cos_az)

# Normalize azimuth to [0, 2pi]
az = (az + 2 * np.pi) % (2 * np.pi)

# Apply scan offsets
az += np.radians(scan_path[:,0])
el += np.radians(scan_path[:,1])
# Wrap azimuth again if needed
az = (az + 2 * np.pi) % (2 * np.pi)

fig, axs = plt.subplots(1,2,figsize=(5,3),dpi=200)
axs[0].plot(np.degrees(az), np.degrees(el), c='k', lw=1, ls='-')
axs[0].set_xlabel("Azimuth [deg]")
axs[0].set_ylabel("Elevation [deg]")
# "Unwrap" from [0, 2π) to (-π, π]
az_unwrapped = (az + np.pi) % (2 * np.pi) - np.pi
ped = (az + np.pi) % (2 * np.pi) - np.pi
axs[1].plot(np.degrees(az_unwrapped), np.degrees(el), c='k', lw=1, ls='-')
axs[1].set_xlabel("Azimuth [deg]")
axs[1].set_ylabel("Elevation [deg]")
plt.tight_layout()

# Define observer location and times
location = EarthLocation(lat=lat*u.deg, lon=0*u.deg, height=0*u.m)
time_obs = Time('2025-04-17T00:00:00') + T * u.s  # Properly time-stamped

# Define the fixed sky center
sky_center = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')

# Get the AltAz position of the fixed center at each time
altaz_center = sky_center.transform_to(AltAz(obstime=time_obs, location=location))
az_c = altaz_center.az.radian
alt_c = altaz_center.alt.radian

# Apply scan offsets to (az_c, alt_c)
# These are small-angle offsets in azimuth and altitude
az_total = az_c + np.radians(scan_path[:, 0])
alt_total = alt_c + np.radians(scan_path[:, 1])

# Now convert each (az_total, alt_total) back to RA/Dec
altaz = AltAz(az=az_total*u.rad, alt=alt_total*u.rad,
              obstime=time_obs, location=location)
from astropy.coordinates import ICRS
skycoord_icrs = altaz.transform_to(ICRS())
RA = skycoord_icrs.ra.radian
Dec = skycoord_icrs.dec.radian

# Build final path
path = np.vstack((np.degrees(RA), np.degrees(Dec))).T

# Plotting
fig, axs = plt.subplots(1,2,figsize=(6,3),dpi=200)
axs[0].plot(path[:,0], path[:,1], 'k', lw=1)
axs[0].set_title("RA/Dec Scan Path")
axs[0].set_xlabel("RA [deg]")
axs[0].set_ylabel("Dec [deg]")

RA_unwrapped = (RA + np.pi) % (2 * np.pi) - np.pi
axs[1].plot(np.degrees(RA_unwrapped), np.degrees(Dec), 'g', lw=1)
axs[1].set_title("Unwrapped RA/Dec")
axs[1].set_xlabel("RA [deg]")
axs[1].set_ylabel("Dec [deg]")
plt.tight_layout()
plt.show()
