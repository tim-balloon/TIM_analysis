import numpy as np
from astropy import constants as c
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import Angle, SkyCoord, EarthLocation, AltAz
import matplotlib.pyplot as plt
from matplotlib import rc
from IPython.display import clear_output
import astropy.units as u
from IPython import embed

def BuildPointing(field_altaz, field_radec, times, location, az_scans):
    pointing = SkyCoord(alt = field_altaz.alt, az = field_altaz.az+az_scans, 
                       frame='altaz', obstime = times, location = location)
    pointing_radec = pointing.transform_to('icrs') # the fuck? why can't I do this in place?
    ra_off = pointing_radec.ra - field_radec.ra
    dec_off = pointing_radec.dec - field_radec.dec
    
    return pointing, pointing_radec, ra_off, dec_off

# Define the reference time and location
mcmurdo = EarthLocation(lat='-77.8419', lon='166.6863', height=37000.)
time0 = Time('2019-12-15 18:00:00', scale='utc', location = mcmurdo)

dt = 0.9*u.second
times = time0 + np.arange(0,24,dt.to(u.hr).value)*u.hr
t_sec = ((times.jd-times.jd[0])*u.day).to(u.second)
t_hr= t_sec.to(u.hr)

az_scan_rate = 0.1*u.degree/(1.0*u.second)
az_scan_extent = 1.0*u.degree
az_scans = np.zeros_like(t_sec.value)*u.degree
az_scans[0] = -az_scan_extent/2.
scan_dir = 1
daz = az_scan_rate*dt


field_S = SkyCoord(ra=12*15.*u.degree,dec=-55*u.degree, 
                   obstime=times, location=mcmurdo)
# You can either have it in altaz, or in RA/Dec, but not both for some reason
field_S_altaz = field_S.transform_to('altaz')

field_N = SkyCoord(ra=3.*15*u.degree,dec=-27*u.degree, 
                   obstime=times, location=mcmurdo)
field_N_altaz = field_N.transform_to('altaz')

fig, (axaz, axalt) = plt.subplots(1,2,figsize=(8,5))
axaz.plot(t_hr,field_S_altaz.az.deg)
axaz.plot(t_hr,field_N_altaz.az.deg)
axalt.plot(t_hr,field_S_altaz.alt.deg)
axalt.plot(t_hr,field_N_altaz.alt.deg)


for i in np.arange(len(az_scans)-1):
    # if on the next step you would exceed the extent in the positive direction, reverse course
    if az_scans[i-1] > (az_scan_extent/2.-daz): 
        scan_dir = -1
    if az_scans[i-1] < (-az_scan_extent/2.+daz): 
        scan_dir = 1
    az_scans[i] = az_scans[i-1] + scan_dir*daz

fig, (axaz, axalt) = plt.subplots(1,2,figsize=(5,8))
axaz.plot(az_scans[0:100],'.-')
axalt.plot(t_hr, field_S_altaz.az+az_scans)

pointing_S, pointing_S_radec, dra_S, ddec_S = BuildPointing(field_S_altaz, field_S, times, mcmurdo, az_scans)
pointing_N, pointing_N_radec, dra_N, ddec_N = BuildPointing(field_N_altaz, field_N, times, mcmurdo, az_scans)

embed()

fig, (axs, axn) = plt.subplots(1,2,figsize=[8,6])
axs.plot(dra_S.to(u.arcmin), ddec_S.to(u.arcmin), '.', label='S')
axs.plot(dra_S.to(u.arcmin)-5.*u.arcmin, ddec_S.to(u.arcmin)+5.*u.arcmin, '.', label='S')
axs.plot(dra_S.to(u.arcmin)+5.*u.arcmin, ddec_S.to(u.arcmin)-5.*u.arcmin, '.', label='S')
#plt.plot(dra_N.to(units.arcmin)[0:100], ddec_N.to(units.arcmin)[0:100], '.', label='N')
axs.set_xlabel('RA offset (arcmin)')
axs.set_ylabel('Dec offset (arcmin)')
axs.legend()

axn.plot(dra_N.to(u.arcmin), ddec_N.to(u.arcmin), '.', label='N')
axn.plot(dra_N.to(u.arcmin)-5.*u.arcmin, ddec_N.to(u.arcmin)+5.*u.arcmin, '.', label='N')
axn.plot(dra_N.to(u.arcmin)+5.*u.arcmin, ddec_N.to(u.arcmin)-5.*u.arcmin, '.', label='N')
#plt.plot(dra_N.to(units.arcmin)[0:100], ddec_N.to(units.arcmin)[0:100], '.', label='N')
axn.set_xlabel('RA offset (arcmin)')
axn.set_ylabel('Dec offset (arcmin)')
axn.legend()
plt.show()