import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
import matplotlib.colors as colors

from scipy.integrate import simps
from time import time 

import importlib

from scan_fcts import *
from TIM_scan_strategy import *

mcmLat = -77.83
minLat = -85
maxLat = -75
latList = np.arange(minLat,maxLat,2.)

# http://simbad.u-strasbg.fr/simbad/sim-id?Ident=GOODS+South+field
c=SkyCoord.from_name('Goods-S Field')
print('GOODS-S:   {0:}'.format(c.to_string('hmsdms')))
goodsSDec = c.dec.value
sptDeepDec = -55.

HA = np.arange(-12,12,.02)

elMCMgoods = elevationAngle(goodsSDec,mcmLat,HA)
paMCMgoods = parallacticAngle(goodsSDec,mcmLat,HA)
elMCMspt = elevationAngle(sptDeepDec,mcmLat,HA)
paMCMspt = parallacticAngle(sptDeepDec,mcmLat,HA)


elMin = np.radians(30.)
plt.clf()
plt.plot(HA[elMCMgoods>elMin],np.degrees(paMCMgoods[elMCMgoods>elMin]),label='GOODS-S')
plt.plot(HA[elMCMspt>elMin],np.degrees(paMCMspt[elMCMspt>elMin]),label='SPTDeep')
plt.xlabel('Hour Angle (h)'); #plt.xticks(np.arange(-6,6.1,1));
plt.ylabel('Parallactic Angle (deg)')
plt.legend(handlelength=1,loc='best',fontsize='small')
plt.title('PA from MCM')
# plt.savefig('mcmPA')

plt.clf()
for latVal in latList:
	elVals = elevationAngle(goodsSDec,latVal,HA)
	paVals = parallacticAngle(goodsSDec,latVal,HA)
	plt.plot(HA[elVals>elMin],np.degrees(paVals[elVals>elMin]),label=str(latVal))
plt.xlabel('Hour Angle (h)'); plt.xticks(np.arange(-6,6.1,1));
plt.ylabel('Parallactic Angle (deg)')
plt.legend(handlelength=1,loc='best',fontsize='small',title='Latitude')
plt.title('GOODS-S PA')
# plt.savefig('goodsPA')

plt.clf()
for latVal in latList:
	elVals = elevationAngle(sptDeepDec,latVal,HA)
	paVals = parallacticAngle(sptDeepDec,latVal,HA)
	plt.plot(HA[elVals>elMin],np.degrees(paVals[elVals>elMin]),label=str(latVal))
plt.xlabel('Hour Angle (h)'); # plt.xticks(np.arange(-12,12.1,1));
plt.ylabel('Parallactic Angle (deg)')
plt.legend(handlelength=1,loc='best',fontsize='small',title='Latitude')
plt.title('SPT-Deep PA')
# plt.savefig('sptPA')

HA = 0
dec= 0

alt = elevationAngle(dec,mcmLat,HA)
azi = azimuthAngle(dec,mcmLat,HA)

dec2  = declinationAngle(np.degrees(azi), np.degrees(alt), mcmLat)
ha2   = hourAngle(np.degrees(azi), np.degrees(alt), mcmLat)
np.degrees([dec2,ha2])

az, alt, flag = genLocalPath(az_size=0.2, alt_size=0.12, alt_step=0.01, acc=0.05, scan_v=0.05, dt=0.001)

print('Length:', len(az))

fig, ax = plt.subplots(figsize=(8,6), dpi=160)

plt.scatter(az,alt,lw=0.6)
# plt.plot(az[flag==1],alt[flag==1],lw=0.5,)
ax.set_aspect(aspect=1)

# plt.xlim([-.5,.5])
# plt.ylim([-.5,.5])

plt.xlabel('Az [deg]')
plt.ylabel('El [deg]')

plt.legend(frameon=1, fontsize=8)

# plt.savefig("scan_route.png")

def scan_eff(az_size = 0.1):
    az, alt, flag = genLocalPath(az_size=az_size, alt_size=0.12, alt_step=0.01, acc=0.05, scan_v=0.05, dt=0.001)

    return np.sum(flag)/len(az)

az_sizes = np.arange(0.05,2.5,0.025)

eff = []

for az_size in az_sizes:
    eff.append(scan_eff(az_size=az_size))


fig, ax = plt.subplots(figsize=(6,4), dpi=160)

plt.plot(az_sizes,eff,lw=1., label='Scan Efficiency')

plt.xlim([0,1])
# plt.ylim([0,1])
plt.ylabel('Scan Efficiency')
plt.xlabel('Az scan size')
plt.grid()
# plt.legend(frameon=1, fontsize=10)

# plt.savefig("scan_eff.png")

goodsSDec = c.dec.value
sptDeepDec = -55.

mcmLat = -77.83
minLat = -85
maxLat = -75
latList = np.arange(minLat,maxLat,2.)

dec = goodsSDec

az_size=0.2

T_duration = 3600*15
dt = np.pi/3.14/122
T = np.arange(0,T_duration,dt)

HA = (-1*T_duration/2 + T)/3600.

az, alt, flag = genLocalPath(az_size=az_size, alt_size=0.08, alt_step=0.02, acc=0.05, scan_v=0.05, dt=0.001)
scan_path, scan_flag = genScanPath(T, alt, az, flag)

scan_path = scan_path#[scan_flag==1]
T_trim = T#[scan_flag==1]
HA_trim = HA#[scan_flag==1]

theta = np.radians(0)

pixel_offset = pixelOffset(64, 0.0148)
pixel_offset_LW = pixelOffset(51, 0.0186)

pixel_paths  = genPixelPath(scan_path, pixel_offset, theta)

#pixel_xy = np.array([-1*pixel_offset*np.sin(theta)-0.036*np.cos(theta), pixel_offset*np.cos(theta)-0.036*np.sin(theta)])
#pixel_xy_LW = np.array([-1*pixel_offset_LW*np.sin(theta)+0.036*np.cos(theta), pixel_offset_LW*np.cos(theta)+0.072*np.sin(theta)])#
# pixel_xy = np.array([-1*pixel_offset*np.sin(theta)+ 0.036*np.cos(theta), pixel_offset*np.cos(theta)+0.072*np.sin(theta)])
# pixel_xy_LW = np.array([-1*pixel_offset_LW*np.sin(theta)-0.036*np.cos(theta), pixel_offset_LW*np.cos(theta)-0.036*np.sin(theta)])

pointing_paths = [genPointingPath(T_trim, pixel_path, HA_trim, mcmLat, dec) for pixel_path in pixel_paths]

res=0.0033
f_range=0.6
xedges,yedges,hit_map = binMap(pointing_paths,res=res,f_range=f_range)

fig, (ax1) = plt.subplots(1, figsize=(8,6), dpi=160)
# fig.suptitle(t='TIM Deep field scan with az scan size = %1.1f deg\n'%az_sizes[i])


img = ax1.imshow((hit_map), \
    interpolation='nearest', origin='lower', vmin=0, vmax=np.max(hit_map), )
fig.colorbar(img, ax=ax1,)

# plt.ylim([150,200])
# plt.xlim([150,200])

az_size = 0.2

T_duration = 3600*15
dt = np.pi/3.1/11
T = np.arange(0,T_duration,dt)

HA = (-1*T_duration/2 + T)/3600.

az, alt, flag = genLocalPath(az_size=az_size, alt_size=0.08, alt_step=0.02, acc = 0.05, scan_v=0.05, dt=0.001)
scan_path, scan_flag = genScanPath(T, alt, az, flag)

scan_path_sky = genPointingPath(T, scan_path, HA, mcmLat, dec)

N1 =  0
N2 = -1
dN =  1

fig, ax = plt.subplots(figsize=(8,6), dpi=160)

scat = ax.scatter(scan_path_sky[N1:N2:dN,0], scan_path_sky[N1:N2:dN,1], s=1., lw=0.5, alpha=.8)

# plt.xlim([-0.06,-0.045])
# plt.ylim([-27.8,-27.78])
ax.set_aspect(aspect=np.abs(round(np.cos(np.radians(dec)),2)))


plt.show()