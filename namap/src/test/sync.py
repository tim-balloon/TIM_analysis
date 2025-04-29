import pygetdata as gd
import numpy as np
from scipy.interpolate import interp1d

def intersync_roach(data, bins):

    start = np.append(0, np.cumsum(bins[:-1]))
    end = np.cumsum(bins)

    ln = np.linspace(start, end-1, 488)
    idx = np.reshape(np.transpose(ln), np.size(ln))
    idx_plus = np.append(idx[:-1]+1, idx[-1])
    
    return (data[idx_plus.astype(int)]-data[idx.astype(int)])*(idx-idx.astype(int))+data[idx.astype(int)]

path = '/mnt/d/xystage/'
d = gd.dirfile(path)

starting_frame = 21600
ending_frame = 42070
buffer_frame = 100
num_frames = ending_frame-starting_frame

x = d.getdata('x_stage', first_frame=starting_frame-buffer_frame, num_frames=num_frames+2*buffer_frame)
y = d.getdata('y_stage', first_frame=starting_frame-buffer_frame, num_frames=num_frames+2*buffer_frame)
pps = d.getdata('pps_count_roach3', first_frame=starting_frame-buffer_frame, num_frames=num_frames+2*buffer_frame)
kidi = d.getdata('kidG_roachN', first_frame=starting_frame-buffer_frame, num_frames=num_frames+2*buffer_frame)
kidq = d.getdata('kidH_roachN', first_frame=starting_frame-buffer_frame, num_frames=num_frames+2*buffer_frame)
ctime_roach = d.getdata('ctime_packet_roach3', first_frame=starting_frame-buffer_frame, num_frames=num_frames+2*buffer_frame)
ctime_mcp = d.getdata('time', first_frame=starting_frame-buffer_frame, num_frames=num_frames+2*buffer_frame)
ctime_usec = d.getdata('time_usec', first_frame=starting_frame-buffer_frame, num_frames=num_frames+2*buffer_frame)
framecount_100hz = d.getdata('mcp_100hz_framecount', first_frame=starting_frame-buffer_frame, num_frames=num_frames+2*buffer_frame)

idx, = np.where(ctime_mcp == ctime_mcp[0])

#Create CTIME array for MCP
ctime_start_temp = ctime_mcp[0]+ctime_usec[0]/1e6+0.2
ctime_mcp = ctime_start_temp + (framecount_100hz-framecount_100hz[0])/100

ctime_mcp = ctime_mcp[buffer_frame*100:buffer_frame*100+num_frames*100]

ctime_start = ctime_mcp[0]
ctime_end = ctime_mcp[-1]
#Create CTIME array for the XY Stage
freq_array = np.append(0, np.cumsum(np.repeat(1/5, 5*num_frames-1)))
ctime_xystage = ctime_start+freq_array

print(len(x))
x_data = x[buffer_frame*5:buffer_frame*5+num_frames*5]
y_data = y[buffer_frame*5:buffer_frame*5+num_frames*5]
print(len(x_data))
#x_data = x_data/np.amax(x_data)*360
#y_data = y_data/np.amax(y_data)*360

#Create CTIME array for the roaches and Data
bn = np.bincount(pps)
bins = bn[bn>0]

mag = np.sqrt(kidi**2+kidq**2)

if bins[0] < 350:
    pps = pps[bins[0]:]
    ctime_roach = ctime_roach[bins[0]:]
    mag = mag[bins[0]:]

if bins[-1] < 350:
    pps = pps[:-bins[-1]]
    ctime_roach = ctime_roach[:-bins[-1]]
    mag = mag[:-bins[-1]]

ctime_roach = ctime_roach*1e-2
ctime_roach += 1570000000

pps_duration = pps[-1]-pps[0]+1
pps_final = pps[0]+np.arange(0, pps_duration, 1/488)

ctime_roach = intersync_roach(ctime_roach, bins[bins>350])
ctime_roach += pps_final
print(ctime_roach[0])

mag = intersync_roach(mag, bins[bins>350])

idx_roach_start, = np.where(np.abs(ctime_roach-ctime_start) == np.amin(np.abs(ctime_roach-ctime_start)))
idx_roach_end, = np.where(np.abs(ctime_roach-ctime_end) == np.amin(np.abs(ctime_roach-ctime_end)))

detdata = mag[idx_roach_start[0]:idx_roach_end[0]]
timedet = ctime_roach[idx_roach_start[0]:idx_roach_end[0]]

#Interpolate XY Stage position and time to roach time

x_int = interp1d(ctime_xystage, x_data, kind='linear')
y_int = interp1d(ctime_xystage, y_data, kind= 'linear')

index1, = np.where(np.abs(timedet-ctime_xystage[0]) == np.amin(np.abs(timedet-ctime_xystage[0])))
index2, = np.where(np.abs(timedet-ctime_xystage[-1]) == np.amin(np.abs(timedet-ctime_xystage[-1])))

x_final = x_int(timedet[index1[0]+200:index2[0]-200])
y_final = y_int(timedet[index1[0]+200:index2[0]-200])
detfinal = detdata[index1[0]+200:index2[0]-200]
timedet = timedet[index1[0]+200:index2[0]-200]

