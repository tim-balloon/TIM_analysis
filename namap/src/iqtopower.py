import pygetdata as gd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

kidstable = '../kidstable.txt'

kidsarray = np.loadtxt(kidstable,skiprows=1) 

responsivity_pos = 4
det_pos = 2

def detToR(det_num):
    r = kidsarray[kidsarray[:,det_pos]==det_num,responsivity_pos][0]
    return r

def rotatePhase(i_ts,q_ts,startframe,endframe):
    X = i_ts+1j*q_ts
    phi_avg = np.arctan2(np.mean(q_ts[startframe:endframe]),np.mean(i_ts[startframe:endframe]))
    E = X*np.exp(-1j*phi_avg)
    i_out = E.real 
    q_out = E.imag
    return i_out, q_out

def dirfileToUseful(file_name,data_type):
    nom_path = '../roach_data'
    q = gd.dirfile(nom_path,gd.RDONLY)
    values = q.getdata(file_name,data_type)
    return values


def phasetopower(ifile,qfile,det_num,startframe,endframe):
    r = detToR(det_num)
    i = dirfileToUseful(ifile,gd.FLOAT32)
    q = dirfileToUseful(qfile,gd.FLOAT32)
    #i = i[startframe:endframe]
    #q = q[startframe:endframe]
    i_rot, q_rot = rotatePhase(i,q,startframe,endframe)
    phi = np.arctan2(q_rot,i_rot)
    phibar = np.arctan2(np.mean(q[startframe:endframe]),np.mean(i[startframe:endframe]))
    delphi = phi-phibar
    power = delphi/r
    #power = power-np.mean(power[len(power)-1001:len(power)-1])
    return power

def powertoTS(power,outname):
    out_file = open(outname,'w')
    for i in range(len(power)):
        print(power[i],file=out_file)
    out_file.close()
    return

p = phasetopower('kidA_roachN','kidB_roachN',2,6737816,6743184)
print(len(p), len(p)/488)
plt.plot(p)
plt.show()

powertoTS(p,'power_ts.txt')

