import pygetdata as gd
import numpy as np 
import matplotlib.pyplot as plt
from astropy import wcs

path = '/mnt/d/xystage/'

d = gd.dirfile(path)

ra = d.getdata('RA', first_frame=21600, num_frames=42070-21600)
dec = d.getdata('DEC', first_frame=21600, num_frames=42070-21600)

radec = np.transpose(np.vstack((ra*15., dec)))

w = wcs.WCS(naxis=2) 
w.wcs.crpix = [50, 50]
w.wcs.cdelt = [0.3, 0.3]
w.wcs.crval = [210, -32.]
w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

try:  
    xy = w.all_world2pix(radec, 1, maxiter=20, tolerance=1.0e-4, adaptive=True, detect_divergence=True, quiet=True)
except wcs.wcs.NoConvergence as e:
    print("Indices of diverging points: {0}".format(e.divergent))
    print("Indices of poorly converging points: {0}".format(e.slow_conv))
    print("Best solution:\n{0}".format(e.best_solution))
    print("Achieved accuracy:\n{0}".format(e.accuracy))