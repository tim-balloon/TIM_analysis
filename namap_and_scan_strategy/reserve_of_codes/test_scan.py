import matplotlib.pyplot as plt
import numpy as np
from IPython import embed
import astropy.units as u
import matplotlib.patches as patches

def zigzag_scan(rows, cols):
    x = []
    y = []

    for row in range(rows):
        if row % 2 == 0:  # Left to right
            x.extend(np.arange(cols))
        else:  # Right to left
            x.extend(np.arange(cols - 1, -1, -1))
        y.extend([row] * cols)

    return np.asarray(x), np.asarray(y)


def strategy(npix, beam_pix=45*u.arcsec, Tint_tot_s =60, 
             acq_frequency_Hz = 122, v = 0.00003, dec_step = 0.1, origin=1):

    npix_angle = np.arange(origin, origin+npix*beam_pix.to(u.deg).value,beam_pix.to(u.deg).value)

    fig, (ax1,ax2) = plt.subplots(1,2)
    ax1.set_title('one array')
    
    ax1.set_xlabel('deg')
    ax1.set_ylabel('deg')
    ax1.set_ylim(0, 2.5)
    ax1.set_xlim(-1, 4)

    # Define the number of rows and columns
    dt = 1 / acq_frequency_Hz
    rows = 3
    cols = int(Tint_tot_s // dt)
    # Get the zigzag pattern coordinates
    x, y = zigzag_scan(rows, cols)
    # Plotting
    ax2.plot(np.asarray(x)*v, np.asarray(y)*dec_step, marker='o')
    ax2.set_title('Zigzag Scanning Pattern \n of one pixel')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.grid(True)
    ax2.invert_yaxis()  # Invert y-axis to match typical scanning patterns
    
    x_combined = np.zeros((npix, len(x)))
    for i in range(npix):
        x_combined[i,:] = origin + (x*v)
    y_combined = npix_angle[:, np.newaxis] +y[np.newaxis,:]*dec_step

    ax1.hist2d(x_combined, y_combined, bins=100, cmap='inferno')

    for pix_center in npix_angle:
        circle = patches.Circle((origin,pix_center), beam_pix.to(u.deg).value, edgecolor='blue', facecolor='cyan')
        #ax1.add_patch(circle)
        #ax1.plot(np.asarray(x)*v+origin,pix_center+ np.asarray(y)*dec_step, c='g',marker='o', alpha=0.2)
    
    # Creating a 2D histogram
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')

    fig.tight_layout()
    plt.show()
    embed()

if __name__ == "__main__":

    strategy(64)
