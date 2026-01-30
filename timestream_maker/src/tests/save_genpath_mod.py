
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
