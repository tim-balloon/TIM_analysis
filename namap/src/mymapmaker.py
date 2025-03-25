import numpy as np
from astropy import wcs
from astropy.convolution import Gaussian2DKernel, convolve
from IPython import embed
from nonguy_namap_wrapper import *

def convolution(std, map_value):

    '''
    Function to convolve the maps with a gaussian.
    STD is in pixel values
    '''

    kernel = Gaussian2DKernel(x_stddev=std)

    convolved_map = convolve(map_value, kernel)

    return convolved_map

def world(crpix, crdelt, crval, ctype, coord, parang, telcoord): 
        
        '''
        Function for creating a wcs projection and a pixel coordinates 
        from sky/telescope coordinates
        '''

        w = wcs.WCS(naxis=2)
        w.wcs.crpix = crpix
        w.wcs.cdelt = crdelt
        w.wcs.crval = crval

        if telcoord is False:
            if ctype == 'XY Stage':
                world = np.zeros_like(coord)
                try:
                    world[:,0] = coord[:,0]/(np.amax(coord[:,0]))*360.
                    world[:,1] = coord[:,1]/(np.amax(coord[:,1]))*360.
                except IndexError:
                    world[0,0] = coord[0,0]/(np.amax(coord[0,0]))*360.
                    world[0,1] = coord[0,1]/(np.amax(coord[0,1]))*360.
                w.wcs.ctype = ["TLON-CAR", "TLAT-CAR"]
            else:
                if ctype == 'RA and DEC':
                    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
                elif ctype == 'AZ and EL':
                    w.wcs.ctype = ["TLON-ARC", "TLAT-ARC"]
                elif ctype == 'CROSS-EL and EL':
                    w.wcs.ctype = ["TLON-CAR", "TLAT-CAR"]
                print(crpix, crdelt, crval)
                print('TEST', coord[:,0], coord[:,1])
                world = w.all_world2pix(coord, 1)
                print('WORLD', world[:,0])
        else:
            w.wcs.ctype = ["TLON-TAN", "TLAT-TAN"]
            world = np.zeros_like(coord)
            px = w.wcs.s2p(coord, 1)
            #Use the parallactic angle to rotate the projected plane
            world[:,0] = (px['imgcrd'][:,0]*np.cos(parang)-px['imgcrd'][:,1]*np.sin(parang))/self.crdelt[0]+self.crpix[0]
            world[:,1] = (px['imgcrd'][:,0]*np.sin(parang)+px['imgcrd'][:,1]*np.cos(parang))/self.crdelt[1]+self.crpix[1]
        
        return world, w

def fmapmaking(data, weight, polangle, number, pixelmap): 
    '''

    Class to generate the maps. For more information about the system to be solved
    check Moncelsi et al. 2012

    data               #detector TOD
    weight           #weights associated with the detector values
    polangle       #polarization angles of each detector
    number           #Number of detectors to be mapped
    pixelmap       #Coordinates of each point in the TOD in pixel coordinates
    '''

    #----
    if value is None:
        value = data.copy()
    if noise is not None:
        sigma = 1/noise**2
    else:
        sigma = 1.
    if np.size(angle) > 1:
        angle = angle.copy()
    else:
        angle = angle*np.ones(np.size(value))

    '''
    sigma is the inverse of the sqared white noise value, so it is 1/n**2
    ''' 
    x_map = idxpixel[:,0]   #RA 
    y_map = idxpixel[:,1]   #DEC
    # index_1 = np.arange(0, len(x_map), 50)
    # plt.scatter(x_map[index_1], y_map[index_1], c=self.data[index_1])
    # plt.show()
    
    if (np.amin(x_map)) <= 0:
        x_map = np.floor(x_map+np.abs(np.amin(x_map)))
    else:
        x_map = np.floor(x_map-np.amin(x_map))
    if (np.amin(y_map)) <= 0:
        y_map = np.floor(y_map+np.abs(np.amin(y_map)))
    else:
        y_map = np.floor(y_map-np.amin(y_map))

    x_len = np.amax(x_map)-np.amin(x_map)+1
    param = x_map+y_map*x_len
    param = param.astype(int)

    flux = value

    cos = np.cos(2.*angle)
    sin = np.sin(2.*angle)
    print('ARRAY', param, np.size(param))
    print('FLUX', flux, np.size(flux))
    I_est_flat = np.bincount(param, weights=flux)*sigma
    Q_est_flat = np.bincount(param, weights=flux*cos)*sigma
    U_est_flat = np.bincount(param, weights=flux*sin)*sigma

    N_hits_flat = 0.5*np.bincount(param)*sigma
    c_flat = np.bincount(param, weights=0.5*cos)*sigma
    c2_flat = np.bincount(param, weights=0.5*cos**2)*sigma
    s_flat = np.bincount(param, weights=0.5*sin)*sigma
    s2_flat = N_hits_flat-c2_flat
    m_flat = np.bincount(param, weights=0.5*cos*sin)*sigma

    #return I_est_flat, Q_est_flat, U_est_flat, N_hits_flat, c_flat, c2_flat, s_flat, s2_flat, m_flat, param
    #----

def fmaps(ctype, crpix, cdelt, crval, data, coord1, coord2, convolution,std):
    '''
    ctype             #see wcs_world for explanation of this parameter
    crpix             #see wcs_world for explanation of this parameter
    cdelt             #see wcs_world for explanation of this parameter
    crval             #see wcs_world for explanation of this parameter
    coord1           #array of the first coordinate
    coord2           #array of the second coordinate
    data               #cleaned TOD that is used to create a map
    convolution      #bool parameters to check if the convolution is required
    std              #std of the gaussian is the convolution is required

    '''
    std_pixel = std/3600./np.abs(cdelt[0])

    wcsworld = world(ctype, crpix, cdelt, crval)
    w, proj = wcsworld.world(np.transpose(np.array([coord1, coord2])), parang)

def map_singledetector(crpix, pixelmap, value=None, sigma=None, angle=None, ):

    '''
    Function to reshape the previous array to create a 2D map for a single detector
    if also polarization maps are requested
    '''
    idxpixel  = pixelmap 

    (I_est_flat, Q_est_flat, U_est_flat, N_hits_flat, c_flat, c2_flat, s_flat, s2_flat, m_flat, param) = map_param(crpix=crpix, idxpixel=idxpixel, value=value, \
                                                                        noise=1/weight**2,angle=polangle)

    Delta = (c_flat**2*(c2_flat-N_hits_flat)+2*s_flat*c_flat*m_flat-c2_flat*s_flat**2-\
                N_hits_flat*(c2_flat**2+m_flat**2-c2_flat*N_hits_flat))
    A = -(c2_flat**2+m_flat**2-c2_flat*N_hits_flat)
    B = c_flat*(c2_flat-N_hits_flat)+s_flat*m_flat
    C = c_flat*m_flat-s_flat*c2_flat
    D = -((c2_flat-N_hits_flat)*N_hits_flat+s_flat**2)
    E = c_flat*s_flat-m_flat*N_hits_flat
    F = c2_flat*N_hits_flat-c_flat**2

    I_pixel_flat = np.zeros(len(I_est_flat))
    Q_pixel_flat = np.zeros(len(Q_est_flat))
    U_pixel_flat = np.zeros(len(U_est_flat))

    index, = np.where(np.abs(Delta)>0.)
    
    I_pixel_flat[index] = (A[index]*I_est_flat[index]+B[index]*Q_est_flat[index]+\
                            C[index]*U_est_flat[index])/Delta[index]
    Q_pixel_flat[index] = (B[index]*I_est_flat[index]+D[index]*Q_est_flat[index]+\
                            E[index]*U_est_flat[index])/Delta[index]
    U_pixel_flat[index] = (C[index]*I_est_flat[index]+E[index]*Q_est_flat[index]+\
                            F[index]*U_est_flat[index])/Delta[index]

    x_len = np.amax(idxpixel[:,0])-np.amin(idxpixel[:,0])
    y_len = np.amax(idxpixel[:,1])-np.amin(idxpixel[:,1])

    if len(I_est_flat) < (x_len+1)*(y_len+1):
        valmax = (x_len+1)*(y_len+1)
        pmax = np.amax(param)
        I_fin = 0.*np.arange(pmax+1, valmax)
        Q_fin = 0.*np.arange(pmax+1, valmax)
        U_fin = 0.*np.arange(pmax+1, valmax)
        
        I_pixel_flat = np.append(I_pixel_flat, I_fin)
        Q_pixel_flat = np.append(Q_pixel_flat, Q_fin)
        U_pixel_flat = np.append(U_pixel_flat, U_fin)

    ind_pol, = np.nonzero(Q_pixel_flat)
    #pol = np.sqrt(Q_pixel_flat**2+U_pixel_flat**2)

    I_pixel = np.reshape(I_pixel_flat, (y_len+1,x_len+1))
    Q_pixel = np.reshape(Q_pixel_flat, (y_len+1,x_len+1))
    U_pixel = np.reshape(U_pixel_flat, (y_len+1,x_len+1))

    return I_pixel, Q_pixel, U_pixel

def map_multidetectors(crpix, pixelmap, number):

    Xmin = np.inf
    Xmax = -np.inf
    Ymin = np.inf
    Ymax = -np.inf

    for i in range(number):
        if np.size(np.shape(pixelmap)) == 2:
            idxpixel = pixelmap.copy()
            Xmin, Xmax = np.amin(idxpixel[:, 0]), np.amax(idxpixel[:, 0])
            Ymin, Ymax = np.amin(idxpixel[:, 1]), np.amax(idxpixel[:,1])
            break
        else:
            idxpixel = pixelmap[i].copy()
            Xmin = np.amin(np.array([Xmin,np.amin(idxpixel[:, 0])]))
            Xmax = np.amax(np.array([Xmax,np.amax(idxpixel[:, 0])]))
            Ymin = np.amin(np.array([Ymin,np.amin(idxpixel[:, 1])]))
            Ymax = np.amax(np.array([Ymax,np.amax(idxpixel[:, 1])]))
    
    finalmap_I_est = np.zeros((int(np.abs(Ymax-Ymin)+1), int(np.abs(Xmax-Xmin)+1)))
    finalmap_Q_est = np.zeros((int(np.abs(Ymax-Ymin)+1), int(np.abs(Xmax-Xmin)+1)))
    finalmap_U_est = np.zeros((int(np.abs(Ymax-Ymin)+1), int(np.abs(Xmax-Xmin)+1)))
    finalmap_N_hits = np.zeros((int(np.abs(Ymax-Ymin)+1), int(np.abs(Xmax-Xmin)+1)))
    finalmap_c = np.zeros((int(np.abs(Ymax-Ymin)+1), int(np.abs(Xmax-Xmin)+1)))
    finalmap_c2 = np.zeros((int(np.abs(Ymax-Ymin)+1), int(np.abs(Xmax-Xmin)+1)))
    finalmap_s = np.zeros((int(np.abs(Ymax-Ymin)+1), int(np.abs(Xmax-Xmin)+1)))
    finalmap_s2 = np.zeros((int(np.abs(Ymax-Ymin)+1), int(np.abs(Xmax-Xmin)+1)))
    finalmap_m = np.zeros((int(np.abs(Ymax-Ymin)+1), int(np.abs(Xmax-Xmin)+1)))
    finalmap_I = np.zeros((int(np.abs(Ymax-Ymin)+1), int(np.abs(Xmax-Xmin)+1)))
    finalmap_Q = np.zeros((int(np.abs(Ymax-Ymin)+1), int(np.abs(Xmax-Xmin)+1)))
    finalmap_U = np.zeros((int(np.abs(Ymax-Ymin)+1), int(np.abs(Xmax-Xmin)+1)))

    for i in range(number):
        if np.size(np.shape(pixelmap)) == 2:
            idxpixel = pixelmap.copy()
        else:
            idxpixel = pixelmap[i].copy()

        (I_est_flat, Q_est_flat, U_est_flat, N_hits_flat, \
            c_flat, c2_flat, s_flat, s2_flat, m_flat, param) = map_param(crpix=crpix, idxpixel=idxpixel, value=data[i], \
                                                                            noise=1/weight[i],angle=polangle[i])

        Xmin_map_temp, Xmax_map_temp = np.amin(idxpixel[:,0]), np.amax(idxpixel[:,0])
        Ymin_map_temp, Ymax_map_temp = np.amin(idxpixel[:,1]), np.amax(idxpixel[:,1])

        index1x = int(Xmin_map_temp-Xmin)
        index2x = int(index1x + np.abs(Xmax_map_temp-Xmin_map_temp))
        index1y = int(Ymin_map_temp-Ymin)
        index2y = int(index1y + np.abs(Ymax_map_temp-Ymin_map_temp))

        x_len = Xmax_map_temp-Xmin_map_temp
        y_len = Ymax_map_temp-Ymin_map_temp

        if len(I_est_flat) < (x_len+1)*(y_len+1):
            valmax = (x_len+1)*(y_len+1)
            pmax = np.amax(param)
            I_num_fin = 0.*np.arange(pmax+1, valmax)
            Q_num_fin = 0.*np.arange(pmax+1, valmax)
            U_num_fin = 0.*np.arange(pmax+1, valmax)
            N_hits_fin = 0.*np.arange(pmax+1, valmax)
            c_fin = 0.*np.arange(pmax+1, valmax)
            c2_fin = 0.*np.arange(pmax+1, valmax)
            s_fin = 0.*np.arange(pmax+1, valmax)
            s2_fin = 0.*np.arange(pmax+1, valmax)
            m_fin = 0.*np.arange(pmax+1, valmax)
            
            I_est_flat = np.append(I_est_flat, I_num_fin)
            Q_est_flat = np.append(Q_est_flat, Q_num_fin)
            U_est_flat = np.append(U_est_flat, U_num_fin)
            N_hits_flat = np.append(N_hits_flat, N_hits_fin)
            c_flat = np.append(c_flat, c_fin)
            c2_flat = np.append(c2_flat, c2_fin)
            s_flat = np.append(s_flat, s_fin)
            s2_flat = np.append(s2_flat, s2_fin)
            m_flat = np.append(m_flat, m_fin)

        I_est = np.reshape(I_est_flat, (y_len+1,x_len+1))
        Q_est = np.reshape(Q_est_flat, (y_len+1,x_len+1))
        U_est = np.reshape(U_est_flat, (y_len+1,x_len+1))
        N_hits_est = np.reshape(N_hits_flat, (y_len+1,x_len+1))
        c_est = np.reshape(c_flat, (y_len+1,x_len+1))
        c2_est = np.reshape(c2_flat, (y_len+1,x_len+1))
        s_est = np.reshape(s_flat, (y_len+1,x_len+1))
        s2_est = np.reshape(s2_flat, (y_len+1,x_len+1))
        m_est = np.reshape(m_flat, (y_len+1,x_len+1))


        finalmap_I_est[index1y:index2y+1,index1x:index2x+1] += I_est
        finalmap_Q_est[index1y:index2y+1,index1x:index2x+1] += Q_est
        finalmap_U_est[index1y:index2y+1,index1x:index2x+1] += U_est
        finalmap_N_hits[index1y:index2y+1,index1x:index2x+1] += N_hits_est
        finalmap_c[index1y:index2y+1,index1x:index2x+1] += c_est
        finalmap_c2[index1y:index2y+1,index1x:index2x+1] += c2_est
        finalmap_s[index1y:index2y+1,index1x:index2x+1] += s_est
        finalmap_s2[index1y:index2y+1,index1x:index2x+1] += s2_est
        finalmap_m[index1y:index2y+1,index1x:index2x+1] += m_est


    Delta = (c_flat**2*(c2_flat-N_hits_flat)+2*s_flat*c_flat*m_flat-c2_flat*s_flat**2-\
                N_hits_flat*(c2_flat**2+m_flat**2-c2_flat*N_hits_flat))
    A = -(c2_flat**2+m_flat**2-c2_flat*N_hits_flat)
    B = c_flat*(c2_flat-N_hits_flat)+s_flat*m_flat
    C = c_flat*m_flat-s_flat*c2_flat
    D = -((c2_flat-N_hits_flat)*N_hits_flat+s_flat**2)
    E = c_flat*s_flat-m_flat*N_hits_flat
    F = c2_flat*N_hits_flat-c_flat**2

    index, = np.where(np.abs(Delta)>0.)
    print('INDEX', i, index, np.amin(Delta[index]), np.amax(Delta[index]))
    finalmap_I[index] = (A[index]*finalmap_I_est[index]+B[index]*finalmap_Q_est[index]+\
                            C[index]*finalmap_U_est[index])/Delta[index]
    finalmap_Q[index] = (B[index]*finalmap_I_est[index]+D[index]*finalmap_Q_est[index]+\
                            E[index]*finalmap_U_est[index])/Delta[index]
    finalmap_U[index] = (C[index]*finalmap_I_est[index]+E[index]*finalmap_Q_est[index]+\
                            F[index]*finalmap_U_est[index])/Delta[index]

    return finalmap_I, finalmap_Q, finalmap_U

def map2d(data, noise, pol_angle, w, crpix, std, convolution ):

    '''
    Function to generate the maps using the pixel coordinates to bin
    '''

    if np.size(np.shape(data)) == 1:
        mapmaker = mapmaking(data, noise, pol_angle, 1, np.floor(w).astype(int))

        Imap, Qmap, Umap = mapmaker.map_singledetector(crpix)
        if not convolution:
            return Imap, Qmap, Umap
        else:
            Imap_con = mapmaker.convolution(std, Imap)
            Qmap_con = mapmaker.convolution(std, Qmap)
            Umap_con = mapmaker.convolution(std, Umap)
            return Imap_con, Qmap_con, Umap_con

    else:
        mapmaker = mapmaking(data, noise, pol_angle, np.size(np.shape(data)), np.floor(w).astype(int))
 
        Imap, Qmap, Umap = mapmaker.map_multidetectors(crpix)
        if not convolution:
            return Imap, Qmap, Umap
        else:
            Imap_con = mapmaker.convolution(std, Imap)
            Qmap_con = mapmaker.convolution(std, Qmap)
            Umap_con = mapmaker.convolution(std, Umap)
            return Imap_con, Qmap_con, Umap_con

if __name__ == "__main__":

    #---------------------------------
    filepath = '/home/mvancuyck/Desktop/master'
    kid_num = 4
    detfreq = '488.0'
    highpassfreq = '0.1'
    list_conv = [['A', 'B'], ['D', 'E'], ['G', 'H'], ['K', 'I'], ['M', 'N']]
    det_I_string = 'kid'+list_conv[kid_num-1][0]+'_roachN'
    det_Q_string = 'kid'+list_conv[kid_num-1][1]+'_roachN'
    I_data = load(det_I_string)
    Q_data = load(det_Q_string)
    kidutils = det.kidsutils()
    det_data = kidutils.KIDmag(I_data, Q_data)
    coord2 = 'DEC'
    coord2_data = load(coord2)  
    coord1 = 'RA'
    coord1_data = load(coord1)
    hwp_data=0
    zoomsyncdata = ld.frame_zoom_sync(det_data, detfreq, detfreq, coord1_data, coord2_data, ('100.0'), ('100.0'), ('72373'), '79556', ('BLAST-TNG'),
                                   None, None, None, None, None, roach_number='3', roach_pps_path = filepath, xystage=False)
    (timemap, detslice, coord1slice, coord2slice, hwpslice) = zoomsyncdata.sync_data()
    det_tod = tod.data_cleaned(detslice, detfreq, highpassfreq, str(kid_num), 5, True, 5, 5)
    cleaned_data = det_tod.data_clean()
    det_off = np.zeros((np.size(kid_num),2))
    noise_det = np.ones(np.size(kid_num))
    grid_angle = np.zeros(np.size(kid_num))
    pol_angle_offset = np.zeros(np.size(kid_num))
    resp = np.ones(np.size(kid_num))
    ctype = 'RA and DEC'
    crpix = np.asarray([50, 50])
    cdelt = np.asarray([0.1, 0.1])
    crval = np.asarray([230.  , -55.79])
    pixnum = np.asarray([100., 100.])
    convolution = False
    std = 0
    pol_angle = np.zeros(3504903)
    noise_det = 1
    parallactic = 0
    #---------------------------------

    wcsworld = mp.wcs_world(ctype, crpix, cdelt, crval)

    maps = mp.maps(ctype, crpix, cdelt, crval, 
                    cleaned_data, coord1slice, coord2slice, 
                    convolution, std, True, 
                    pol_angle=pol_angle, noise=noise_det, 
                    telcoord = False, 
                    parang=parallactic)
     
    maps.wcs_proj()
    proj = maps.proj
    w = maps.w
    map_value = maps.map2d()

    position = SkyCoord(crval[0], crval[1], unit='deg', frame='icrs')
    cutout = Cutout2D(map_value, position, (100,100) , wcs=proj)

    #mp_ini = MapPlotsGroup()
    #mp_ini.updateTab(data=maps.map2d(), coord1 = coord1slice, coord2 = coord2slice, crval = crval, ctype = ctype, pixnum = pixnum, telcoord = False,  crpix = crpix, cdelt = cdelt, projection = proj, xystage=False)
    #coord_test, proj_new = wcsworld.world(np.reshape(crval, (1,2)), parallactic)

    fig, axis = plt.subplots(subplot_kw={'projection':proj},dpi=150)
    size = (pixnum[1], pixnum[0])     # pixels

    #proj = cutout.wcs
    #print('PROJ', proj)
    mapdata = cutout.data

    levels = np.linspace(0.5, 1, 3)*np.amax(mapdata)

    if ctype == 'RA and DEC':
        ra = axis.coords[0]
        dec = axis.coords[1]
        ra.set_axislabel('RA (deg)')
        dec.set_axislabel('Dec (deg)')
        dec.set_major_formatter('d.ddd')
        ra.set_major_formatter('d.ddd')

    im = axis.imshow(mapdata, origin='lower', cmap=plt.cm.viridis)
    #axis.contour(mapdata, levels=levels, colors='white', alpha=0.5)
    #fig.canvas.draw()       

    fig.colorbar(im, ax=axis)
    plt.close()
    

    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 10**6:.2f} MB; Peak: {peak / 10**6:.2f} MB")

    tracemalloc.stop()



