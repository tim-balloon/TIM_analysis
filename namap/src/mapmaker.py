import numpy as np
from astropy import wcs
from astropy.convolution import Gaussian2DKernel, convolve
from IPython import embed
import matplotlib.pyplot as plt
import os
from astropy.io import fits
import datetime

class maps():

    '''
    Wrapper class for the wcs_word class and the mapmaking class.
    Parameters
    ----------
    Returns
    -------
    '''

    def __init__(self, ctype, crpix, cdelt, crval, pixnum, data, coord1, coord2, convolution, std, coadd=False, noise=1., telcoord=False, parang=None):
        '''
        Create an instance of maps
        Parameters
        ----------
        Returns
        -------
        '''

        self.ctype = ctype             #see wcs_world for explanation of this parameter
        self.crpix = crpix             #see wcs_world for explanation of this parameter
        self.cdelt = cdelt             #see wcs_world for explanation of this parameter
        self.crval = crval             #see wcs_world for explanation of this parameter
        self.pixnum = pixnum           #Max number of pixel
        self.coord1 = coord1           #array of the first coordinate
        self.coord2 = coord2           #array of the second coordinate
        self.data = data               #cleaned TOD that is used to create a map
        self.w = 0.                    #initialization of the coordinates of the map in pixel coordinates
        self.proj = 0.                 #inizialization of the wcs of the map. see wcs_world for more explanation about projections
        self.convolution = convolution #parameters to check if the convolution is required
        self.std = float(std)          #std of the gaussian is the convolution is required
        self.noise = noise             #white level noise of detector(s)
        self.telcoord = telcoord       #If True the map is drawn in telescope coordinates. That means that the projected plane is rotated
        self.coadd = coadd       #If to coadd all the detectors maps or return their individual maps. 
        if parang is not None:
            self.parang = [np.radians(p) for p in parang ] #Parallactic Angle. This is used to compute the pixel indices in telescopes coordinates
        else:
            self.parang = parang

    def wcs_proj(self):

        '''
        Function to compute the projection and the pixel coordinates
        Parameters
        ----------
        Returns
        -------
        '''
        wcsworld = wcs_world(self.ctype, self.crpix, self.cdelt, self.crval, self.telcoord)
        proj, w = wcsworld.world(self.coord1,self.coord2, self.parang)
        self.proj = proj
        self.w = w

    def map2d(self):

        '''
        Function to generate the maps using the pixel coordinates to bin
        Parameters
        ----------
        Returns
        -------
        '''
        mapmaker = mapmaking(self.data, self.noise, len(self.data), self.proj, self.coadd)
        Pow_map = mapmaker.map_Ionly(coadd=self.coadd)

        if not self.convolution: return Pow_map
        else:
            #Needs to be modify ! 
            std_pixel = self.std/3600./self.cdelt[0]
            return mapmaker.convolution(std_pixel, Pow_map)
        
    def map_plot(self, data_maps, kid_num):

        """
        Plot the map out of the data timestreams.     
        Parameters
        ---------- 
        data_maps: list
            list of maps to plot
        kid_num: list: 
            names of the kids used to generate the list of maps.   
        Returns
        -------
        """    
        crval = self.w.wcs.crval
        cdelt = self.w.wcs.cdelt[0]
        ctype = self.ctype
        pixnum=self.pixnum


        xform ='d.ddd'
        yform ='d.ddd'
        if self.telcoord is False or True:
            if ctype == 'RA and DEC':
                xlab = 'RA (deg)'
                ylab = 'Dec (deg)'
            
            elif ctype == 'AZ and EL':
                xlab = 'AZ (deg)'
                ylab = 'EL (deg)'
            
            elif ctype == 'CROSS-EL and EL':
                xlab = 'xEL (deg)'
                ylab = 'EL (deg)'

            elif ctype == 'XY Stage':
                xlab = 'X'
                ylab = 'Y'
        else:
            xlab = 'YAW (deg)'
            ylab = 'PITCH (deg)'


        if(self.coadd):
            fig, ax = plt.subplots(dpi=150, subplot_kw={'projection': self.w})
            im = ax.imshow(data_maps, origin='lower', interpolation='None', cmap='cividis' )
                        
            cbar = fig.colorbar(im, ax=ax, orientation='vertical',)
            cbar.set_label('Intensity')  # Adjust the label if needed

            ax.set_title('Coadd Map')

            xel = ax.coords[0]
            el = ax.coords[1]
            xel.set_axislabel(xlab)
            el.set_axislabel(ylab)
            
            plt.tight_layout()
            path = os.getcwd()+'/plot/'+f'coadd.png'
            plt.savefig(path, transparent=True)
            #plt.show()
            
            f = fits.PrimaryHDU(data_maps, header=self.w.to_header())
            hdu = fits.HDUList([f])
            hdr = hdu[0].header
            hdr.set("map")
            hdr.set("Datas")
            hdr["BITPIX"] = ("64", "array data type")
            hdr["BUNIT"] = 'MJy/sr'
            hdr["DATE"] = (str(datetime.datetime.now()), "date of creation")
            hdu.writeto( os.getcwd()+'/fits_and_hdf5/'+f'coadd.fits', overwrite=True)
            hdu.close()

        else: 
            for m, name in zip(data_maps, kid_num): 

                fig, ax = plt.subplots(dpi=150, subplot_kw={'projection': self.w})
                im = ax.imshow(m, origin='lower', interpolation='None', cmap='cividis' )
                cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
                cbar.set_label('Intensity')  # Adjust the label if needed
                ax.set_title(f'Map of {name}')

                xel = ax.coords[0]
                el = ax.coords[1]
                xel.set_axislabel(xlab)
                el.set_axislabel(ylab)

                plt.tight_layout()
                path = os.getcwd()+'/plot/'+f'{name}.png'
                plt.savefig(path, transparent=True)
                #if(len(kid_num)<6): plt.show()
                #else: plt.close()
                plt.close()

                f = fits.PrimaryHDU(m, header=self.w.to_header())
                hdu = fits.HDUList([f])
                hdr = hdu[0].header
                hdr.set("map")
                hdr.set("Datas")
                hdr["BITPIX"] = ("64", "array data type")
                hdr["BUNIT"] = 'MJy/sr'
                hdr["DATE"] = (str(datetime.datetime.now()), "date of creation")
                hdu.writeto(os.getcwd()+'/fits_and_hdf5/'+f'map_{name}.fits', overwrite=True)
                hdu.close()

class wcs_world():

    '''
    Class to generate a wcs using astropy routines.

    Parameters
    ----------
    Returns
    -------
    '''
    def __init__(self, ctype, crpix, cdelt, crval, telcoord=False):

        self.ctype = ctype    #ctype of the map, which projection is used to convert coordinates to pixel numbers
        self.cdelt = cdelt  #cdelt of the map, distance in deg between two close pixels
        self.crpix = crpix    #crpix of the map, central pixel of the map in pixel coordinates
        self.crval = crval    #crval of the map, central pixel of the map in sky/telescope (depending on the system) coordinates
        self.telcoord = telcoord #Telescope coordinates boolean value. Check map class for more explanation

    def world(self, coord1, coord2, parang): 
        
        '''
        Function for creating a wcs projection and a pixel coordinates 
        from sky/telescope coordinates
        Parameters
        ----------
        coord1: list
            list of timestreams of sky coordinates 1 
        coord2: list
            list of timestreams of sky coordinates 2
        parang: array
            list of parallactic angle in degree. 
        Returns
        -------
        world: list
            pixel projection of coord1 and coord2 given w
        w: wcs object
            the world coordinate system object of Astropy
        '''        

        w = wcs.WCS(naxis=2)
        w.wcs.crpix = self.crpix #wo.wcs.crpix
        w.wcs.cdelt = self.cdelt
        w.wcs.crval = self.crval

        if self.telcoord is False: w.wcs.ctype = ["TLON-CAR", "TLAT-CAR"]
        if self.ctype == 'RA and DEC':  w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        elif self.ctype == 'AZ and EL': w.wcs.ctype = ["TLON-ARC", "TLAT-ARC"]
        elif self.ctype == 'CROSS-EL and EL': w.wcs.ctype = ["TLON-CAR", "TLAT-CAR"]

        world = []
        for c1,c2 in zip(coord1, coord2):
            world.append(  w.world_to_pixel_values(c1,c2) )
        # w.world_to_pixel_values(pointing_paths[detector][:,0]+hdr['CRVAL1'], pointing_paths[detector][:,1])    
        
        return world, w

class mapmaking(object):

    '''
    Class to generate the maps. 
    Parameters
    ----------
    Returns
    -------
    
    '''

    def __init__(self, data, weight, number, pixelmap, coadd):

        self.data = data               #detector TOD
        self.weight = weight           #weights associated with the detector values
        self.number = number           #Number of detectors to be mapped
        self.pixelmap = pixelmap       #Coordinates of each point in the TOD in pixel coordinates
        self.coadd = coadd       #If to coadd all the detectors maps or return their individual maps. 

    def map_Ionly(self, coadd=False, value=None, noise=None, pixelmap = None):
        
        '''
        Function to create the 2D map
        Parameters
        ----------
        coadd: bool
            to return the coadd map between all detectors or the individual maps. 
        value: list
            amplitude timestreams of the detectors
        noise: array
            list of the noise in the detectors
        pixelmap: list
            list of pixel coordinates timestreams of the detectors
        Returns
        -------
        '''

        if value is None: value = self.data.copy()
        else: value = value

        if pixelmap is None: pixelmap = self.pixelmap.copy()
        else: pixelmap = pixelmap
        
        if noise is None:  noise = self.weight**2
        else: noise = noise

        import pickle

        Xmin = np.inf
        Xmax = -np.inf
        Ymin = np.inf
        Ymax = -np.inf

        for i in range(self.number):
            idxpixel = self.pixelmap[i]
            
            # Extract min and max for x and y
            xmin, xmax = idxpixel[0].min(), idxpixel[0].max()
            ymin, ymax = idxpixel[1].min(), idxpixel[1].max()
            
            # Update global min and max
            Xmin = min(Xmin, xmin)
            Xmax = max(Xmax, xmax)
            Ymin = min(Ymin, ymin)
            Ymax = max(Ymax, ymax)
            
        edges = np.round((Xmin, Xmax, Ymin, Ymax))
        X_edges = np.arange(edges[0]-0.5, edges[1]+1.5,1)        
        Y_edges = np.arange(edges[2]-0.5, edges[3]+1.5,1)        

        samples = []
        coord1samples = []
        coord2samples = []
        individual_maps = []

        for pix, val, n, i in zip(self.pixelmap, value, noise, range(self.number)):
            #------
            if n !=0: sigma = 1/n**2
            else: sigma = 1
            #val *= sigma
            samples.append(val)
            coord1samples.append(pix[0])
            coord2samples.append(pix[1])
            hits, x_edges, y_edges = np.histogram2d(pix[0], pix[1], bins = (X_edges, Y_edges) )
            flux, x_edges, y_edges = np.histogram2d(pix[0], pix[1], bins = (X_edges, Y_edges), weights=val )
            w = np.where(hits>0)
            flux[w] /= hits[w]
            individual_maps.append(flux.T)

        if not coadd: return individual_maps
        else: 
            norm, edges = np.histogramdd(sample=(np.concatenate(coord1samples), np.concatenate(coord2samples)), bins= (X_edges, Y_edges)  )
            hist, edges = np.histogramdd(sample=(np.concatenate(coord1samples), np.concatenate(coord2samples)),  bins= (X_edges, Y_edges), weights=np.concatenate(samples))
            hist /= norm
            return hist.T

    def convolution(self, std, map_value):

        '''
        Function to convolve the maps with a gaussian.
        Parameters
        ----------
        std: float
            std of the gaussian in pixel values
        Returns
        -------
        '''

        kernel = Gaussian2DKernel(x_stddev=std)

        convolved_map = convolve(map_value, kernel)

        return convolved_map

    