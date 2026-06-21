import numpy as np
from astropy import wcs
from astropy.convolution import Gaussian2DKernel, convolve
from IPython import embed
import os
from astropy.io import fits
import datetime
import json

class maps():

    '''
    Wrapper class for the wcs_word class and the mapmaking class.
    Parameters
    ----------
    Returns
    -------
    '''

    def __init__(self, ctype, crpix, cdelt, crval, pixnum, data, coord1, coord2, convolution, std, output_file, DT,IT,coadd=False, variance_weighting=False, parang=None, params=None): #telcoord=False,
        '''
        Create an instance of maps
        Parameters
        ----------
        ctype: str
            coordinates type (RA-DEC, AZ-EL, ect...)
        crpix: (float, float)
            coordinates of the reference pixel, usually the center of the map. 
        cdelt: (float, float)
            the pixel size in the y and x direction
        crval: (float, float)
            Sky coordinates at the reference pixel
        pixnum: (int, int)
            the maximum pixel sizes in the y and x direction allowed for the maps. 
        data: list
            list of the detector data
        coord1: list
            coordinate 1 for each detector data
        coord2: list
            coordinate 2 for each detector data
        convolution: bool
            If True, convolve the maps
        std: float
            std of the beam to convolve the maps
        output_file: str
            the path, name and format to save the maps. 
        DT: type
            Float precision required
        IT: type
            Int precision required
        coadd: bool
            If True, coadd all the detectors provided.
        variance_weighting: bool
            If True, compute the variance of each detector and weight its data by it in the map. 
        parang: 1d array
            The paralactic angle
        params: dictionnary
            The parameter dictionnary will be saved in 'COMMENTS' of the header
        
        Returns
        -------
        '''

        self.ctype = ctype             #see wcs_world for explanation of this parameter
        self.crpix = crpix             #see wcs_world for explanation of this parameter
        self.cdelt = cdelt             #see wcs_world for explanation of this parameter
        self.crval = crval             #see wcs_world for explanation of this parameter
        self.pixnum = pixnum           #Max number of pixel
        self.data = data               #cleaned TOD that is used to create a map
        self.coord1 = coord1           #array of the first coordinate
        self.coord2 = coord2           #array of the second coordinate
        self.params = params           #parameters used to create the map
        self.convolution = convolution #parameters to check if the convolution is required
        self.std = float(std)          #std of the gaussian is the convolution is required
        self.output_file = output_file #Name under which to save the coadd map.
        #self.telcoord = telcoord       #If True the map is drawn in telescope coordinates. That means that the projected plane is rotated
        self.DT = DT                    #Float precision required
        self.IT = IT                    #Integer precision required
        self.coadd = coadd       #If to coadd all the detectors maps or return their individual maps. 
        self.variance_weighting = variance_weighting
        if parang is not None:
            self.parang = [np.radians(p) for p in parang ] #Parallactic Angle. This is used to compute the pixel indices in telescopes coordinates
        else:
            self.parang = parang
        self.w = 0.                    #initialization of the coordinates of the map in pixel coordinates
        self.proj = 0.                 #inizialization of the wcs of the map. see wcs_world for more explanation about projections

    def wcs_proj(self):

        '''
        Function to compute the projection and the pixel coordinates
        Parameters
        ----------
        Returns
        -------
        '''
        wcsworld = wcs_world(self.ctype, self.crpix, self.cdelt, self.crval, self.DT, self.IT,)
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
        
        if(self.variance_weighting): weights = [np.std(d) for d in self.data] 
        else:                        weights = np.ones(len(self.data))
        
        mapmaker = mapmaking(self.data, weights, len(self.data), self.proj, self.coadd, self.DT, self.IT) # self.noise,
        Pow_map, crpix = mapmaker.map_Ionly( crpix = self.crpix, pixnum = self.pixnum, coadd=self.coadd,)
        
        self.w.wcs.crpix = crpix

        if not self.convolution: return Pow_map
        else:
            std_pixel = self.std/3600./self.cdelt[0]
            return mapmaker.convolution(std_pixel, Pow_map)
        
    def map_plot(self, data_maps, kid_num):

        """
        Save the map out of the data timestreams.     
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
        '''
        if self.telcoord :
            xlab = 'YAW (deg)'
            ylab = 'PITCH (deg)'
        '''


        if(self.coadd):

            '''
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
            '''
            f = fits.PrimaryHDU(data_maps, header=self.w.to_header())
            hdu = fits.HDUList([f])
            hdr = hdu[0].header
            hdr.set("map")
            hdr.set("Datas")
            hdr["BITPIX"] = ("64", "array data type")
            hdr["BUNIT"] = 'MJy/sr'
            hdr["DATE"] = (str(datetime.datetime.now()), "date of creation")
            hdr["INFO"] = json.dumps(self.params, ensure_ascii=True)
            hdu.writeto(self.output_file, overwrite=True) # os.getcwd()+'/fits_and_hdf5/'+
            hdu.close()
            print(f'Saved the coadded map {self.output_file}')

        else: 
            filename = self.output_file
            name_before_fits = filename.rsplit('.fits', 1)[0]
            fits_and_after = filename[filename.find('.fits'):]  

            for m, name in zip(data_maps, kid_num): 

                '''
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
                plt.show()
                '''
                f = fits.PrimaryHDU(m, header=self.w.to_header())
                hdu = fits.HDUList([f])
                hdr = hdu[0].header
                hdr.set("map")
                hdr.set("Datas")
                hdr["INFO"] = json.dumps(self.params, ensure_ascii=True)
                hdr["BITPIX"] = ("64", "array data type")
                hdr["BUNIT"] = 'MJy/sr'
                hdr["DATE"] = (str(datetime.datetime.now()), "date of creation")

                hdu.writeto(name_before_fits+'_'+name+fits_and_after, overwrite=True)
                print(f"Saved individual map {name_before_fits+'_'+name+fits_and_after}")
                hdu.close()

class wcs_world():

    '''
    Class to generate a wcs using astropy routines.

    Parameters
    ----------
    Returns
    -------
    '''
    def __init__(self, ctype, crpix, cdelt, crval, DT, IT):
        '''
        create an instance of the class to generate a wcs.

        Parameters
        ----------
        ctype: str
            ctype of the map, which projection is used to convert coordinates to pixel numbers
        cdelt: str
            cdelt of the map, distance in deg between two close pixels
        crpix: str
            crpix of the map, central pixel of the map in pixel coordinates
        crval: str
            crval of the map, central pixel of the map in sky/telescope (depending on the system) coordinates
        DT: type
            Float precision required
        IT: type
            Integer precision required
        Returns
        -------
        '''

        self.ctype = ctype    #ctype of the map, which projection is used to convert coordinates to pixel numbers
        self.cdelt = cdelt  #cdelt of the map, distance in deg between two close pixels
        self.crpix = crpix    #crpix of the map, central pixel of the map in pixel coordinates
        self.crval = crval    #crval of the map, central pixel of the map in sky/telescope (depending on the system) coordinates
        #self.telcoord = telcoord #Telescope coordinates boolean value. Check map class for more explanation
        self.DT = DT #Float precision required
        self.IT = IT #Integer precision required

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

        #if self.telcoord is False: w.wcs.ctype = ["TLON-CAR", "TLAT-CAR"]
        if self.ctype == 'RA and DEC':  w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        elif self.ctype == 'AZ and EL': w.wcs.ctype = ["TLON-ARC", "TLAT-ARC"]
        elif self.ctype == 'CROSS-EL and EL': w.wcs.ctype = ["TLON-CAR", "TLAT-CAR"]
        world = []

        for c1,c2 in zip(coord1, coord2):
            #world.append( w.world_to_pixel_values(c1,c2) )

            x_pix, y_pix = w.world_to_pixel_values(c1, c2)
            
            # Convert both arrays to DT
            x_pix = np.array(x_pix, dtype=self.DT)
            y_pix = np.array(y_pix, dtype=self.DT)
            
            world.append((x_pix, y_pix))   

        return world, w

class mapmaking(object):

    '''
    Class to generate the maps. 
    Parameters
    ----------
    Returns
    -------
    
    '''

    def __init__(self, data, weight, number, pixelmap, coadd, DT, IT):

        '''
        Create an instance of the class to generate the maps. 
        Parameters
        ----------
        data: list    
            detector TOD
        weight: array       
            weights associated with the detector values
        number: int
            Number of detectors to be mapped
        pixelmap: array  
            Coordinates of each point in the TOD in pixel coordinates
        coadd: bool
            If to coadd all the detectors maps or return their individual maps. 
        Returns
        -------
        
        '''

        self.data = data               #detector TOD
        self.weight = weight           #weights associated with the detector values
        self.number = number           #Number of detectors to be mapped
        self.pixelmap = pixelmap       #Coordinates of each point in the TOD in pixel coordinates
        self.coadd = coadd       #If to coadd all the detectors maps or return their individual maps. 
        self.DT = DT
        self.IT = IT

    def map_Ionly(self, crpix, pixnum, coadd=False, value=None, var=None, pixelmap = None):
        
        '''
        Function to create the 2D map
        Parameters
        ----------
        coadd: bool
            to return the coadd map between all detectors or the individual maps. 
        value: list
            list of the detector data
        noise: array
            list of the noise in the detectors
        pixelmap: list
            list of pixel coordinates timestreams of the detectors
        Returns
        -------
        '''

        if value is None: value = self.data.copy()

        if pixelmap is None: pixelmap = self.pixelmap.copy()
        
        if var is None: var = self.weight**2


        Xmin = np.inf
        Xmax = -np.inf
        Ymin = np.inf
        Ymax = -np.inf

        # --------------------------------------------- 
        # Compute extrema from your pixel list
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

        edges = np.round((Xmin, Xmax, Xmin, Ymax)) #np.round((Xmin, Xmax, Ymin, Ymax))

        # ---------------------------------------------
        # 2) Enforce that the cutout cannot exceed pixnum
        # --------------------------------------------- 
        
        cut_width  = min(edges[1] - edges[0],  pixnum[0])
        cut_height = min(edges[3] - edges[2],  pixnum[1])
        
        # ---------------------------------------------
        # 3) Center cutout on the extrema
        # ---------------------------------------------
        cx = (edges[0] + edges[1]) / 2
        cy = (edges[2] + edges[3]) / 2

        idx_xmin = int(np.floor(cx - cut_width/2))
        idx_xmax = idx_xmin + cut_width

        idx_ymin = int(np.floor(cy - cut_height/2))
        idx_ymax = idx_ymin + cut_height
                 
        # ---------------------------------------------
        # 5) Update WCS: crpix 
        # ---------------------------------------------
        # Shift crpix into new cutout
        
        crpix[0] -= idx_xmin
        crpix[1] -= idx_ymin
                
        # ---------------------------------------------
        # 6) Build the final edges vectors
        # ---------------------------------------------
        X_edges = np.arange(idx_xmin - 0.5, idx_xmax + 1.5, 1).astype(self.DT)
        Y_edges = np.arange(idx_ymin - 0.5, idx_ymax + 1.5, 1).astype(self.DT)
        
        samples = []
        coord1samples = []
        coord2samples = []
        individual_maps = []

        for pix, val, v, i in zip(self.pixelmap, value, var, range(self.number)):
            #------
            if v!=0: sigma = 1/v**2
            else: sigma = 1
            val *= sigma
            if(coadd): samples.append(val)
            if(coadd): coord1samples.append(pix[0])
            if(coadd): coord2samples.append(pix[1])
            hits, x_edges, y_edges = np.histogram2d(pix[0], pix[1], bins = (X_edges, Y_edges) )
            flux, x_edges, y_edges = np.histogram2d(pix[0], pix[1], bins = (X_edges, Y_edges), weights=val )
            flux /= hits
            individual_maps.append(flux.T)


        if not coadd: return individual_maps, crpix
        else: 
            norm, edges = np.histogramdd(sample=(np.concatenate(coord1samples), np.concatenate(coord2samples)), bins= (X_edges, Y_edges)  )
            hist, edges = np.histogramdd(sample=(np.concatenate(coord1samples), np.concatenate(coord2samples)),  bins= (X_edges, Y_edges), weights=np.concatenate(samples))
            hist /= norm
            return hist.T, crpix

    def convolution(self, std, map_value):

        '''
        Function to convolve the maps with a gaussian.
        Parameters
        ----------
        std: float
            std of the gaussian in pixel values
        map_values: 2d array
            the map to be convolved
        Returns
        -------
        convolved_map: 2d array
            the convolved map
        '''

        kernel = Gaussian2DKernel(x_stddev=std)

        convolved_map = convolve(map_value, kernel)

        return convolved_map

    
