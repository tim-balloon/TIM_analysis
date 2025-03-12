import numpy as np
import matplotlib.pyplot as plt 
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord

#--- Dan Functions ---- 

def zenithAngle(dec,lat,HA):
    """
    source zenith angle (rad)

    Parameters
    ----------
    dec: float 
        declination angle in degrees     
    lat: float
        lattitude angle in degrees
    HA: array
        hour angle in degree

    Returns
    -------
    za: array
        zenith angle in degree
    """ 
    HArad = HA * np.pi/12; dec = np.radians(dec) ; lat = np.radians(lat)
    za = np.arccos(np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(HArad))
    return za

def elevationAngle(dec,lat,HA): 
    """
    elevation angle (rad)

    Parameters
    ----------
    dec: float 
        declination angle in degrees     
    lat: float
        lattitude angle in degrees
    HA: array
        hour angle in degree

    Returns
    -------
    ea: array
        elevation angle in degree
    """ 

    return np.pi/2 - zenithAngle(dec,lat,HA)

def azimuthAngle(dec,lat,HA):
    """
    source azimuth angle (rad)

    Parameters
    ----------
    dec: float 
        declination angle in degrees     
    lat: float
        lattitude angle in degrees
    HA: array
        hour angle in degree

    Returns
    -------
    aa: array
        source azimuth angle (rad)
    """ 

    za = zenithAngle(dec,lat,HA)
    HArad = HA * np.pi/12; dec = np.radians(dec) ; lat = np.radians(lat)
    cosAz = (np.sin(dec) - np.sin(lat) * np.cos(za))/(np.cos(lat) * np.sin(za))
    sinAz = np.sin(HArad) * np.cos(dec) / np.sin(za)
    return np.arctan2(sinAz,cosAz)

def parallacticAngle(dec,lat,HA,unwrapPA=True):
    """
    source parrallactic angle (rad)

    Parameters
    ----------
    dec: float 
        declination angle in degrees     
    lat: float
        lattitude angle in degrees
    HA: array
        hour angle in degree

    Returns
    -------
    pa: array
        source parrallactic angle (rad)
    """ 
    HArad = HA * np.pi/12; dec = np.radians(dec) ; lat = np.radians(lat)
    pa = np.arctan2(np.sin(HArad), np.cos(dec) * np.tan(lat) - np.sin(dec) * np.cos(HArad))
    if unwrapPA == True:
        pa = np.unwrap(pa)
        if np.mean(pa)<0:
            pa = pa + 2*np.pi
    return pa

def declinationAngle(azi, alt, lat):
    """
    source declination angle (rad)

    Parameters
    ----------
    azi: float 
        azimuth in degrees     
    alt: float
        lattitude angle in degrees
    lat: float
        lattitude angle in degree

    Returns
    -------
    Dec: float
        source declination angle (rad)
    """ 

    azi = np.radians(azi); alt = np.radians(alt); lat = np.radians(lat)
    sinDec = np.sin(alt)*np.sin(lat) + np.cos(alt)*np.cos(lat)*np.cos(azi)
    return np.arcsin(sinDec)

def hourAngle(azi, alt, lat):
    """
    source hour angle (rad)

    Parameters
    ----------
    azi: float 
        azimuth in degrees     
    alt: float
        lattitude angle in degrees
    lat: float
        lattitude angle in degree

    Returns
    -------
    ha: float
        source hour angle (rad)
    """ 
    dec = declinationAngle(azi, alt, lat)
    azi = np.radians(azi); alt = np.radians(alt); lat = np.radians(lat)
    cosHA = ( np.sin(alt)- np.sin(dec)*np.sin(lat) )/ (np.cos(dec)*np.cos(lat))
    return np.arccos(cosHA)*np.where((azi > 0)&(azi<=180), 1, -1)

if __name__ == "__main__":

    # HA = np.arange(-5,5,0.04)
    # lat=np.radians(19.82547)
    # dec=np.radians(16.14821)
    # pa = parallacticAngle(dec,lat,HA)
    # el = elevationAngle(dec,lat,HA)
    # za = zenithAngle(dec,lat,HA)

    mcmLat = -77.83
    minLat = -85
    maxLat = -75
    latList = np.arange(minLat,maxLat,2.)
    from astropy.coordinates import SkyCoord

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
    plt.savefig('figs/mcmPA')

    plt.clf()
    for latVal in latList:
        elVals = elevationAngle(goodsSDec,latVal,HA)
        paVals = parallacticAngle(goodsSDec,latVal,HA)
        plt.plot(HA[elVals>elMin],np.degrees(paVals[elVals>elMin]),label=str(latVal))
    plt.xlabel('Hour Angle (h)'); plt.xticks(np.arange(-6,6.1,1));
    plt.ylabel('Parallactic Angle (deg)')
    plt.legend(handlelength=1,loc='best',fontsize='small',title='Latitude')
    plt.title('GOODS-S PA')
    plt.savefig('figs/goodsPA')

    plt.clf()
    for latVal in latList:
        elVals = elevationAngle(sptDeepDec,latVal,HA)
        paVals = parallacticAngle(sptDeepDec,latVal,HA)
        plt.plot(HA[elVals>elMin],np.degrees(paVals[elVals>elMin]),label=str(latVal))
    plt.xlabel('Hour Angle (h)'); # plt.xticks(np.arange(-12,12.1,1));
    plt.ylabel('Parallactic Angle (deg)')
    plt.legend(handlelength=1,loc='best',fontsize='small',title='Latitude')
    plt.title('SPT-Deep PA')
    plt.savefig('figs/sptPA')
	
    plt.show()