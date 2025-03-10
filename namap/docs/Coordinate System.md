# Coordinate System 

The inflight naive-mapmaker for BLAST can use 3 different coordinate system to create maps

* RA-DEC: Right ascension and Declination
* AZ-EL: Azimuth and Elevation 
* xEL-EL: Cross-Elevation and Elevation

where the cross-elevation is defined as:

$$ xEL = AZ\cdot \cos{EL} $$

where the AZ angle is in radians.

The RA, DEC, AZ and EL values are extracted from the DIRFILE.

## Projection

The naive-mapmaker uses Astropy to create a world coordinate system (WCS) and then to save the maps in .fits file.

For the  RA and DEC map, the projection used is gnonomic projection (TAN), so the FITS file ctype is ***RA---TAN, DEC--TAN***. For AZ-EL and xEL-EL maps the coordinates system used is TLON-TLAT. However, the projection for the two maps is different. Indeed, for the AZ-EL map is used the zenithal/azimuthal equidistant (ARC) so the resulting ctype is ***TLON-ARC,TLAT-ARC***. Finally, for the xEL-EL map the projection used is the cartesian one (CAR) with a resulting cytpe of ***TLON-CAR,TLAT-CAR***.

## Offset Calculation

In order to compute the offset between the centroid of the map (see Shariff, PhD Thesis, 2016) and a reference point, the formula is based on the coordinate system chosen for the map (the offset calculation for AZ-EL map is not implemented yet). 

For the RA-DEC offset, the spherical distance is used and this one is computed using the astropy package. 

Instead for the xEL-EL map, the standard cartesian distance is used to compute the offset. 

