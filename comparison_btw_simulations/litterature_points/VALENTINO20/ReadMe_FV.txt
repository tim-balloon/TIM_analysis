================================================================================
Title: The properties of the interstellar medium of galaxies across time 
       as traced by the neutral atomic carbon [C I]
Authors: Valentino F., Magdis G.E., Daddi E., Liu D., Aravena M., Bournaud F., 
         Cortzen I., Gao Y., Jin S., Juneau S., Kartaltepe J.S., Kokorev V., 
	 Lee M.-Y., Madden S.C., Narayanan D., Popping G., Puglisi A. 
================================================================================
Description of contents: Two .fits table files. The files are:

  table_ISM19_highz_release_V2.fits
  table_ISM19_local_LIRGs_release_standard_V2.fits

System requirements: Any FITS capable reader.

Additional comments: 
 table_ISM19_highz_release_V2.fits contains information about the high-redshift sample.
 table_ISM19_local_LIRGs_release_standard_V2.fits contains information about the local sample.

Each table contains:

 ID: a galaxy identifier (ID)
 ZSPEC: spectroscopic redshift
 LIR(8-1000um) (solLum): Total IR luminosity integrated within 8-1000 um, corrected for the dusty torus emission if the galaxy is not AGN dominated, and its error
 TDUST (K): Dust temperature from a optically thin modified black body model of the IR emission, and its error
 L’(LINE) (K.km/s/pc2), I(LINE) (Jy.km/s): L’ line luminosities and velocity-integrated line fluxes, and their errors for the lines:
 CI 1-0
 CI 2-1
 CO 1-0
 CO 2-1
 CO 3-2
 CO 4-3
 CO 5-4
 CO 6-5 
 CO 7-6
 CO 8-7
 CO 9-8
 CO 10-9
 CO 11-10
 CII
Facility: Facility used to detect the [CI] line emission
References: References to the original works describing the data for each galaxy



The table for the high-redshift sample (table_ISM19_highz_release_V2.fits) further contains: 

MAGNIFICATIONFACTOR_DUST, MAGNIFICATIONFACTOR_GAS: Magnification factors for lensed sources from dust and molecular gas observations, and their errors
FAGN: Fraction of IR emission due to dusty tori 
TYPE: Galaxy type (MS=Main Sequence; SB=StarBurst; AGN/QSO=Active Galactic Nucleus/Quasar)



The table for the local sample (table_ISM19_local_LIRGs_release_standard_V2.fits) further contains:

 D (Mpc): Distance
 AGN: Galaxy with AGN contamination and an entry in Veron-Cetty & Veron (2010)

================================================================================
