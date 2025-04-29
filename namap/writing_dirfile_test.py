            
import pygetdata as gd
import numpy as np
from IPython import embed
import src.detector as det
import src.loaddata as ld
import src.detector as tod
import src.mapmaker as mp
from src.gui import MapPlotsGroup
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from astropy.nddata import Cutout2D
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import array
import os
import tracemalloc
import sys

def B(s):
  if sys.version[:1] == '3':
    return bytes(s, "ASCII")
  return s

def load(file, filepath = '/home/mvancuyck/Desktop/master' ):

    '''
    Return the values of the DIRFILE as a numpy array
    
    filepath: path of the DIRFILE to be read
    file: name of the value to be read from the dirfile, e.g. detector name or
            coordinate name
    file_type: data type conversion string for the DIRFILE data
    '''
    
    d = gd.dirfile(filepath, gd.RDONLY)
    print(f'Load the {d.name} dirfile')
    gdtype = 132

    num = 7383
    first_frame = 72273

    values = d.getdata(file, gdtype, num_frames = num, first_frame=first_frame)

    return values

if __name__ == "__main__":

    # create the dirfile first
    data=array.array("B",range(1,81))
    os.system("rm -rf dirfile")
    os.mkdir("dirfile")
    file=open("dirfile/data", 'wb')
    data.tofile(file)
    file.close()

    ne = 0

    fields = [ B("bit"), B("div"), B("data"), B("mult"), B("sbit"), B("INDEX"),
    B("alias"), B("const"), B("indir"), B("mplex"), B("phase"), B("recip"),
    B("carray"), B("lincom"), B("sarray"), B("sindir"), B("string"),
    B("window"), B("linterp"), B("polynom"), ]

    nfields = 20
    file=open("dirfile/format", 'w')
    file.write(
    "/ENDIAN little\n"
    "data RAW INT8 8\n"
    "lincom LINCOM data 1.1 2.2 INDEX 2.2 3.3;4.4 linterp const const\n"
    "/META data mstr STRING \"This is a string constant.\"\n"
    "/META data mconst CONST COMPLEX128 3.3;4.4\n"
    "/META data mcarray CARRAY FLOAT64 1.9 2.8 3.7 4.6 5.5\n"
    "/META data mlut LINTERP DATA ./lut\n"
    "const CONST FLOAT64 5.5\n"
    "carray CARRAY FLOAT64 1.1 2.2 3.3 4.4 5.5 6.6\n"
    "linterp LINTERP data ./lut\n"
    "polynom POLYNOM data 1.1 2.2 2.2 3.3;4.4 const const\n"
    "bit BIT data 3 4\n"
    "sbit SBIT data 5 6\n"
    "mplex MPLEX data sbit 1 10\n"
    "mult MULTIPLY data sbit\n"
    "div DIVIDE mult bit\n"
    "recip RECIP div 6.5;4.3\n"
    "phase PHASE data 11\n"
    "window WINDOW linterp mult LT 4.1\n"
    "/ALIAS alias data\n"
    "string STRING \"Zaphod Beeblebrox\"\n"
    "sarray SARRAY one two three four five six seven\n"
    "data/msarray SARRAY eight nine ten eleven twelve\n"
    "indir INDIR data carray\n"
    "sindir SINDIR data sarray\n"
    )
    file.close()

    file=open("dirfile/form2", 'w')
    file.write("const2 CONST INT8 -19\n")
    file.close()

    # 1: error check
    outfile = gd.dirfile("dirfile", gd.RDONLY)
    outfile = gd.dirfile("dirfile", gd.RDWR)
    n = outfile.getdata("data", gd.INT, first_frame=5, num_frames=1)
    #print(f'There is n={outfile.nfields()} fields, frames = {outfile.nframes}, the list of fields are:')
    #print(outfile.field_list())

    p = [ 13, 14, 15, 16 ]
    m = outfile.putdata("data", p, gd.INT, first_frame=5, first_sample=1)
    ent = outfile.entry("data") #lincom polynom linterp bit sbit mult phase const string 

    ent = gd.entry(gd.RAW_ENTRY, "new1", 0, (gd.FLOAT64, 3))
    outfile.add(ent)
    ent = outfile.entry("new1")
