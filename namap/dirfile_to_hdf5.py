            
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
import h5py
import os
import tracemalloc
import time
from progress.bar import Bar

def check(d):

  """
  Check if all the entries on the dirfile are of known types.
  Parameters
  ----------
  d: string, 
      the name of the dirfile       
  Returns
  -------
  """
  
  for f, field in enumerate(d.field_list()):

  dect = d.entry(field)

  item_str =field.decode(encoding="utf-8")
  
  if dect.field_type == gd.RAW_ENTRY: 
    print(item_str, dect.spf)
  elif(dect.field_type == gd.PHASE_ENTRY): 
    try:
        #print(dect.in_fields, dect.shift)
        print(field,(dect.in_fields, dect.shift))
    except IOError as e:
      # Handle IOError and continue
      print(f"IOError for field {field}: {e}")
    except Exception as e:
      # Handle other unexpected errors (optional)
      print(f"Unexpected error for field {field}: {e}")
  
  elif(dect.field_type == gd.BIT_ENTRY):
    try:
        #print(dect.in_fields[0],dect.bitnum,dect.numbits)
        print(field,(dect.in_fields,dect.bitnum,dect.numbits))

    except IOError as e:
      # Handle IOError and continue
      print(f"IOError for field {field}: {e}")
    except Exception as e:
      # Handle other unexpected errors (optional)
      print(f"Unexpected error for field {field}: {e}")
  
  elif(dect.field_type == gd.LINCOM_ENTRY): 
    try:
        print(dect.n_fields, dect.m, dect.b,)
    except IOError as e:
      # Handle IOError and continue
      print(f"IOError for field {field}: {e}")
    except Exception as e:
      # Handle other unexpected errors (optional)
      print(f"Unexpected error for field {field}: {e}")
  
  elif(dect.field_type == gd.DIVIDE_ENTRY): 
    try:
        print(dect.in_fields[0],dect.in_fields[1])

    except IOError as e:
      # Handle IOError and continue
      print(f"IOError for field {field}: {e}")
    except Exception as e:
      # Handle other unexpected errors (optional)
      print(f"Unexpected error for field {field}: {e}")

  elif(dect.field_type == gd.LINTERP_ENTRY): 
    print(dect.table, dect.in_fields[0])

  elif(dect.field_type == gd.MPLEX_ENTRY): 
    print(dect.in_fields, dect.count_val, dect.period)

  elif(dect.field_type == gd.SINDIR_ENTRY): 
    print(dect.in_fields)

  elif(dect.field_type == gd.STRING_ENTRY): 
    print(field)

  elif(dect.field_type == gd.SARRAY_ENTRY): 
    print(dect.array_len)

  elif(dect.field_type == gd.INDEX_ENTRY): 
    print(field)

  else: print(f'{f} {field} is of type {dect.field_type_name}')
    
def hdf5_loaddata(file, field, num_frames=7383, first_frame=72273):
    """
    Equivalent build-in method d.getdata() for the .hdf5 file format. 
    
    Parameters
    ----------
    file: str
        the name of the .hdf5 file        
    field: float
        the name of the data field to be loaded, with N=spf samples in each frame.
    num_frames: int
        the number of frames to be loaded
    first_frame: int
        the index of the first frame to be loaded   

    Returns
    -------
    data: array, data stored in the field. 
    """
    
    H = h5py.File(file, "a")
    f = H[field]
    if('spf' in f.keys()):
      spf = f['spf'][()]
      data = f['data'][first_frame*spf:(first_frame+num_frames)*spf]
    else: 
      data = f['data'][()]
    H.close()
    return data

def dirfile_to_hdf5(dirfile, hdf5file, list_keys, fmin=0, fmax=-1):

  """
  Convert a dirfile in .hdf5 format
  
  Parameters
  ----------
  dirfile: string
      the name of the dirfile to be converted to .hdf5       
  hdf5file: string
      the name of the output .hdf5 file.
  list_keys: array of string
      if a field contains one of the keys listed, it is converted to .hdf5. 
  fmin: int
      the index of the first field to convert.
  fmax: int
      the index of the last field to convert. 

  Returns
  -------
  """

  d = gd.dirfile(dirfile, gd.RDONLY)
  bar = Bar('Saving fields from dirfile to hdf5: ', max=len(d.field_list()[fmin:fmax]))

  H = h5py.File(hdf5file, "a")
  H.create_dataset('nframes', data=d.nframes)

  for f, field in enumerate(d.field_list()):
    bar.next()
    #Check if the field is not corrupted
    try:
      d.validate(field)
    except gd.BadCodeError as e:
      #print(f'In validate({field}), caught exception of type {type(e).__name__}, arguments {e.args}')
      continue
    #Load info on the field
    dect = d.entry(field)
    item_str =field.decode(encoding="utf-8")

    if len(list_keys)==0 or any(key in item_str for key in list_keys):

      try:
          data = d.getdata(field)
      except gd.IOError as e:
          #print(f'In getdata({field}), caught exception of type {type(e).__name__}, arguments {e.args}')
          continue
      except gd.DimensionError as e:
          #print(f'In getdata({field}), caught exception of type {type(e).__name__}, arguments {e.args}')
          continue
      
      grp = H.create_group(item_str)
      grp.create_dataset('data', data=data, compression='gzip', compression_opts=9)

      try: 
        d.spf(field)
        grp.create_dataset('spf', data=d.spf(field))#dect.spf)
      except gd.DimensionError as e:
          #print(f'In validate({field}), caught exception of type {type(e).__name__}, arguments {e.args}')
          continue

      if dect.field_type == gd.RAW_ENTRY: grp.create_dataset('type', data='raw')
      elif(dect.field_type == gd.PHASE_ENTRY): 
        grp.create_dataset('in_fields', data=dect.in_fields)
        grp.create_dataset('shift', data=dect.shift)
        grp.create_dataset('type', data='phase')
      elif(dect.field_type == gd.BIT_ENTRY):
        grp.create_dataset('_in_fields',dect.in_fields)
        grp.create_dataset('_bitnum',dect.bitnum)
        grp.create_dataset('_numbits',dect.numbits)
        grp.create_dataset('type', data='bit')
      elif(dect.field_type == gd.LINCOM_ENTRY): 
        grp.create_dataset('n_fields',dect.n_fields)
        grp.create_dataset('m',dect.m)
        grp.create_dataset('b',dect.b)
        grp.create_dataset('type', data='lincom')
      elif(dect.field_type == gd.DIVIDE_ENTRY): 
        grp.create_dataset('in_fields',dect.in_fields)
        grp.create_dataset('type', data='divide')
      elif(dect.field_type == gd.LINTERP_ENTRY): 
        grp.create_dataset('type', data='linterp')
        string_dt = h5py.string_dtype(encoding='utf-8')  # For h5py versions >=3.0
        # Data to store
        table = np.array([dect.table], dtype=object)
        dset = grp.create_dataset('table', table.shape, dtype=string_dt)
        # Store the strings in the dataset
        dset[:] = table
      elif(dect.field_type == gd.MPLEX_ENTRY): 
        grp.create_dataset('type', data='mplex')
        grp.create_dataset('count_val',dect.count_val)
        grp.create_dataset('period',dect.period)

        string_dt = h5py.string_dtype(encoding='utf-8')  # For h5py versions >=3.0
        # Data to store
        table = np.array([dect.in_fields], dtype=object)
        dset = grp.create_dataset('in_fields', table.shape, dtype=string_dt)
        # Store the strings in the dataset
        dset[:] = table 
      elif(dect.field_type == gd.SINDIR_ENTRY):
        string_dt = h5py.string_dtype(encoding='utf-8')  # For h5py versions >=3.0
        # Data to store
        table = np.array([dect.in_fields], dtype=object)
        dset = grp.create_dataset('in_fields', table.shape, dtype=string_dt)
        # Store the strings in the dataset
        dset[:] = table 
        grp.create_dataset('type', data='sindir')
      elif(dect.field_type == gd.STRING_ENTRY): grp.create_dataset('type', data='string')
      elif(dect.field_type == gd.SARRAY_ENTRY): 
        grp.create_dataset('array_len',(dect.array_len))
        grp.create_dataset('type', data='sarray')
      elif(dect.field_type == gd.INDEX_ENTRY): grp.create_dataset('type', data='index')
      else: 
        print(f'{f} {field} is of type {dect.field_type_name}')
        continue

  H.close()
  d.close()

if __name__ == "__main__":

  tracemalloc.start() 
  start_time = time.time()
  cut = 3000

  list_keys = ('roach' ,'time' , 'RA', 'DEC', 'ra', 'dec', 'el', 'EL','AZ', 'az', 'hwpr','lat','lst', 'stage', 'lat', 'lst')

  dirfile_to_hdf5('/home/mvancuyck/Desktop/master', "master.hdf5", list_keys)

  #print(f'This field is of type {dect.field_type_name} with data type {dect.data_type_name}')
  print("end --- %s seconds ---" % np.round((time.time() - start_time),2))
  current, peak = tracemalloc.get_traced_memory()
  print(f"Current memory usage: {current / 10**6:.2f} MB; Peak: {peak / 10**6:.2f} MB")
  tracemalloc.stop()

