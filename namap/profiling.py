
import os
import time
import tracemalloc
from namap_main import *
import pickle
import copy
import io
import cProfile
import pstats
import matplotlib.pyplot as plt
import sysconfig, site, os
from astropy.io import fits
import platform
import psutil
from unittest import mock

import os
import psutil
import platform
from unittest import mock
import pickle
import time
import matplotlib.pyplot as plt
from astropy.io import fits
import os
import psutil
import platform
from unittest import mock
from collections import namedtuple
import glob
import os


# --- Intel Celeron 4305UE Specs ---
SIM_CPU = "Intel(R) Celeron(R) 4305UE @ 2.00GHz"
SIM_CORES = 2
SIM_THREADS = 2
SIM_MEMORY_GB = 8  # realistic config (max supported = 64 GB)
SIM_MEMORY_BYTES = SIM_MEMORY_GB * 1024**3


def simulate_celeron_system():
    """Return mocks for CPU and memory to simulate Intel Celeron 4305UE."""
    svmem = namedtuple('svmem', [
        'total', 'available', 'percent', 'used', 'free',
        'active', 'inactive', 'buffers', 'cached', 'shared', 'slab'
    ])
    fake_mem = svmem(
        total=SIM_MEMORY_BYTES,
        available=SIM_MEMORY_BYTES,
        percent=0,
        used=0,
        free=SIM_MEMORY_BYTES,
        active=0,
        inactive=0,
        buffers=0,
        cached=0,
        shared=0,
        slab=0
    )

    cpu_mock = mock.patch("os.cpu_count", return_value=SIM_CORES)
    mem_mock = mock.patch("psutil.virtual_memory", return_value=fake_mem)
    return cpu_mock, mem_mock

def profiling_functions():

    key = 'profiling fcts'
    results.setdefault(key, {})

    for precision in ('float64', 'float32'): 

        results[key][precision] = {}

        P = load_par_file(f'params_namap.par')
        P['hdf5_file'] = 'TODs_profiling/TOD_0h3min0sec_64bits_488Hz.hdf5' 
        P['cdelt'] = 40/3600, 40/3600
        freq_list = 715.0 + 4.0 * np.arange(63)
        P['frequencies'] = freq_list
        #P['frequencies'] = (715.0,) #GHz
        P['precision'] = precision

        #---------------- Precision -----------------------#
        _prec = str(P['precision'].lower())
        dtype_map = {
            '16': np.float16, 'float16': np.float16, 'half': np.float16,
            '32': np.float32, 'float32': np.float32, 'single': np.float32,
            '64': np.float64, 'float64': np.float64, 'double': np.float64
        }
        int_map = {
            '16': np.int16, 'float16': np.int16, 'half': np.int16,
            '32': np.int32, 'float32': np.int32, 'single': np.int32,
            '64': np.int64, 'float64': np.int64, 'double': np.int64
        }
        try:
            DT = dtype_map[_prec]
            IT = int_map[_prec]
        except KeyError:
            raise ValueError(f"Unsupported precision '{_prec}'. Choose float32/64 or 32/64.")
        print(f"Using numeric dtype: {DT}")
        #-----------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------
        #### start mathilde code 
        #---------------------------------
        num_frames, first_frame = P['num_frames'], P['first_frame']

        #Also need to be implemented. 
        telemetry = P['telemetry']

        #So far, only 'RA and DEC' is implemented and working.   

        if P['input_ctype'] == 'RA and DEC':
            coord1 = str('RA')
            coord2 = str('DEC')
            xystage = False
        elif P['input_ctype'] == 'AZ and EL':
            coord1 = str('AZ')
            coord2 = str('EL')
            xystage = False
        elif P['input_ctype'] == 'CROSS-EL and EL':
            coord1 = str('xEL')
            coord2 = str('EL')
            xystage = False
        elif P['input_ctype'] == 'XY Stage':
            coord1 = str('X')
            coord2 = str('Y')
            xystage = True

        filepath = P['hdf5_file']
        btable = tb.Table.read(P['detector_table'], format='ascii.tab')
        if P['frequencies'] is not None:
            filtered = btable[np.isin(btable['Frequency'], P['frequencies'])]
        if P['detectors_to_use'] is not None:
            good_kid_table = tb.Table.read(P['detectors_to_use'], format='ascii.tab')
            filtered = btable[np.isin(btable['Name'], good_kid_table['Name'])]
        if P['frequencies'] is None and P['detectors_to_use'] is None:
            filtered = btable
        #option in the par file to good kids list
        kid_num = filtered['Name']
        print('Nb dets: ', len(kid_num))
        #load the table
        dettable = ld.det_table(kid_num, P['detector_table']) 
        det_off, noise_det, resp = dettable.loadtable()
        #Cleaning data parameters
        highpassfreq = P['highpassfreq']
        polynomialorder = P['polynomialorder']
        despike_bool = P['despike']
        sigma,prominence = P['sigma'],P['prominence']
        #Beam convolution parameters
        convolution, std = P['gaussian_convolution'], P['std'] 
        #---------------------------------

        #-------------------------------------------------------------------------------------------------------------------------
        results[key][precision]['loaddata'] = {}
        start = time.time()
        for i in range(10):
            #Load the data
            dataload = ld.data_value(filepath, kid_num, coord1, coord2, first_frame, num_frames,  DT, IT,telemetry)
            _, _, _, _, _, _, _, _, _, _, _ = dataload.values()
        end = time.time()
        timing = end - start
        results[key][precision]['loaddata']['time [s]'] = timing / 10
        
        tracemalloc.start()
        dataload = ld.data_value(filepath, kid_num, coord1, coord2, first_frame, num_frames,  DT, IT,telemetry)
        det_data, coord1_data, coord2_data, lst_data, lat_data, spf_data, spf_coord, lat_spf, acqfreq_data, acqfreq_coord, acqfreq_lstlat = dataload.values()
        current, peak = tracemalloc.get_traced_memory()
        results[key][precision]['loaddata']['peak memory [MB]'] = peak/1e6
        tracemalloc.stop()

        #-------------------------------

        #---------------------------------
        #First remove noise peaks
        results[key][precision]['despiking'] = {}
        start = time.time()
        for i in range(10):
            det_tod = tod.data_cleaned(det_data, spf_data, kid_num, 0, 0, despike_bool, sigma, prominence)
            _ = det_tod.data_clean()
        end = time.time()
        timing = end - start
        results[key][precision]['despiking']['time [s]'] = timing / 10
        
        tracemalloc.start()
        det_tod = tod.data_cleaned(det_data, spf_data, kid_num, 0, 0, despike_bool, sigma, prominence)
        cleaned_data = det_tod.data_clean()
        current, peak = tracemalloc.get_traced_memory()
        results[key][precision]['despiking']['peak memory [MB]'] = peak/1e6
        tracemalloc.stop()
        #---------------------------------
                
        #---------------------------------
        results[key][precision]['zoom_sync'] = {}
        DATA = copy.deepcopy(cleaned_data)

        start = time.time()
        for i in range(10): cl = copy.deepcopy(DATA)
        end = time.time()
        time_deepcopy = (start-end)/10
                
        start = time.time()
        for i in range(10):
            
            cl = copy.deepcopy(DATA)
            zoomsyncdata = ld.frame_zoom_sync(filepath, cl, acqfreq_data, spf_data,  coord1_data, 
                                                coord2_data, acqfreq_coord, spf_coord, first_frame, num_frames, 
                                                lst_data, lat_data, acqfreq_lstlat, lat_spf, DT, IT, offset=0)

            _, _, _, _, _, _ = zoomsyncdata.sync_data()  

        
        end = time.time()
        timing = end - start
        results[key][precision]['zoom_sync']['time [s]'] = timing / 10 -time_deepcopy
        
        tracemalloc.start()
        zoomsyncdata = ld.frame_zoom_sync(filepath, cleaned_data, acqfreq_data, spf_data,  coord1_data, 
                                                coord2_data, acqfreq_coord, spf_coord, first_frame, num_frames, 
                                                lst_data, lat_data, acqfreq_lstlat, lat_spf, DT, IT, offset=0)
        timemap, cleaned_data, coord1slice, coord2slice, lstslice, latslice = zoomsyncdata.sync_data()  
        current, peak = tracemalloc.get_traced_memory()
        
        results[key][precision]['zoom_sync']['peak memory [MB]'] = peak/1e6
        tracemalloc.stop()
        #---------------------------------

    
        #---------------------------------
        results[key][precision]['zoom_sync_v2'] = {}
        start = time.time()
        for i in range(10):
            
            cl = copy.deepcopy(DATA)
            zoomsyncdata = ld.frame_zoom_sync(filepath, cl, acqfreq_data, spf_data,  coord1_data, 
                                                coord2_data, acqfreq_coord, spf_coord, first_frame, num_frames, 
                                                lst_data, lat_data, acqfreq_lstlat, lat_spf, DT, IT, offset=0)

            _, _, _, _, _, _ = zoomsyncdata.sync_data_v2()  
            
        end = time.time()
        timing = end - start
        results[key][precision]['zoom_sync_v2']['time [s]'] = timing / 10-time_deepcopy
        
        cl = copy.deepcopy(DATA)
        tracemalloc.start()
        zoomsyncdata = ld.frame_zoom_sync(filepath, cl, acqfreq_data, spf_data,  coord1_data, 
                                                coord2_data, acqfreq_coord, spf_coord, first_frame, num_frames, 
                                                lst_data, lat_data, acqfreq_lstlat, lat_spf, DT, IT, offset=0)
        _, _, _, _, _, _ = zoomsyncdata.sync_data_v2()  
        current, peak = tracemalloc.get_traced_memory()
        results[key][precision]['zoom_sync_v2']['peak memory [MB]'] = peak/1e6
        tracemalloc.stop()        
        #---------------------------------


        #---------------------------------
        results[key][precision]['zoom_sync_v3'] = {}
        start = time.time()
        for i in range(10):
            
            cl = copy.deepcopy(DATA)
            zoomsyncdata = ld.frame_zoom_sync(filepath, cl, acqfreq_data, spf_data,  coord1_data, 
                                                coord2_data, acqfreq_coord, spf_coord, first_frame, num_frames, 
                                                lst_data, lat_data, acqfreq_lstlat, lat_spf, DT, IT, offset=0)

            _, _, _, _, _, _ = zoomsyncdata.sync_data_v3()  
            
        end = time.time()
        timing = end - start
        results[key][precision]['zoom_sync_v3']['time [s]'] = timing / 10-time_deepcopy
        
        cl = copy.deepcopy(DATA)
        tracemalloc.start()
        zoomsyncdata = ld.frame_zoom_sync(filepath, cl, acqfreq_data, spf_data,  coord1_data, 
                                                coord2_data, acqfreq_coord, spf_coord, first_frame, num_frames, 
                                                lst_data, lat_data, acqfreq_lstlat, lat_spf, DT, IT, offset=0)
        _, _, _, _, _, _ = zoomsyncdata.sync_data_v3()  
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        results[key][precision]['zoom_sync_v3']['peak memory [MB]'] = peak/1e6
        
        #---------------------------------


        #---------------------------------
        results[key][precision]['zoom_sync_v4'] = {}
        start = time.time()
        for i in range(10):
            
            cl = copy.deepcopy(DATA)
            zoomsyncdata = ld.frame_zoom_sync(filepath, cl, acqfreq_data, spf_data,  coord1_data, 
                                                coord2_data, acqfreq_coord, spf_coord, first_frame, num_frames, 
                                                lst_data, lat_data, acqfreq_lstlat, lat_spf, DT, IT, offset=0)

            _, _, _, _, _, _ = zoomsyncdata.sync_data_v4()  
            
        end = time.time()
        timing = end - start
        results[key][precision]['zoom_sync_v4']['time [s]'] = timing / 10-time_deepcopy
        
        cl = copy.deepcopy(DATA)
        tracemalloc.start()
        zoomsyncdata = ld.frame_zoom_sync(filepath, cl, acqfreq_data, spf_data,  coord1_data, 
                                                coord2_data, acqfreq_coord, spf_coord, first_frame, num_frames, 
                                                lst_data, lat_data, acqfreq_lstlat, lat_spf, DT, IT, offset=0)
        _, _, _, _, _, _ = zoomsyncdata.sync_data_v4()  
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        results[key][precision]['zoom_sync_v4']['peak memory [MB]'] = peak/1e6
        #--------------------------------


        #---------------------------------
        results[key][precision]['polynome_fft_filter'] = {}
        
        start = time.time()
        for i in range(10):
            det_tod = tod.data_cleaned(cleaned_data, spf_data, kid_num, highpassfreq, polynomialorder, False, 0, 0)
            _ = det_tod.data_clean()
        end = time.time()
        timing = end - start
        results[key][precision]['polynome_fft_filter']['time [s]'] = timing / 10
        
        tracemalloc.start()
        #Clean the TOD by removing smooth polynomial component and apply a high pass filter
        det_tod = tod.data_cleaned(cleaned_data, spf_data, kid_num, highpassfreq, polynomialorder, False, 0, 0)
        cleaned_data = det_tod.data_clean()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        results[key][precision]['polynome_fft_filter']['peak memory [MB]'] = peak/1e6

        
        #Apply detector's response
        cleaned_data = [arr * resp for arr, resp in zip(cleaned_data, resp)]
        #---------------------------------

        #---------------------------------
        #Offset with respect to star cameras in xEL and EL
        xsc_offset = (P['xsc_offset'],P['det_offset']) #needs to be tested with real offsets. 
        #xsc_file = ld.xsc_offset(P['pointing_table'], first_frame, num_frames+first_frame)
        #xsc_offset = xsc_file.read_file()
        results[key][precision]['correct_coords'] = {}
        start = time.time()
        for i in range(10):
            corr = pt.apply_offset(P['input_ctype'], coord1slice, coord2slice, P['ctype'], xsc_offset, DT,IT, det_offset = det_off, lst = lstslice, lat = latslice, )
            _, _ = corr.correction()
        end = time.time()
        timing = end - start
        results[key][precision]['correct_coords']['time [s]'] = timing / 10

        tracemalloc.start()
        #Clean the TOD by removing smooth polynomial component and apply a high pass filter
        corr = pt.apply_offset(P['input_ctype'], coord1slice, coord2slice, P['ctype'], xsc_offset, DT,IT, det_offset = det_off, lst = lstslice, lat = latslice, )
        coord1slice, coord2slice = corr.correction()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        results[key][precision]['correct_coords']['peak memory [MB]'] = peak/1e6
        #---------------------------------

        #--------------------
        #Need to be implemented ! So far, set parallactic angle to 0.
        parallactic=[]
        if P['telescope_coordinate']:
            for j, (c1, c2) in enumerate(zip(coord1slice,coord2slice)): 
                tel = pt.utils(c1, c2, lstslice, latslice)
                parallactic.append( tel.parallactic_angle() )
        else:
            for j, (c1, c2) in enumerate(zip(coord1slice,coord2slice)): 
                parallactic.append(np.zeros_like(c1, dtype=DT))
        
        #---------------------------------

        #--------------------
                
        results[key][precision]['map_making'] = {}
        start = time.time()
        for i in range(10):
            maps = mp.maps(P['ctype'], np.asarray([P['crpix'][0],P['crpix'][1]]), np.asarray([P['cdelt'][0],P['cdelt'][1]]), np.asarray([P['crval'][0], P['crval'][1]]), np.asarray([P['pixnum'][0],P['pixnum'][1]]), 
                        cleaned_data, coord1slice, coord2slice, convolution, std, P['output_map'], DT,IT,
                        coadd=P['coadd'], noise=noise_det, telcoord = P['telescope_coordinate'], parang=parallactic, params=str(P))
            
            maps.wcs_proj()
            map_values = maps.map2d()
            maps.map_plot(data_maps = map_values, kid_num=kid_num)

        end = time.time()
        timing = end - start
        results[key][precision]['map_making']['time [s]'] = timing / 10

        tracemalloc.start()
        #Clean the TOD by removing smooth polynomial component and apply a high pass filter
        #Create the maps
        maps = mp.maps(P['ctype'], np.asarray([P['crpix'][0],P['crpix'][1]]), np.asarray([P['cdelt'][0],P['cdelt'][1]]), np.asarray([P['crval'][0], P['crval'][1]]), np.asarray([P['pixnum'][0],P['pixnum'][1]]), 
                    cleaned_data, coord1slice, coord2slice, convolution, std, P['output_map'], DT,IT,
                    coadd=P['coadd'], noise=noise_det, telcoord = P['telescope_coordinate'], parang=parallactic, params=str(P))
        maps.wcs_proj()
        map_values = maps.map2d()
        maps.map_plot(data_maps = map_values, kid_num=kid_num)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        results[key][precision]['map_making']['peak memory [MB]'] = peak/1e6

    if(False):

        P = load_par_file(f'params_namap.par')
        P['hdf5_file'] = 'TODs_profiling/TOD_0h3min0sec_64bits_488Hz.hdf5' 
        P['cdelt'] = 40/3600, 40/3600
        freq_list = 715.0 + 4.0 * np.arange(63)
        P['frequencies'] = freq_list
        P['frequencies'] = (715.0,) #GHz
        P['precision'] = 'float32'

        # Run main() under cProfile
        profiler = cProfile.Profile()
        profiler.enable()
        main(P)
        profiler.disable()

        # Print the stats
        stats = pstats.Stats(profiler)
        stats.strip_dirs()       # remove extraneous path info
        stats.sort_stats('cumulative')  # sort by cumulative time
        stats.print_stats(50)    # print top 50 functions
        profiler.dump_stats("main_profile.prof")
        

        MYCODE_PATH = "TIM_analysis/namap/"  # <-- change this to your code path

        # ---- Filter only your code (exclude stdlib & site-packages) ----
        stdlib_dir = sysconfig.get_path('stdlib')
        site_dir = site.getsitepackages()[0]

        func_names = []
        cum_times = []

        for func, (cc, nc, tt, ct, callers) in stats.stats.items():
            filename, line, funcname = func
            if('main' in funcname ): continue
            if not (filename.startswith(stdlib_dir) or filename.startswith(site_dir)):

                func_names.append(f"{os.path.basename(filename)}:{funcname}")
                print(f"{os.path.basename(filename)}:{funcname}")
                cum_times.append(ct)

        # ---- Sort by cumulative time (descending) ----
        sorted_pairs = sorted(zip(cum_times, func_names), reverse=True)
        cum_times, func_names = zip(*sorted_pairs)

        # ---- Plot ----
        plt.figure(figsize=(10, 6))
        plt.barh(func_names[:20], cum_times[:20])  # top 20
        plt.gca().invert_yaxis()  # most time-consuming at top
        plt.xlabel("Cumulative time (s)")
        plt.ylabel("Function")
        plt.title("Top Functions by Cumulative Execution Time")
        plt.tight_layout()
        plt.show()
        #Profile specific sub-functions individually by wrapping them in cProfile.Profile().
        #pip install snakeviz
        #snakeviz main_profile.prof

if(False): 

    import tracemalloc

    P = load_par_file(f'params_namap.par')
    P['hdf5_file'] = 'TODs_profiling/TOD_0h3min0sec_64bits_488Hz.hdf5' 
    P['cdelt'] = 40/3600, 40/3600
    freq_list = 715.0 + 4.0 * np.arange(63)
    #P['frequencies'] = freq_list
    P['frequencies'] = (715.0,) #GHz
    P['precision'] = 'float32'

    tracemalloc.start()
    main(P)
    snapshot = tracemalloc.take_snapshot()

    top_stats = snapshot.statistics('lineno')
    for stat in top_stats[:10]:  # top 10 memory consumers
        print(stat)

    with open(path, 'wb') as f: pickle.dump(results, f)

    
def test_I(profiling_vs_tod_time = True, profiling_vs_nb_of_detectors=True, profiling_vs_nb_bands=True):

        path = 'fits_and_hdf5/namap_perf_profiling_coadded_maps.p'

        if os.path.exists(path):
            with open(path, 'rb') as f:
                results = pickle.load(f)
        else:
            results = {}

        if(profiling_vs_tod_time):

            key = 'profiling vs tod time'
            results.setdefault(key, {})

            results[key]["t_int"] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 30)

            for precision in ('float64', 'float32'): 

                results[key][precision] = {}

                for map_compression in ('coadd.fits','coadd.fits.gz'):
                    
                    results[key][precision][map_compression] = {}    
                    results[key][precision][map_compression]['peak memory [MB]'] = []
                    results[key][precision][map_compression]['time [s]'] = []
                    results[key][precision][map_compression]['output size [MB]'] = []

                    for t in results[key]["t_int"]:

                        par = load_par_file(f'params_namap.par')
                        par['cdelt'] = 40/3600, 40/3600
                        par['frequencies'] = (715.0,) #GHz        
                        par['precision'] = precision
                        par['num_frames'] = t * 60 #seconds 
                        par['output_map'] = map_compression
                        par['coadd'] = True

                        #------------------------------------------------------
                        tracemalloc.start()
                        start = time.time()
                        main(par)
                        current, peak = tracemalloc.get_traced_memory()
                        tracemalloc.stop()
                        end = time.time()
                        timing = end - start
                        #------------------------------------------------------

                        # Measure output file size (adapt this path!)
                        output_file = 'fits_and_hdf5/'+par['output_map']  # or whatever your naming convention is
                        if os.path.exists(output_file): file_size_mb = os.path.getsize(output_file) / 1e6
                        else: file_size_mb = float('nan')  # file not found → record NaN or 0

                        print(f"time {t}min| time={timing:.2f}s | peak={peak/1e6:.2f}MB | output={file_size_mb:.2f}MB")

                        # Store results
                        results[key][precision][map_compression]['peak memory [MB]'].append(peak / 1e6)
                        results[key][precision][map_compression]['time [s]'].append(timing)
                        results[key][precision][map_compression]['output size [MB]'].append(file_size_mb)

        if(profiling_vs_nb_of_detectors):

            key = 'profiling vs nb of detectors'
            results.setdefault(key, {})

            results[key] = {}
            results[key]["nb detectors"] = (1,5,10,20, 30, 40, 50, 64)

            for precision in ('float64', 'float32'): 

                results[key][precision] = {}
                results[key][precision]['peak memory [MB]'] = []
                results[key][precision]['time [s]'] = []
                results[key][precision]['output size [MB]'] = []

                for nb in results['profiling vs nb of detectors']["nb detectors"]:

                    par = load_par_file(f'params_namap.par')
                    par['cdelt'] = 40/3600, 40/3600
                    par['frequencies'] = (715.0,) #GHz
                    par['precision'] = precision
                    par['num_frames'] = 5 * 60 #seconds 
                    par['coadd'] = True

                    #------------------------------------------------------
                    tracemalloc.start()
                    start = time.time()
                    main(par, nb)
                    current, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    end = time.time()
                    timing = end - start
                    #------------------------------------------------------

                    # Measure output file size (adapt this path!)
                    output_file = 'fits_and_hdf5/coadd.fits'  # or whatever your naming convention is
                    if os.path.exists(output_file): file_size_mb = os.path.getsize(output_file) / 1e6
                    else: file_size_mb = float('nan')  # file not found → record NaN or 0

                    print(f"Nb dets {nb}| time={timing:.2f}s | peak={peak/1e6:.2f}MB | output={file_size_mb:.2f}MB")

                    # Store results
                    results[key][precision]['peak memory [MB]'].append(peak / 1e6)
                    results[key][precision]['time [s]'].append(timing)
                    results[key][precision]['output size [MB]'].append(file_size_mb)

        if(profiling_vs_nb_bands):

            key = 'profiling vs nb bands'
            results.setdefault(key, {})
            results[key]["nb bands"] = (1,5, 10, 16, 20, 30, 40, 50, 64)

            for precision in ('float64', 'float32'): 

                results[key][precision] = {}
                results[key][precision]['peak memory [MB]'] = []
                results[key][precision]['time [s]'] = []
                results[key][precision]['output size [MB]'] = []


                for nb in results[key]["nb bands"]:

                    freq_list = 715.0 + 4.0 * np.arange(nb)

                    par = load_par_file(f'params_namap.par')
                    par['cdelt'] = 40/3600, 40/3600
                    par['frequencies'] = freq_list
                    par['precision'] = precision
                    par['num_frames'] = 5 * 60 #seconds 
                    par['coadd'] = True

                    #------------------------------------------------------
                    tracemalloc.start()
                    start = time.time()
                    main(par)
                    current, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    end = time.time()
                    timing = end - start
                    #------------------------------------------------------

                    # Measure output file size (adapt this path!)
                    output_file = 'fits_and_hdf5/coadd.fits'  # or whatever your naming convention is
                    if os.path.exists(output_file): file_size_mb = os.path.getsize(output_file) / 1e6
                    else: file_size_mb = float('nan')  # file not found → record NaN or 0

                    print(f"Nb bands {nb}| time={timing:.2f}s | peak={peak/1e6:.2f}MB | output={file_size_mb:.2f}MB")

                    # Store results
                    results[key][precision]['peak memory [MB]'].append(peak / 1e6)
                    results[key][precision]['time [s]'].append(timing)
                    results[key][precision]['output size [MB]'].append(file_size_mb)

        if(False): #XX

            key = 'profiling vs resolution'
            results.setdefault(key, {})  # ← preserve if already exists
            results[key] = {}
            results[key]["resolution ['']"] = (10, 20, 30, 40,50,60)

            for precision in ('float64','float32'): 

                results[key][precision] = {}
                results[key][precision]['peak memory [MB]'] = []
                results[key][precision]['time [s]'] = []
                results[key][precision]['output size [MB]'] = []
                    
                for res in results['profiling vs resolution']["resolution ['']"]:

                    par = load_par_file(f'params_namap.par')
                    par['hdf5_file'] = 'TODs_profiling/TOD_0h3min0sec_64bits_488Hz.hdf5' 
                    par['cdelt'] = res/3600, res/3600
                    par['frequencies'] = (715.0,) #GHz
                    par['precision'] = precision
                    par['coadd'] = True

                    #------------------------------------------------------
                    tracemalloc.start()
                    start = time.time()
                    main(par)
                    current, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    end = time.time()
                    timing = end - start
                    #------------------------------------------------------

                    # Measure output file size (adapt this path!)
                    output_file = 'fits_and_hdf5/coadd.fits'  # or whatever your naming convention is
                    if os.path.exists(output_file): file_size_mb = os.path.getsize(output_file) / 1e6
                    else: file_size_mb = float('nan')  # file not found → record NaN or 0

                    print(f"Resolution={res}'' | time={timing:.2f}s | peak={peak/1e6:.2f}MB | output={file_size_mb:.2f}MB")

                    # Store results
                    results[key][precision]['peak memory [MB]'].append(peak / 1e6)
                    results[key][precision]['time [s]'].append(timing)
                    results[key][precision]['output size [MB]'].append(file_size_mb)

        with open(path, 'wb') as f: pickle.dump(results, f)

def test_II(profiling_vs_tod_time = True, profiling_vs_nb_bands=True):

    path = 'fits_and_hdf5/namap_perf_profiling_individual_maps.p'

    if os.path.exists(path):
        with open(path, 'rb') as f:
            results = pickle.load(f)
    else:
        results = {}

    if(profiling_vs_tod_time):

        key = 'profiling vs tod time'
        results.setdefault(key, {})

        results[key]["t_int"] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 30)

        for precision in ('float64', 'float32'): 

            results[key][precision] = {}

            for map_compression in ('individual.fits','individual.fits.gz'):
                
                results[key][precision][map_compression] = {}    
                results[key][precision][map_compression]['peak memory [MB]'] = []
                results[key][precision][map_compression]['time [s]'] = []
                results[key][precision][map_compression]['output size [MB]'] = []

                for t in results[key]["t_int"]:

                    par = load_par_file(f'params_namap.par')
                    par['cdelt'] = 40/3600, 40/3600
                    par['frequencies'] = (715.0,) #GHz        
                    par['precision'] = precision
                    par['num_frames'] = t * 60 #seconds 
                    par['output_map'] = map_compression
                    par['coadd'] = False

                    #------------------------------------------------------
                    tracemalloc.start()
                    start = time.time()
                    main(par)
                    current, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    end = time.time()
                    timing = end - start
                    #------------------------------------------------------

                    # Store results
                    results[key][precision][map_compression]['peak memory [MB]'].append(peak / 1e6)
                    results[key][precision][map_compression]['time [s]'].append(timing)

                    # Path to your files (adjust if needed)
                    folder = 'fits_and_hdf5/'  # current directory
                    filename = par['output_map']
                    name_before_fits = filename.rsplit('.fits', 1)[0]
                    fits_and_after = filename[filename.find('.fits'):]  
                    pattern = f'{name_before_fits}_*{fits_and_after}'

                    # Get all matching files
                    files = glob.glob(os.path.join(folder, pattern))
                    # Sum their sizes in bytes
                    total_size_bytes = sum(os.path.getsize(f) for f in files)
                    # Optionally, convert to MB
                    total_size_mb = total_size_bytes / (1024**2)
                    results[key][precision][map_compression]['output size [MB]'].append(total_size_mb)
                    print(f"time {t}min| time={timing:.2f}s | peak={peak/1e6:.2f}MB | output={total_size_mb:.2f}MB")

                    # Delete them
                    for f in files:
                        try:
                            os.remove(f)
                        except OSError as e:
                            print(f"Error deleting {f}: {e}")

    if(profiling_vs_nb_bands):

        key = 'profiling vs nb bands'
        results.setdefault(key, {})
        results[key]["nb bands"] = (1,5, 10, 16, 20, 30, 40, 50, 64)

        for precision in ('float64', 'float32'): 

            results[key][precision] = {}
            results[key][precision]['peak memory [MB]'] = []
            results[key][precision]['time [s]'] = []
            results[key][precision]['output size [MB]'] = []

            for nb in results[key]["nb bands"]:

                freq_list = 715.0 + 4.0 * np.arange(nb)

                par = load_par_file(f'params_namap.par')
                par['cdelt'] = 40/3600, 40/3600
                par['frequencies'] = freq_list
                par['precision'] = precision
                par['num_frames'] = 5 * 60 #seconds 
                par['output_map'] = 'individual.fits.gz'
                par['coadd'] = False

                #------------------------------------------------------
                tracemalloc.start()
                start = time.time()
                main(par)
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                end = time.time()
                timing = end - start
                #------------------------------------------------------

                # Store results
                results[key][precision]['peak memory [MB]'].append(peak / 1e6)
                results[key][precision]['time [s]'].append(timing)

                # Path to your files (adjust if needed)
                folder = 'fits_and_hdf5/'  # current directory
                filename = par['output_map']
                name_before_fits = filename.rsplit('.fits', 1)[0]
                fits_and_after = filename[filename.find('.fits'):]  
                pattern = f'{name_before_fits}_*{fits_and_after}'

                # Get all matching files
                files = glob.glob(os.path.join(folder, pattern))
                # Sum their sizes in bytes
                total_size_bytes = sum(os.path.getsize(f) for f in files)
                # Optionally, convert to MB
                total_size_mb = total_size_bytes / (1024**2)
                results[key][precision]['output size [MB]'].append(total_size_mb)
                print(f"Nb bands {nb}| time={timing:.2f}s | peak={peak/1e6:.2f}MB | output={total_size_mb:.2f}MB")

                # Delete them
                for f in files:
                    try:
                        os.remove(f)
                    except OSError as e:
                        print(f"Error deleting {f}: {e}")

    with open(path, 'wb') as f: pickle.dump(results, f)

if __name__ == "__main__":

    # --- Toggle simulation mode ---
    USE_FAKE_SYSTEM = True  # 🔄 Set False to use your real machine

    #I: coadded maps
    I_coadded_maps = False
    #II: individual maps
    II_individual_maps = True
    #III TODs 
    #IV profile functions
    
    if USE_FAKE_SYSTEM:
        print("⚙️  Simulating Intel Celeron 4305UE environment...")
        os.environ["OMP_NUM_THREADS"] = str(SIM_CORES)  # restrict OpenMP threads
        os.environ["OPENBLAS_NUM_THREADS"] = str(SIM_CORES)
        os.environ["MKL_NUM_THREADS"] = str(SIM_CORES)
        os.environ["NUMEXPR_NUM_THREADS"] = str(SIM_CORES)
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(SIM_CORES)

        cpu_mock, mem_mock = simulate_celeron_system()
        with cpu_mock, mem_mock:
            print("Simulated CPU count:", os.cpu_count())
            print("Simulated RAM (GB):", psutil.virtual_memory().total / 1024**3)
            print("Simulated CPU model:", SIM_CPU)
            print("OpenMP threads limited to:", os.environ["OMP_NUM_THREADS"])

            # Place your performance or profiling code here
            if(I_coadded_maps): test_I()
            if(II_individual_maps): test_II()
    else:
        print("💻 Using your real system:")
        print("Real CPU count:", os.cpu_count())
        print("Real RAM (GB):", psutil.virtual_memory().total / 1024**3)
        print("Real CPU model:", platform.processor())
        print("OMP threads (default):", os.environ.get("OMP_NUM_THREADS", "not set"))
