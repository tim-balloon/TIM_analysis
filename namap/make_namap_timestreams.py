from strategy import *

def add_polynome_to_timestream(timestream, time, percent_slope=30):

    #Add a slope
    delta = percent_slope/100 * (np.max(timestream) - np.min(timestream))  # 30% of the data range
    slope = delta / (time[-1] - time[0])
    return slope * (time - time[0]) 

"""
def add_polynome_to_timestream(timestream, time, percent_slope=30):
    # Compute the total amplitude delta (30% of data range)
    delta = percent_slope / 100 * (np.max(timestream) - np.min(timestream))
    
    # Normalize time to [0, 1] for numerical stability
    t_norm = (time - time[0]) / (time[-1] - time[0])
    
    # Construct a 3rd-order polynomial with zero mean and scaled amplitude
    # Example: p(t) = a*t^3 + b*t^2 + c*t + d, but we'll use a simple shape
    # Here we use a centered cubic curve like: (t - 0.5)^3, scaled
    poly_shape = (t_norm - 0.5)**3
    
    # Scale the polynomial to match the target amplitude (delta)
    poly_shape -= np.mean(poly_shape)  # Ensure zero mean
    poly_shape *= delta / (np.max(poly_shape) - np.min(poly_shape))
    
    return poly_shape
"""
'''
from scipy.interpolate import interp1d

# Step 1: create interpolation function from data1
interp_func = interp1d(t1, data1, kind='linear', fill_value='extrapolate')

# Step 2: evaluate at t2 points
resampled_data1 = interp_func(t2)
'''

def add_peaks_to_timestream(timestream, nb_peaks=3, sigma_peak=7, spike_width=3):
      
    sigma = np.std(timestream)
    peak_indices = np.random.choice(len(timestream), size=nb_peaks, replace=False)

    for center in peak_indices:
        # Define the range of the spike
        start = max(0, center - spike_width // 2)
        end = min(len(timestream), center + spike_width // 2 + 1)
        x = np.arange(start, end)

        # Gaussian centered at `center`
        gaussian_spike = sigma_peak * sigma * np.exp(-0.5 * ((x - center) / (spike_width / 4))**2)

        # Add the spike to the timestream
        timestream[x] += gaussian_spike

    return timestream


if __name__ == "__main__":

    '''
    Generate noise TOD. 
    The noise TOD are frequency, detector, and time independant.
    The noise TODs are gaussian. 
    The noise TODs for pixels seeing the same beam (at different frequency bands) are correlated. 
    '''
    #------------------------------------------------------------------------------------------
    #load the .par file parameters
    print('load the parameters')
    parser = argparse.ArgumentParser(description="strategy parameters",
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    #options
    parser.add_argument('params', help=".par file with params", default = None)
    args = parser.parse_args()
    P = load_params(args.params)
    #------------------------------------------------------------------------------------------

    #Load the scan duration and generate the time coordinates with the desired acquisition rate. 
    T_duration = P['T_duration'] 
    dt = P['dt']*np.pi/3.14 #Make the timestep non rational to avoid some stripes in the hitmap. 
    T = np.arange(0,T_duration,dt) * 3600

    tod_file= P['path'] +'TOD_'+P['file'][:-5]+'.hdf5'# P['path']


    det_names_dict = pd.read_csv(P['detectors_name_file'], sep='\t')
    #Each pixel with the same offset sees the same beam, but in different frequency band. 
    same_offset_groups = det_names_dict.groupby(['Frequency'])['Name'].apply(list).reset_index()

    #For each group of pixels seeing the same beam: 
    for group in range(len(same_offset_groups)):

        print(f'starting group {group}')

        start = time.time()
        H = h5py.File(tod_file, "a")    
        for j, name in  enumerate(same_offset_groups.iloc[group]['Name']):
            namegrp = f'kid_{name}_roach'
            f = H[namegrp]
            data = f['data'][()]
            data_with_slope = data + add_polynome_to_timestream(data, T)
            data_with_peak = add_peaks_to_timestream(data_with_slope)
            embed()
            if('namap_data' in f): del f['namap_data'] 
            f.create_dataset('namap_data', data=data_with_peak,
                                compression='gzip', compression_opts=9)
            
        H.close()
        
        end = time.time()
        timing = end - start
        
    print(f'Generate the TODs of group {group} in {np.round(timing,2)} sec!')
        

