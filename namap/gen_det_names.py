import random
from scan_fcts import pixelOffset
from strategy import load_params
import argparse
import numpy as np
import os
from IPython import embed

random.seed(42)   # Fixes the seed for Python's random module
np.random.seed(42)  # Fixes the seed for NumPy

def generate_strings(total_count, groups):
    """
    Generate unique detector names in the format A_XX_NNNN.
    
    Parameters
    ----------
    total_count:int
        Total number of detector names to generate.  
    group: list
        the index refering to each sub-array.   
    Returns
    -------
    list: list
        List of unique detector names.
    """
    n_per_group = total_count // 4  # Number of names per XX group (1008 each)

    all_strings = []

    for group in groups:
        strings = set()
        while len(strings) < n_per_group:
            rand_digits = f"{random.randint(0, 9999):04d}"  # Generate 4-digit random number
            strings.add(f"A_{group}_{rand_digits}")  # Format as A_XX_NNNN
        
        all_strings.extend(sorted(strings))  # Sort for consistency

    return all_strings

if __name__ == "__main__":
    '''
    Do python gen_det_names.py params_strategy.par
    before running strategy.py or namap_main.py
    Generate a .cvs file containing the info of the detectors. 
    '''

    random.seed(42)   # Fixes the seed for Python's random module
    np.random.seed(42)  # Fixes the seed for NumPy

    #---------------------------
    #Iinitialization
    parser = argparse.ArgumentParser(description='Gen a random file of names for TIM kids', \
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    #options
    parser.add_argument('params', help=".par file with params", default = None)
    args = parser.parse_args()

    P = load_params(args.params)
    #---------------------------

    #---------------------------    
    #Generate the offset of the pixels with respect to the center of the two arrays, in degrees. 
    pixel_offset_HW, pixel_shift_HW = pixelOffset(P['nb_pixel_HW'], P['offset_HW'], -P['arrays_separation']/2)
    pixel_offset_LW, pixel_shift_LW = pixelOffset(P['nb_pixel_LW'], P['offset_LW'], P['arrays_separation']/2) 

    pixel_offset_HW_tot = np.tile(pixel_offset_HW, P['nb_channels_per_array'])
    pixel_offset_LW_tot = np.tile(pixel_offset_LW, P['nb_channels_per_array'])
    pixel_offset = np.concatenate((pixel_offset_HW_tot, pixel_offset_LW_tot))

    pixel_shift_HW_tot = np.tile(pixel_shift_HW, P['nb_channels_per_array'])
    pixel_shift_LW_tot = np.tile(pixel_shift_LW, P['nb_channels_per_array'])
    pixel_shift = np.concatenate((pixel_shift_HW_tot, pixel_shift_LW_tot))
    #---------------------------    

    #---------------------------
    # Generate detector names
    det_names_HW = generate_strings(P['nb_pixel_HW'] * P['nb_channels_per_array'], ['01', '02', '03', '04'])
    det_names_LW = generate_strings(P['nb_pixel_LW'] * P['nb_channels_per_array'], ['05', '06', '07', '08'])
    det_names = np.concatenate((det_names_HW,det_names_LW))
    #---------------------------

    #---------------------------    
    #The response of the detectors
    resp = np.ones(len(det_names))
    #The noise of the detectors
    noise = np.zeros(len(det_names))
    #Time delay between pointing solution and detector data
    time_offset = np.zeros(len(det_names)) 
    #Offsets
    EL = pixel_offset
    XEL = pixel_shift
    #EM Frequency band
    freqs = np.zeros(len(det_names))
    #---------------------------

    # Ensure pixel_offset_HW has the same length as det_names
    if len(resp) < len(det_names) or len(noise) < len(det_names) or  len(time_offset) < len(det_names):
        raise ValueError("Not enough pixel offsets for detector names.")

    #---------------------------
    # Save as a TSV file
    # Define the output file path
    file = P['detectors_name_file']    
    with open(file, 'w') as f:
        f.write("Name\tResp.\tWhiteNoise\ttime offset\tEL\tXEL\tFrequency\n")  # Column headers
        for name,  r,n,t, e, xe,F in zip(det_names, resp, noise,time_offset, EL, XEL, freqs):
            f.write(f"{name}\t{r:1f}\t{n:1f}\t{t:1f}\t{e:1f}\t{xe:1f}\t{F:1f}\n")  # Tab-separated values
    #---------------------------

    #---------------------------
    import pandas as pd
    df = pd.read_csv(os.getcwd()+'/'+file, sep='\t')
    print(df.head())  # Show first 5 rows
    #---------------------------
