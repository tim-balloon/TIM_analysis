import random
from scan_fcts import pixelOffset
from strategy import load_params
import argparse
import numpy as np

random.seed(42)   # Fixes the seed for Python's random module
np.random.seed(42)  # Fixes the seed for NumPy

def generate_strings(n=64):
    strings = set()
    while len(strings) < n:
        rand_digits = f"{random.randint(0, 999):03d}"
        strings.add(f"A_{rand_digits}")
    return sorted(strings)

if __name__ == "__main__":

    random.seed(42)   # Fixes the seed for Python's random module
    np.random.seed(42)  # Fixes the seed for NumPy

    #Iinitialization
    parser = argparse.ArgumentParser(description='Gen a random file of names for TIM kids', \
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    #options
    parser.add_argument('params', help=".par file with params", default = None)
    args = parser.parse_args()

    P = load_params(args.params)
    pixel_offset_HW = pixelOffset(P['nb_pixel_HW'], P['offset_HW']) 
    pixel_offset_LW = pixelOffset(P['nb_pixel_LW'], P['offset_LW']) 

    det_name_HF = generate_strings()
    pixel_offset = pixel_offset_HW
    resp = np.ones(len(det_name_HF))
    noise = np.zeros(len(det_name_HF))
    time_offset = np.zeros(len(det_name_HF)) #Time delay between pointing solution and detector data
    EL = np.zeros(len(det_name_HF))
    XEL = pixel_offset_HW

    # Ensure pixel_offset_HW has the same length as det_names
    if len(pixel_offset_HW) < len(det_name_HF) or len(resp) < len(det_name_HF) or len(resp) < len(noise) or  len(resp) < len(time_offset):
        raise ValueError("Not enough pixel offsets for detector names.")

    # Save as a TSV file
    file = 'config/TIM_kid_table.tsv'
    with open(file, 'w') as f:
        f.write("Name\tResp.\tWhiteNoise\ttime offset\tEL\tXEL\n")  # Column headers
        for name,  r,n,t, e, xe in zip(det_name_HF, resp, noise,time_offset, EL, XEL):
            f.write(f"{name}\t{r:4f}\t{n:4f}\t{t:4f}\t{e:4f}\t{xe:4f}\n")  # Tab-separated values

    import pandas as pd
    df = pd.read_csv(file, sep='\t')
    print(df.head())  # Show first 5 rows
