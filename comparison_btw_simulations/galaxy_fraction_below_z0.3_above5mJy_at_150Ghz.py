import sys
import os
from pysides.make_cube import *
from pysides.load_params import *
import argparse
import time
import matplotlib
import numpy as np
from matplotlib.colors import LogNorm
import matplotlib.patches as patches
import re
import glob
from progress.bar import Bar
from comparison_main import make_log_bin
import matplotlib.pyplot as plt

def differential_number_count(cat, channels, params_sides, tile_size):

    log_S_bins, S_Deltabin, log_S_mean = make_log_bin(-6, -2, 20)
    
    lambda_list =  ( cst.c * (u.m/u.s)  / (np.asarray(channels)* u.Hz)  ).to(u.um)
    SED_dict = pickle.load(open(params_sides['SED_file'], "rb"))    
    Snu_arr = gen_Snu_arr(lambda_list.value, 
                        SED_dict, cat["redshift"],
                        cat["LIR"]*cat['mu'], cat["Umean"], 
                        cat["Dlum"], cat["issb"])
    
    # Compute histogram
    fig, (axd, axc) = plt.subplots(1,2, figsize=(10,5), dpi=100)
    for f, (freq,c) in enumerate(zip(channels, ('r','g','b'))):

        S = 10**(log_S_mean-3)
        dN_dS = np.histogram(np.log10(Snu_arr[:,f].value), bins=log_S_bins)[0]/tile_size
        # Plot cumulative histogram
        n, bins, patches = axd.hist(np.log10(Snu_arr[:,f].value), bins=log_S_bins)
        axd.step(S, dN_dS, where='post', label='$\\rm \\nu$='+f'{freq/1e9:.1f}GHz',c=c)

        #dS = np.diff(S_reversed)
        #N_gt_S = np.zeros_like(S)
        N_gt_S = np.cumsum(dN_dS[::-1])
        #N_gt_S[1:] = np.cumsum(0.5 * (dN_dS[:-1] + dN_dS[1:]) )#* dS)
        # Reverse result back to match original S order
        #N_gt_S = N_gt_S[::-1]

        axc.step(S, np.cumsum(dN_dS[::-1])[::-1], where='post', label='$\\rm \\nu$='+f'{freq/1e9:.1f}GHz',c=c)
        
        
    axd.set_xlabel('Flux density $\\rm S_{\\nu}$ [mJy]')
    axc.set_xlabel('Flux density $\\rm S_{\\nu}$ [mJy]')
    axc.set_ylabel('N($\\rm > S_{\\nu}$) [$\\rm deg^{-2}$]')
    axd.set_ylabel('dN/dS [$\\rm Jy^{-1} deg^{-2}$]')
    axc.set_ylim(1e0, 1e6)
    axd.set_ylim(1e0, 1e6)
    axd.set_yscale('log')
    axd.set_xscale('log')
    axc.set_yscale('log')
    axc.set_xscale('log')

    axd.legend()
    fig.tight_layout()
    plt.show()
if __name__ == "__main__":

    params_sides = load_params('PAR_FILES/SIDES_from_original_with_fir_lines.par')

    cat = Table.read('pySIDES_from_uchuu_tile_0_0.2deg_x_1deg.fits')
    cat = cat.to_pandas()
    tile_size = 0.2 * 1

    differential_number_count(cat, (90e9,150e9,220e9), params_sides, tile_size)


