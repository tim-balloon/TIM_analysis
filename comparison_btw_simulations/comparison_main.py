import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.io import fits
from astropy.cosmology import Planck13 as cosmo
from IPython import embed
from functools import partial
from astropy.table import Table
import os
import pickle 
import scipy.constants as cst
from set_k import *
from progress.bar import Bar
import vaex as vx
import time 
from load_params import *
from gen_cats import * 
from Mpc_power_spectra_from_TIM import *

def lf_from_Snudv(cat, Lline, nu_rest, nu_obs, Vslice, Deltabin, 
                  log_bins, min_logLbin_value, max_logLbin_value):
    
    '''
    Function that constructs the luminosity function given:
    - L_line: the line (e.g., CO(J - J-1), [CII], [CI])
    - field_size: the size of the field (survey size)
    - z1, z2: the redshift slice
    - bins: the number of luminosity bins (default = 28)
    - min_bin_value, max_bin_value: the range of the luminosities (default = [1e+5, 1e+12]*units)
    OUTPUT: the luminosity bins (L) and the Φ(L) value for each L
    '''

    logL_inzbin     = np.log10(cat['I'+Lline] * (1.04e-3 * cat['Dlum']**2 * nu_rest / (1 + cat['redshift']))) 
    histo = np.histogram(logL_inzbin, bins = log_bins, range = (min_logLbin_value, max_logLbin_value))
    LF_array= histo[0] / Deltabin / Vslice

    return LF_array

def add_to_dict(d, path, element):
    """
    Add an element to a nested dictionary at a specified path.

    Args:
    - d (dict): The nested dictionary.
    - path (list): List representing the path where the element will be added.
    - element: The element to be added.

    Returns:
    - d
    """
    # Base case: If the path is empty, add the element to the dictionary
    if len(path) == 0:
        d.update(element)
    else:
        # Get the first key in the path
        current_key = path[0]
        # Check if the current key exists in the dictionary
        if current_key not in d:
            # If it doesn't exist, create a new dictionary for it
            d[current_key] = {}
        # Recursively call add_to_dict with the sub-dictionary and the remaining path
        add_to_dict(d[current_key], path[1:], element)
    
    #pickle.dump(d, open(dictname, 'wb'))

    return d

def B_and_sn(cat, line, nu_rest, z, dz, field_size, log_bins):
    
    nu_obs = nu_rest /(1+cat['redshift'])
    dnu=dz*nu_obs/(1+z)
    vdelt = (cst.c * 1e-3) * dnu / nu_obs #km/s
    S = cat['I'+line] / vdelt  #Jy
    B = np.sum(S) / field_size
    shot_noise = np.sum(S**2) / field_size

    S = S.loc[S>0]
    histo, edges = np.histogram(np.log10(S), bins = log_bins, weights=S**2)
    dndS_Ssquare = histo/field_size
    histo, edges = np.histogram(np.log10(S), bins = log_bins, weights=S)
    dndS_S = histo/field_size

    return B.value, dndS_S.value, shot_noise.value, dndS_Ssquare.value

    '''
    x  = logS
    y  = np.log10(dndS_Ssquare)
    sn = np.log10(shot_noise)
    a =  np.log10(dndS_S)
    fig, ax = plt.subplots(figsize=(4,4), dpi=200) 
    w = np.where(~np.isinf(y))[0]
    ax.plot(x[w], y[w], c='b') 
    ax.axhline(sn, -15,5, c='b', ls=':' )
    ax.set_xlabel("ln($\\rm S_{\\nu} $) [mJy]")
    ax.set_ylabel('log(dn/dS $\\rm S^3$) [$\\rm Jy^2$/sr]', color='b')
    ax.tick_params('y', colors='blue')
    ax2 = ax.twinx()
    w = np.where(~np.isinf(a))[0]
    ax2.plot(x[w], a[w],'r' )
    ax2.set_ylabel('log(dn/dS $\\rm S^2$) [$\\rm Jy$/sr]', color='r')
    ax2.axhline(np.log10(B), -15,5, c='r', ls=':' )
    ax2.tick_params('y', colors='r')
    ax.set_title(f'[CII]@z={np.round(z,1)}'+'$\\rm \\pm$'+f'{dz}')
    plt.show()
    '''

def rhoh2(cat, Vslice,dz, alpha_co, toemb=False):

    if(toemb): embed()
    cat = cat.loc[cat['redshift']>0]    

    if(len(cat['ICO10']) > 0 and len(cat['redshift']) > 0):
        nu_obs =  115.27120180 / (1+cat['redshift'])
        L = cat['ICO10'] * (1.04e-3) * nu_obs * cosmo.luminosity_distance(cat['redshift'])**2
        rho_Lprim = np.sum(L) / 3e-11 / 115.27120180**3 / Vslice 
        rhoh2 = rho_Lprim * alpha_co 
        return rhoh2.value
    else: return 0

def make_log_bin(min_logbin_value, max_logbin_value, nbins):
    log_bins = np.linspace(min_logbin_value, max_logbin_value, nbins)
    Deltabin = (max_logbin_value - min_logbin_value) * 1. / nbins #in dex
    log_mean = log_bins[:-1] + Deltabin / 2
    return log_bins, Deltabin, log_mean

def pk_at_z_for_lines(tile_sizeRA, tile_sizeDEC, nsimu, params, z, dz, pathdict = './'):
    
    start = time.time()

    file = pathdict+f'{tile_sizeRA}deg_x_{tile_sizeDEC}deg_z{z}_dz{dz}_pksatzlines.p'
    if( not os.path.isfile(file) ):

        dict = {}

        tile_size = (tile_sizeRA * tile_sizeDEC *u.deg**2).to(u.sr)

        pk_list = np.zeros((len(line_list)+1, 100,  nsimu ))

        bar = Bar(f'Processing P of k for lines of {tile_sizeRA}x{tile_sizeDEC}deg2, z={z}, dz={dz}', max=Nsimu)

        for l in range(nsimu):
            dico = {}

            cat = Table.read(params['sides_cat_path']+f"pySIDES_from_uchuu_tile_{l}_{tile_sizeRA}deg_x_{tile_sizeDEC}deg.fits")
            cat = cat.to_pandas()

            for j, (line, rest_freq) in enumerate(zip(line_list, rest_freq_list)):
                dico[line] = {}
                
                k, pk = make_3d_volume_from_cat(cat, f'{tile_sizeRA}deg_x_{tile_sizeDEC}deg_tile{l}',
                                                line, rest_freq, params,
                                                z_center=z, Delta_z=dz, Nvox=params['Nvox'])
                
                pk_list[j, :pk.shape[0],l] = pk.value
                dico[line]['3d_P of k [Jy2_Mpc3_per_sr2]'] = pk.value
                dico[line]['k [per_Mpc3]'] = k

            dict[f'tile_{l}'] = dico 
            bar.next()

        for j, (line, rest_freq) in enumerate(zip(line_list, rest_freq_list)):
            data = pk_list[j,:,:]
            dict['tile_0'][line][f'P_of_k_mean'] = np.mean(data, axis = (-1))
            dict['tile_0'][line][f'P_of_k_std']  = np.std(data, axis = (-1))
            dict['tile_0'][line][f'P_of_k_median'] = np.median(data, axis = (-1))
            p2, p5, p16, p25, p75, p84, p95, p98 =  np.percentile(data,[2,5,16,25,75,84,95,98], axis = (-1))
            for quantile, value in zip((2,5,16,25,75,84,95,98),(p2, p5, p16, p25, p75, p84, p95, p98)):
                dict['tile_0'][line][f'P_of_k_quantile{quantile}'] = value
        
        bar.finish

        pickle.dump(dict, open(file, 'wb'))
    
    else: dict =  pickle.load( open(file, 'rb'))

    end = time.time()
    timing = end - start
    print('')
    print(f'Processing P of k for lines of {tile_sizeRA}x{tile_sizeDEC}deg2, z={z}, dz={dz} in {np.round(timing,2)} sec!')
    print('')

    return ['pks',f'dz{dz:.2f}',f'z{z:.2f}',f'{tile_sizeRA}x{tile_sizeDEC}deg2'], dict

def sfrd_contribution(cat,log_bins, Vslice):

    SFRD = np.sum(cat['SFR']) / Vslice.value 
    #sfrd_contrib = histo / Vslice / Deltabin
    sfrd_contrib = np.histogram(np.log10(cat['SFR']), bins=log_bins, weights=cat['SFR'])[0]/Vslice.value

    return SFRD, sfrd_contrib

def qtties_vs_z_fix_dz(tile_sizeRA, tile_sizeDEC, nsimu, params,
                        zmax=7.5, dz=0.1, pathdict='./'):


    start = time.time()
    file = pathdict+f'{tile_sizeRA}deg_x_{tile_sizeDEC}deg_zmax{zmax}_dz{dz}_qttiesvsz.p'

    if( not os.path.isfile(file) ):
        
        dict = {}

        log_S_bins, S_Deltabin, log_S_mean = make_log_bin(params['min_logSbin_value'], params['max_logSbin_value'], params['nSbins'])
        log_sfr_bins, sfr_Deltabin, log_sfr_mean  = make_log_bin(params['min_logSFRbin_value'], params['max_logSFRbin_value'], params['nSFRbins'])
        
        zbins = np.arange(0,zmax,dz)
        zmean = 0.5*(zbins[0:-1]+zbins[1:])
        Dc_bins = cosmo.comoving_distance(zbins)
        tile_size = (tile_sizeRA * tile_sizeDEC *u.deg**2).to(u.sr)

        bar = Bar(f'Processing qtties vs z for {tile_sizeRA}x{tile_sizeDEC}deg2 and dz={dz}', max=nsimu)
        SFRD_list = np.zeros((len(zmean), nsimu))
        rho_mol_list = np.zeros((len(zmean), 3, nsimu))
        sfrb_contrib_list = np.zeros((len(zmean), len(log_sfr_mean), nsimu)) ##
        B_list = np.zeros((len(zmean), len(line_list)+1, nsimu))
        SN_list = np.zeros((len(zmean), len(line_list)+1, nsimu))
        b_contrib_list = np.zeros((len(zmean), len(log_S_mean),len(line_list)+1, nsimu))
        sn_contrib_list = np.zeros((len(zmean), len(log_S_mean),len(line_list)+1, nsimu))

        dict['z'] = zmean
        dict['log_S_mean'] = log_S_mean
        dict['log_S_bins'] = log_S_bins
        dict['log_SFR_mean'] = log_sfr_mean
        dict['log_SFR_bins'] = log_sfr_bins
        dict['delta_SFR_dex'] = sfr_Deltabin

        for l in range(nsimu):

            dict[f'tile_{l}'] = {}

            cat = Table.read(params['sides_cat_path']+f"pySIDES_from_uchuu_tile_{l}_{tile_sizeRA}deg_x_{tile_sizeDEC}deg.fits")
            cat = cat.to_pandas()
        
            for i,z in enumerate(zmean): 

                cat_bin = cat.loc[ (cat['redshift'] > zbins[i]) & (cat['redshift'] < zbins[i+1])]
                Vslice = tile_size / 3 * (Dc_bins[i+1]**3-Dc_bins[i]**3)
                sfrd, sfrdcontrib =  sfrd_contribution(cat_bin,log_sfr_bins, Vslice,)

                #simim = np.load('simIM/simim_fiducial.npy', allow_pickle=True).item()
                #number = min(simim['axes']['redshift'], key=lambda x: abs(x - float(z))) #6.057910
                #plt.plot(simim['axes']['sfr_df_logsfr'], simim[f"z={number:.6f}"]['sfr_df'],'-ob',markersize=2, label=simim['model_name'])
                #plt.hlines(simim[f"z={number:.6f}"]['sfr_cosmic_density'], xmin=-6.3, xmax=6, color='b', ls =':')
                #if(z==1.55): embed()

                SFRD_list[i,l] = sfrd
                sfrb_contrib_list[i,:,l] = sfrdcontrib
                ms_cat = cat_bin.loc[cat_bin['ISSB']==0]
                sb_cat = cat_bin.loc[cat_bin['ISSB']==1]
                rho_mol_list[i,0,l] = rhoh2(ms_cat, Vslice, dz, params['alpha_co_ms'])  #solar masses per Mpc cube
                if(len(sb_cat)>0): rho_mol_list[i,1,l] = rhoh2(sb_cat, Vslice, dz, params['alpha_co_sb'])  #solar masses per Mpc cube
                rho_mol_list[i,2,l] = rho_mol_list[i,1,l] + rho_mol_list[i,2,l]
                
                for j, (line, rest_freq) in enumerate(zip(line_list, rest_freq_list)):
                    B, B_contrib, shot_noise, shot_noise_contrib = B_and_sn(cat_bin, line, rest_freq, z, dz, tile_size, log_S_bins)   
                    B_list[i,j,l] = B
                    SN_list[i,j,l] = shot_noise
                    b_contrib_list[i,:,j,l] = B_contrib
                    sn_contrib_list[i,:,j,l] = shot_noise_contrib

            #After allr edshifts are computed, add to dict:
            for j, (line, rest_freq) in enumerate(zip(line_list, rest_freq_list)):
                    dict[f'tile_{l}'][line] = {}
                    dict[f'tile_{l}'][line]['B'] = B_list[:,j,l]
                    dict[f'tile_{l}'][line]['shot noise'] = SN_list[:,j,l]
                    dict[f'tile_{l}'][line]['B contrib'] = b_contrib_list[:,:,j,l]
                    dict[f'tile_{l}'][line]['shot noise contrib'] = sn_contrib_list[:,:,j,l]                
            dict[f'tile_{l}']['SFRD'] = SFRD_list[:,l]
            dict[f'tile_{l}']['SFRD_contrib'] = sfrb_contrib_list[:,:,l] 
            dict[f'tile_{l}']['rhoH2_MS'] = rho_mol_list[:,0,l]
            dict[f'tile_{l}']['rhoH2_SB'] = rho_mol_list[:,1,l]
            dict[f'tile_{l}']['rhoH2_TOT'] = rho_mol_list[:,2,l]
            dict[f'tile_{l}']['B'] = B_list[:,j,l] 
            dict[f'tile_{l}']['SN'] = SN_list[:,j,l] 
            dict[f'tile_{l}']['B_contrib'] = b_contrib_list[:,:,j,l] 
            dict[f'tile_{l}']['SN_contrib'] = sn_contrib_list[:,:,j,l] 

            bar.next()
        #---
        
        for key, data in zip( ('SFRD', 'SFRD_contrib'), (SFRD_list, sfrb_contrib_list) ):
            dict[f'{key}_mean'] = np.mean(data, axis = (-1))          
            dict[f'{key}_std']  = np.std(data, axis = (-1))
            dict[f'{key}_median'] = np.median(data, axis = (-1))
            p2, p5, p16, p25, p75, p84, p95, p98 =  np.percentile(data,[2,5,16,25,75,84,95,98], axis = (-1))
            for quantile, p in zip((2,5,16,25,75,84,95,98),(p2, p5, p16, p25, p75, p84, p95, p98)):
                dict[f'{key}_quantile{quantile}'] = p
    
        for key, ikey in zip(('MS', 'SB', 'TOT'), (0,1,2)):
            dict[f'{key}_mean'] = np.mean(rho_mol_list[:,ikey,:], axis=-1)
            dict[f'{key}_std']  = np.std(rho_mol_list[:,ikey,:], axis=-1)
            dict[f'{key}_median']  = np.median(rho_mol_list[:,ikey,:], axis=-1)
            p2, p5, p16, p25, p75, p84, p95, p98 =  np.percentile(rho_mol_list[:,ikey,:],[2,5,16,25,75,84,95,98], axis = (-1))
            for quantile, p in zip((2,5,16,25,75,84,95,98),(p2, p5, p16, p25, p75, p84, p95, p98)):
                dict[f'{key}_quantile{quantile}'] = p

        for j, (line, rest_freq) in enumerate(zip(line_list, rest_freq_list)):
            dict[line] = {} 
            for key, data in zip(('Background','shot_noise' ), (B_list,SN_list )): #'

                dict[line][f'{key}_mean'] = np.mean(data[:,j,:], axis = (-1))
                dict[line][f'{key}_std']  = np.std(data[:,j,:], axis = (-1))
                dict[line][f'{key}_median'] = np.median(data[:,j,:], axis = (-1))
                p2, p5, p16, p25, p75, p84, p95, p98 =  np.percentile(data[:,j,:],[2,5,16,25,75,84,95,98], axis = (-1))
                for quantile, p in zip((2,5,16,25,75,84,95,98),(p2, p5, p16, p25, p75, p84, p95, p98)):
                    dict[line][f'{key}_quantile{quantile}'] = p

        for key, data in zip(('Background_contrib', 'shot_noise_contrib'), (b_contrib_list, sn_contrib_list)):
            for j, (line, rest_freq) in enumerate(zip(line_list, rest_freq_list)):
                dict[line][f'{key}_mean'] = np.mean(data[:,:,j,:], axis = (-1))
                dict[line][f'{key}_std']  = np.std(data[:,:,j,:], axis = (-1))
                dict[line][f'{key}_median'] = np.median(data[:,:,j,:], axis = (-1))
                p2, p5, p16, p25, p75, p84, p95, p98 =  np.percentile(data[:,:,j,:],[2,5,16,25,75,84,95,98], axis = (-1))
                for quantile, p in zip((2,5,16,25,75,84,95,98),(p2, p5, p16, p25, p75, p84, p95, p98)):
                    dict[line][f'{key}_quantile{quantile}'] = p
        
        bar.finish
        pickle.dump(dict, open(file, 'wb'))
    
    else: dict =  pickle.load( open(file, 'rb'))

    end = time.time()
    timing = end - start
    print('')
    print(f'Processing qtties vs z for {tile_sizeRA}x{tile_sizeDEC}deg2 and dz={dz} in {np.round(timing,2)} sec!')
    print('')

    return ['qtties_vs_z',f'dz{dz:.2f}',f'{tile_sizeRA}x{tile_sizeDEC}deg2'], dict

def qtties_at_z_for_lines(tile_sizeRA, tile_sizeDEC, nsimu, params, z, dz, pathdict='./'):
    
    start = time.time()
    file = pathdict+f'{tile_sizeRA}deg_x_{tile_sizeDEC}deg_z{z}_dz{dz}_qttiesatzlines.p'

    if( not os.path.isfile(file) ):

        log_L_bins, L_Deltabin, log_L_mean = make_log_bin(params['min_logLbin_value'], params['max_logLbin_value'], params['nLbins'])
        log_S_bins, S_Deltabin, log_S_mean = make_log_bin(params['min_logSbin_value'], params['max_logSbin_value'], params['nSbins'])
        
        dict = {}
        dict['log_L'] = log_L_mean
        dict['log_S'] = log_S_mean
        dict['delta_L_dex'] = L_Deltabin
        dict['delta_S_dex'] = S_Deltabin

        tile_size = (tile_sizeRA * tile_sizeDEC *u.deg**2).to(u.sr)

        B_list = np.zeros((len(line_list)+1, nsimu))
        SN_list = np.zeros((len(line_list)+1, nsimu))
        b_contrib_list = np.zeros((len(log_S_mean),len(line_list)+1, nsimu))
        sn_contrib_list = np.zeros((len(log_S_mean),len(line_list)+1, nsimu))
        lf_list = np.zeros((len(log_L_mean),len(line_list)+1, nsimu))

        bar = Bar(f'Processing qtties at z for lines of {tile_sizeRA}x{tile_sizeDEC}deg2, z={z}, dz={dz}', max=Nsimu)

        for l in range(nsimu):

            cat = Table.read(params['sides_cat_path']+f"pySIDES_from_uchuu_tile_{l}_{tile_sizeRA}deg_x_{tile_sizeDEC}deg.fits")
            cat = cat.to_pandas()
            cat_bin = cat.loc[ (cat['redshift'] >= z-dz/2) & (cat['redshift'] < z+dz/2)]
            Vslice = tile_size / 3 * (cosmo.comoving_distance(z+dz/2)**3-cosmo.comoving_distance(z-dz/2)**3)

            dico = {}

            for j, (line, rest_freq) in enumerate(zip(line_list, rest_freq_list)):
                
                dico[line] = {}

                lf_list[:, j,l ] = lf_from_Snudv(cat_bin, line, rest_freq, rest_freq/(1+z), Vslice, L_Deltabin, log_L_bins, params['min_logLbin_value'], params['max_logLbin_value'])
                B, B_contrib, shot_noise, shot_noise_contrib = B_and_sn(cat_bin, line, rest_freq, z, dz, tile_size, log_S_bins)   
                B_list[j,l] = B
                SN_list[j,l] = shot_noise
                b_contrib_list[:, j, l ]  = B_contrib
                sn_contrib_list[:, j, l ] = shot_noise_contrib
            
                dico[line]['LF'] = lf_list[:, j,l]
                dico[line]['B'] = B_list[j, l]
                dico[line]['shot noise'] = SN_list[j, l]
                dico[line]['B contrib'] = b_contrib_list[:, j, l]
                dico[line]['shot noise contrib'] = sn_contrib_list[:, j, l]

            dict[f'tile_{l}'] = dico 
            bar.next()

        for j, (line, rest_freq) in enumerate(zip(line_list, rest_freq_list)):
            dict[line] = {} 
            for key, data in zip(('Background','shot_noise' ), (B_list,SN_list )): 
                dict[line][f'{key}_mean'] = np.mean(data[j,:], axis = (-1))
                dict[line][f'{key}_std']  = np.std(data[j,:], axis = (-1))
                dict[line][f'{key}_median'] = np.median(data[j,:], axis = (-1))
                p2, p5, p16, p25, p75, p84, p95, p98 =  np.percentile(data[j,:],[2,5,16,25,75,84,95,98], axis = (-1))
                for quantile, value in zip((2,5,16,25,75,84,95,98),(p2, p5, p16, p25, p75, p84, p95, p98)):
                    dict[line][f'{key}_quantile{quantile}'] = value
        
        for key, data in zip(('Background_contrib', 'shot_noise_contrib', 'LF'), (b_contrib_list, sn_contrib_list, lf_list)):
            for j, (line, rest_freq) in enumerate(zip(line_list, rest_freq_list)):
                dict[line][f'{key}_mean'] = np.mean(data[:,j,:], axis = (-1))
                dict[line][f'{key}_std']  = np.std(data[:,j,:], axis = (-1))
                dict[line][f'{key}_median'] = np.median(data[:,j,:], axis = (-1))
                p2, p5, p16, p25, p75, p84, p95, p98 =  np.percentile(data[:,j,:],[2,5,16,25,75,84,95,98], axis = (-1))
                for quantile, value in zip((2,5,16,25,75,84,95,98),(p2, p5, p16, p25, p75, p84, p95, p98)):
                    dict[line][f'{key}_quantile{quantile}'] = value

        bar.finish

        pickle.dump(dict, open(file, 'wb'))
    
    else: dict =  pickle.load( open(file, 'rb'))

    end = time.time()
    timing = end - start
    print('')
    print(f'Processing qtties at z for lines of {tile_sizeRA}x{tile_sizeDEC}deg2, z={z}, dz={dz} in {np.round(timing,2)} sec!')
    print('')
    
    return ['qtties_at_z',f'dz{dz:.2f}',f'z{z:.2f}',f'{tile_sizeRA}x{tile_sizeDEC}deg2'], dict

def qtties_at_z(tile_sizeRA, tile_sizeDEC, nsimu, params, z, dz, pathdict='./'):
    
    start = time.time()

    file = pathdict+f'{tile_sizeRA}deg_x_{tile_sizeDEC}deg_z{z}_dz{dz}_qttiesatz.p'

    if( not os.path.isfile(file) ):

        log_sfr_bins, sfr_Deltabin, log_sfr_mean  = make_log_bin(params['min_logSFRbin_value'], params['max_logSFRbin_value'], params['nSFRbins'])
        dict = {}
        dict['log_SFR'] = log_sfr_mean
        dict['delta_SFR_dex'] = sfr_Deltabin

        tile_size = (tile_sizeRA * tile_sizeDEC *u.deg**2).to(u.sr)

        SFRD_list = np.zeros((nsimu))
        rho_mol_list = np.zeros((3, nsimu))
        sfrd_contrib_list = np.zeros((len(log_sfr_mean), nsimu))

        bar = Bar(f'Processing qtties at z for {tile_sizeRA}x{tile_sizeDEC}deg2, z={z}, dz={dz}', max=Nsimu)

        for l in range(nsimu):

            cat = Table.read(params['sides_cat_path']+f"pySIDES_from_uchuu_tile_{l}_{tile_sizeRA}deg_x_{tile_sizeDEC}deg.fits")
            cat = cat.to_pandas()
            cat_bin = cat.loc[ (cat['redshift'] >= z-dz/2) & (cat['redshift'] < z+dz/2)]

            Vslice = tile_size / 3 * (cosmo.comoving_distance(z+dz/2)**3-cosmo.comoving_distance(z-dz/2)**3)

            ms_cat = cat_bin.loc[cat_bin['ISSB']==0]
            sb_cat = cat_bin.loc[cat_bin['ISSB']==1]
            rho_mol_list[0,l] = rhoh2(ms_cat, Vslice, dz, params['alpha_co_ms'])  #solar masses per Mpc cube
            if(len(sb_cat)>0): rho_mol_list[1,l] = rhoh2(sb_cat, Vslice, dz, params['alpha_co_sb'])  #solar masses per Mpc cube
            rho_mol_list[2, l] = rho_mol_list[0, l] + rho_mol_list[1, l]

            sfrd, sfrdcontrib =  sfrd_contribution(cat_bin,log_sfr_bins, Vslice,)
            SFRD_list[l] = sfrd
            sfrd_contrib_list[:, l] = sfrdcontrib
            
            dico = {'SFRD': np.asarray(SFRD_list[l]),
                    'rho_mol_MS': np.asarray(rho_mol_list[0,l]),
                    'rho_mol_SB': np.asarray(rho_mol_list[1,l]),
                    'rho_mol_TOT': np.asarray(rho_mol_list[2,l]),
                    'sfrd_contrib': sfrd_contrib_list[:, l]
                    }
            
            dict[f'tile_{l}'] = dico 

            bar.next()

        #---

        for key, data in zip( ('SFRD','SFRD_contrib'), (SFRD_list,sfrd_contrib_list) ):
            dict['tile_0'][f'{key}_mean'] = np.mean(data, axis = (-1))
            dict['tile_0'][f'{key}_std']  = np.std(data, axis = (-1))
            dict['tile_0'][f'{key}_median'] = np.median(data, axis = (-1))
            p2, p5, p16, p25, p75, p84, p95, p98 =  np.percentile(data,[2,5,16,25,75,84,95,98], axis = (-1))
            for quantile, value in zip((2,5,16,25,75,84,95,98),(p2, p5, p16, p25, p75, p84, p95, p98)):
                dict['tile_0'][f'{key}_quantile{quantile}'] = value

        for i, type in zip((0,1,2),('MS', 'SB', 'TOT')):
            data = rho_mol_list[i,:]
            dict['tile_0'][f'rho_mol_{type}_mean'] = np.mean(data, axis = (-1))
            dict['tile_0'][f'rho_mol_{type}_median'] = np.median(data, axis = (-1))
            dict['tile_0'][f'rho_mol_{type}_std'] = np.std(data, axis = (-1))
            p2, p5, p16, p25, p75, p84, p95, p98 =  np.percentile(data,[2,5,16,25,75,84,95,98], axis = (-1))
            for quantile, p in zip((2,5,16,25,75,84,95,98),(p2, p5, p16, p25, p75, p84, p95, p98)):
                dict['tile_0'][f'rho_mol_{type}_quantile{quantile}'] = p
            dict[f'tile_{l}'] = dico 

        bar.finish

        pickle.dump(dict, open(file, 'wb'))
    
    else: dict =  pickle.load( open(file, 'rb'))

    end = time.time()
    timing = end - start
    print('')
    print(f'Processing qtties at z for {tile_sizeRA}x{tile_sizeDEC}deg2, z={z}, dz={dz} in {np.round(timing,2)} sec!')
    print('')
    
    return ['qtties_at_z',f'dz{dz:.2f}',f'z{z:.2f}',f'{tile_sizeRA}x{tile_sizeDEC}deg2'], dict

if __name__ == "__main__":

    params = load_params('PAR_FILES/cubes.par')
    res = params['pixel_size'] * u.arcsec

    pathdict = 'pySIDES_from_uchuu/'
    dict_name= pathdict+f'dict.p'
    dict_simu={}

    for tile_sizeRA, tile_sizeDEC, Nsimu in params['tile_sizes']:

        for dz in params['dz_list_for_I']:
            #'I) The quantities vs z, with fix redshift step dz'
            path, dict = qtties_vs_z_fix_dz(tile_sizeRA, tile_sizeDEC, Nsimu, params, dz=dz, pathdict = pathdict)
            dict_simu = add_to_dict(dict_simu, path, dict)

        for z,dz in zip(params['z_list_for_II'],params['dz_list_for_II']):
            #II) The quantities rather computed at a fix redshift.
            path, dict = qtties_at_z_for_lines(tile_sizeRA, tile_sizeDEC, Nsimu, params, z, dz, pathdict = pathdict)
            dict_simu = add_to_dict(dict_simu, path, dict)

            path, dict = qtties_at_z(tile_sizeRA, tile_sizeDEC, Nsimu, params, z, dz, pathdict = pathdict)
            dict_simu = add_to_dict(dict_simu, path, dict)

            path, dict = pk_at_z_for_lines(tile_sizeRA, tile_sizeDEC, Nsimu, params, z, dz, pathdict = pathdict)
            dict_simu = add_to_dict(dict_simu, path, dict)

    print(f'save {dict_name}')
    pickle.dump(dict_simu, open(dict_name, 'wb'))
    d = pickle.load( open(dict_name, 'rb'))

