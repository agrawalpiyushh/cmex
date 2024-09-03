import numpy as np
from copy import deepcopy as copy
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
from subprocess import run as spawn 
import subprocess
import pickle
import shutil # to remove an non-empty directory; # helps with moving file from 1 folder to another
# from numba import jit, njit, prange # to speed up for loops in python
# from PyAstronomy.pyasl import sunpos # get sun position- ra, dec, elong
# from astropy.io import fits
import glob
from datetime import datetime, timedelta # helps find date differnce 
# from scipy.optimize import curve_fit 
# from scipy.signal import find_peaks
from itertools import islice # used to read a text file line by line
import time

# ------------------------------------
'''
#   list of ideas tried and un-tried
    - _hierarchical_way_ - to find the best chisqr - to weigh in IQUV too

        - increase number of wavelength points across the profile
        - test inversion without Mg II h line
        - optical depth and anisotropy reduction factor - how are they related..!!??
        - read - casini 2005 paper - histogram


#    list of codes
    
        - May 2024
            - pCMEx_may3_mask_coeff_diff_gt_threshold

        - Jan 2024
            - pCMEx_PCA_inv_1_profile_calc_chisqr_err_arr
            - pCMEx_get_IQUV_sqr_wts_for_chisqr_calc
            - pCMEx_reconstruct_IQUV_from_coeff
            - pCMEx_decomp_IQUV_to_PCA_coeff
            - pCMEx_RCasini_model_to_dict
            - pCMEx_RCasini_model_1Darr_to_2Darr
            - pCMEx_RCasini_model_2Darr_to_1Darr

            - pCMEx_calc_PCA_eigen_profiles - from a set of stokes profiles - 

        - Dec 29/30 2023
            - pCMEx_read_RCasini_spectralprofiles_file: to read Roberto generated synthetic profiles

        - pCMEx_Gauss1D_1peak - Single Gauss function
        - pCMEx_Gauss1D_2peak - function that is sum of 2 1D Gaussians
        - pCMEx_Gauss1D_3peak - function that is sum of 3 1D Gaussians
        - pCMEx_extract_IRIS_MgIIk_SIdata

        - write_array_to_txt_file

        - incomplete
            - pCMEx_remove_duplicate_Stokes_profiles

        ----------------------------------------------------------
        DISCARDED - 
            - pCMEx_231213_create_multiple_Gauss
                : code to create function with multiple Gaussians
            - pCMEx_Gauss1D_1peak
                : code to create 1D GAussian with a single peak



    Tasks/storyline
    - [Dec 23 2023]: code: "Dec23_Iris_MgIIk_Gaussfit_stats.ipynb"
        - coded up this to get multi-gauss fit
        - for Rebecca - the reason for bimodal distribution of 1 Gauss - Gsigma - to cut off spicule region and see what the Gsigma of the remaining linewidths correspond to.

        - Detour - to study about scattering polarization
            - code up PCA routine
            - 

    Ask Roberto
        - wavelengths are common for all lines in his spectral profiles
        - how was CMEx data created? - what PSF and stuff he used + what models he sampled? How did he decide on them?


    PA Concerns:
        - [Jan 2024]
            - when creating the database - does it matter if 
                - subtract mean 1st and the normalize
                - normalize 1st and then subtract the mean

            - why normalizing IQUV (all) profiles by Imax makes sense?
            - how is data created that is in accordance with what CMEx will observe?
            - 


'''


# ----------------------------------------------------------------------------
# Code Status: ©reen
@njit()
def pCMEx_may3_rCnl_mask_coeff_if_diff_gt_NL(C_IQUV0=None, C_IQUV=None, 
                            # rCnl_diff_thresh_percent=10., 
                            rCnl_diff_thresh_percent_arr=np.zeros(4) + 10., 
                            rCnl_mask_value=np.nan

                            ):
    '''
    written on May 3 2024
    Goal - to compare non-noisy PCA coefficient arr C_IQUV0 with noisy C_IQUV and 
            where the relative difference is greater than rCnl_diff_thresh_percent, set those
            locations to rCnl_mask_value. 

            rCnl - stands for: restrict Coeffs with change (above some threshold) due to noise level

    Input:
        - C_IQUV0 = (nmodes, 4) = non-noisy coefficient set for a given profile
        - C_IQUV =  (nmodes, 4) = noised coefficient set for the same profile

        - rCnl_diff_thresh_percent - if absolute relative difference in coefficient (for a given mode, IQUV) is
            greter than this value, then set those to rCnl_mask_value. 

    Output:
        C_IQUV - with values set to NaN, 
        mask_IQUV = same shape as C_IQUV, but with values 1 and NaN.

    '''
    if np.shape(C_IQUV0) != np.shape(C_IQUV):
        raise NameError('PAerr: C_IQUV0 and C_IQUV dont have same shape')
    if (C_IQUV0 ==0).any(): # as C_IQUV0 comes in the denominator
        raise NameError()

    # ---------------------------
    # place holders
    NL_C_mask_IQUV = np.ones(C_IQUV0.shape) +0.
    C_IQUV1 = C_IQUV+0. 

    # ---------------------------
    # calculate relative change in each coeff due to noise
    abs_percent_diff_coeff = np.abs(1 - C_IQUV/C_IQUV0)*100.

    # print(abs_percent_diff_coeff)

    # ---------------------------
    # loop over IQUV and find/set locations with change > threshold due to noise, to rCnl_mask_value
    for k in range(4):
        ix = np.argwhere(abs_percent_diff_coeff[:,k] > rCnl_diff_thresh_percent_arr[k])[:,0]
        C_IQUV1[ix,k] = rCnl_mask_value
        NL_C_mask_IQUV[ix,k] = rCnl_mask_value

        

    return C_IQUV1, NL_C_mask_IQUV

# ----------------------------------------------------------------------------
# Code status: Green
@njit()
def pCMEx_get_PCAcoeff_barcode_for_PCAdbase(PCAdbase_C_IQUV=None, max_nmodes = 3):
    '''
    Written on Apr 1 2024
    Goal: Get an array of index to be used to accelerate database search

    Input:
        PCAdbase_C_IQUV = entire PCAdbase Coeff IQUV, shape=(nmodes, 4, nmodels)
        max_nmodes = max number of modes to use to calculate barcode.


    Output:
        Decimal barcode (binary converted to decimal) for each model in the PCAdbase
    '''

    nmodels = PCAdbase_C_IQUV.shape[2]
    PCAdbase_barcodes = np.zeros((nmodels, max_nmodes)) + np.nan

    # loop over each model and 
    for k in range(nmodels):
        PCAdbase_barcodes[k,:], void = pCMEx_get_PCAcoeff_barcode_index(PCA_C_IQUV_2Darr=PCAdbase_C_IQUV[:,:,k], 
                                                      max_nmodes=max_nmodes)
    return PCAdbase_barcodes
# --------------------------------------
# Code status: Green - works correctly
@njit()
def pCMEx_get_PCAcoeff_barcode_index(PCA_C_IQUV_2Darr=None, max_nmodes=1):
    '''
    written on Apr 1 2024
    Goal: to convert PCA_C_IQUV 2Darra in binary 01 (according to +ve 1 and -ve 0) values
        - see RCasini 2013 paper: https://ui.adsabs.harvard.edu/abs/2013ApJ...773..180C/abstract
        
    Input:
        - PCA_C_IQUV_2Darr: each row is PCA coeff for IQUV for a given mode
        - max_nmodes = max number of modes for which binary is to be returned
        
    Output
        - decimal-value of the binary barcode
    
    '''    
    if max_nmodes > PCA_C_IQUV_2Darr.shape[0]:
        raise NameError('max_nmodes > PCA_C_IQUV_2Darr.shape[0].')
    elif max_nmodes <= 0:
        raise NameError('max_nmodes <= 0.')
        
    # --------------------------
    # place holders    
    decimal_all = np.zeros(max_nmodes) # for each mode 
    # --------------------------
    # loop over each mode and calc index
    tstr_binary = ''
    for j in range(max_nmodes):
        tstr1 = ''
        for i in range(4): # loop over IQUV
            if PCA_C_IQUV_2Darr[j,i] >0:
                tstr1 += '1'
            else:
                tstr1 += '0'
            
        tstr_binary +=  tstr1
        decimal_all[j] = pCMEx_convert_binary_to_decimal(str_binary=tstr_binary)

    return decimal_all, tstr_binary
# --------------------------
# Code status: Green - works correctly
@njit()
def pCMEx_convert_binary_to_decimal(str_binary=None):
    '''
    written on Apr 1 2024
    Goal to convert a binary string to decimal

    Input
        - str_binary = string containing 0s an 1s only

    Ouput
        - corresponding Decimal number (in floats)

    '''
    
    decimal_numb = 0. # place holder
    
    k=0
    n=len(str_binary)
    for i in range(n-1, -1, -1):      
        strk = str_binary[i]
        # Numba doesn't allow string to integer conversions
        # so using the following workaround
        if strk == '0':
            k1 = 0
        elif strk == '1':
            k1 = 1
        else:
            raise NameError('Numbers other than 0 and 1 in str_binary.')
        decimal_numb += k1 * 2**k        
        k+=1
        
    return decimal_numb
# ----------------------------------------------------------------------------
def INCOMPLETE_pCMEx_calc_projected_height(h=None, theta=None):
    '''
    written on Apr 3 
    '''
    print(1)

# ----------------------------------------------------------------------------
# Code status: 
#@njit()
def pCMEx_find_PCAdbase_profile_indx_below_an_upplim(PCA_dbase_coeff_all=None, 
                                              IQUV_noise_or_not=None, 
                                              noise_ul_arr=None, 
                                              
                                              ):
    '''
    written on Mar 15 2024
    Goal: for IQUV identified by IQUV_noise_or_not, given a noise level uppler limit 
            (IQUV_noise_ul), find the ones in the database, 
            that are below noise level


    - Input:
        - IQUV_noise_or_not =
            - output from pCMEx_get_IQUV_sqr_wts_for_chisqr_calc
            - 1st row: all 0 and 1s; 0 implies obsS is noise dominated 
            - for rest, see: pCMEx_get_IQUV_sqr_wts_for_chisqr_calc

    - Output


    '''
    PCA_dbase_C_IQUV_L2n = np.linalg.norm(PCA_dbase_coeff_all, ord=2, axis=0) # L2norm of C_IQUV for each profile 
    indx_arr_all = np.ones(PCA_dbase_C_IQUV_L2n.shape[1]) # ~ 500k

    k1 = 0
    for k in range(4):
        if IQUV_noise_or_not[k] ==0: # this = 0 means obsStokes was noisy
            noise_ul = noise_ul_arr[k]

            tindx_arr = np.argwhere(PCA_dbase_C_IQUV_L2n[k,:] > noise_ul)[:,0] # find profile with signal above this level
            if tindx_arr.size > 0:
                k1+=1
                indx_arr_all[tindx_arr] = np.nan # set those profile index loc to NaN 

    # ix = np.argwhere(indx_arr_all !=k1)[:,0]
    # indx_arr_all[ix] = np.nan

    return indx_arr_all, PCA_dbase_C_IQUV_L2n

# --------------------------------------
# Code status: makes sense 
@njit()
def pCMEx_check_if_IQUV_above_noise_level(IQUV=None, noise_level_wrt_Imax=None, 
                PCA_eigenbasis_IQUV=None,                                 
                key_mar5_est_IQUVnoise_with_all_wvlnths=0,
                nmodes=-1,
                # PCA_nonzero_threshold=5e-3 
                ):
    '''
    written on Mar 5 2024
    Goal: to determine if any of IQUV vectors are below the noise level - so that we can discard their contribution from chiqr calculation.
        - work is based on including stokes vector that are all mostly noise, lead to bad PCAinversion.

    Caution: noise_level_wrt_Imax and IQUV should be on similar level 
             i.e. this code won't work if IQUV are in flux units, while noise_level_wrt_Imax is calculated wrt Imax
    
    Input - 
        - IQUV = profile

        - key_mar5_est_IQUVnoise_with_all_wvlnths = 
            - 0 for line core wavelengths only (determined from PCA_nonzero_indx_all)
            - 1 for entire wavelengths

        - noise_level_wrt_Imax - noise level in the profiles, wrt Imax
        - PCA_eigenbasis_IQUV - eigen basis, shape: (4, nlambda, nmodes)



    Output: 3by4 element array with 0s and 1s. 
        # 1st row states: 0 if IQUV < noise: 0; 1 if IQUV > noise
        # 2nd row: L2norm of entire profile 
        # 3rd row: L2norm of linecore region only (based on PCA_nonzero_indx_all)
       
    '''
    # --------------------------
    # Get location near line core 
    PCA_nonzero_indx_all = pCMEx_feb27_get_PCAebasis_nonzero_amplitude_indx(PCA_eigenbasis_IQUV=PCA_eigenbasis_IQUV, 
                            nmodes=nmodes, 
                            # threshold=PCA_nonzero_threshold, 
                            )

    # --------------------------
    # place holder, 
    # 1st row states if IQUV > noise or not 
    # 2nd row: noise upper limit in stokes based on sqrt(Nλ) * noise_level_wrt_Imax
    # 3rd row: avg signal (without noise) content |I_nonoise|**2 ~ (|I_noisy|**2 - |err|**2)
    #        : not using this anymore as √(|I_noisy|**2 - |err|**2) could be NaN, if |err| > |I_noisy|
    # 4nd row: L2norm of entire profile  
    # 5rd row: L2norm of linecore region only (based on PCA_nonzero_indx_all)
    Output_arr = np.zeros((4,4))+ np.nan 


    # --------------------------
    # IQUV loop
    for k in range(4):
        # --------------------------
        # get PCAebasis indx - where ebasis is non-zero
        tindx_arr = PCA_nonzero_indx_all[:,k]+0.
        tindx_arr = tindx_arr[~np.isnan(tindx_arr)].astype('int')
        # --------------------------
        # get stokes profile all, and those for which PCAebasis is non-zero (line core region)
        tprofile_all = IQUV[:,k] + 0.
        tprofile_linecore = tprofile_all[tindx_arr] + 0.

        L2n_tprofile_all = np.linalg.norm(tprofile_all, ord=2) # L2 norm
        L2n_tprofile_linecore = np.linalg.norm(tprofile_linecore, ord=2)      

        Output_arr[2,k] = L2n_tprofile_all # L2norm of entire profile 
        Output_arr[3,k] = L2n_tprofile_linecore # L2norm of linecore region only   

        if key_mar5_est_IQUVnoise_with_all_wvlnths ==1:
            nlambda = tprofile_all.size
            tL2n = L2n_tprofile_all
        else:
            nlambda = tprofile_linecore.size
            tL2n = L2n_tprofile_linecore
            
        tnoise_ul = noise_level_wrt_Imax * np.sqrt(nlambda) # estimated noise content in Stokes profile
        Output_arr[1,k] = tnoise_ul+0. # upper limit to signal ~ noise level (when signal dominated by noise)
        # Output_arr[2,k] = np.sqrt(tL2n**2-tnoise_ul**2) # avg signal content |I_nonoise|**2 ~ (|I_noisy|**2 - |err|**2)

        tnoise_ul1 = 2*tnoise_ul # upplim to tL2n**2 = (signal+noise)**2 ~ 2*noise_level**2, 
                                 # for signal to be dominated by noise
                                 # but for safety, taking it to be (2*noise_level)**2, as the threshold

        # --------------------------
        if tL2n > tnoise_ul1: # net L@n should be greater than twice the noies level
            Output_arr[0,k] = 1
        else:
            Output_arr[0,k] = 0

        # print(tL2n, tnoise_ul1, nlambda)


    if np.isnan(Output_arr).any():
        print(Output_arr)
        raise NameError('PAerr: NaNs in Output_arr. This should not have happened.')

    return Output_arr, PCA_nonzero_indx_all

# --------------------------------------
# code statuss: green - works correct 
@njit()
def pCMEx_feb27_get_PCAebasis_nonzero_amplitude_indx(PCA_eigenbasis_IQUV=None, 
                                        threshold=5e-3, nmodes=-1, ):
    '''
    written on Feb 27 2024
    Goal: to find the wavelenght indices in IQUV - for which the PCA eigenbasis has amplitude above the given threshold

    Note:
        - if threshold ~ 1e-4, then there are super small amplitudes in PCA_Q eigen basis for the 10th mode. So, potentially all the wavelengths get included at this threshold.

    Return: (n, 4) array of non-zero PCAebasis amplitude index locations (in float64)


    '''
    nlambda = PCA_eigenbasis_IQUV.shape[1]
    # ------------------------------------
    # get nmodes from which the line core region is to be identified
    max_nmodes = PCA_eigenbasis_IQUV.shape[2]
    if nmodes > max_nmodes:
        raise NameError('PAerr: nmodes > max_nmodes in PCA_eigenbasis_IQUV.')
    elif nmodes ==-1:
        nmodes = max_nmodes # PCA_eigenbasis_IQUV.shape[2]

    # ------------------------------------
    non_zero_indx_all = np.zeros((int(nlambda), 4)) + np.nan # placeholder
    tindx_all0 = np.zeros(int(nmodes*nlambda)) + np.nan # placeholder
    max_indx0_size = 0 

    # ------------------------------------
    # loop over IQUV and for each store the non-zero PCA indx (near line core region)
    for k1 in range(4): # IQUV loop
        tindx_all = tindx_all0+0. # reset place holder to all NaN
        tix0 = 0 
        for k2 in range(nmodes):
            tPCA = np.abs(PCA_eigenbasis_IQUV[k1,:,k2])
            tix = np.argwhere(tPCA >= threshold)[:,0] # get non-zero amplitude indx 
            tindx_all[tix0:tix0+tix.size] = tix # store this tix indx into tindx_all
            tix0 += tix.size

        # ----------------
        # get the uniq tindx and store them in  non_zero_indx_all
        tindx_all_uniq = np.unique(tindx_all[:tix0])
        non_zero_indx_all[:tindx_all_uniq.size, k1] = np.sort(tindx_all_uniq)

        # ----------------
        if tindx_all_uniq.size > max_indx0_size:
            max_indx0_size = tindx_all_uniq.size

    # ----------------
    # remove excess all NaN values from non_zero_indx_all
    non_zero_indx_all = non_zero_indx_all[:max_indx0_size, :] # might still have nans in them

    return non_zero_indx_all

# --------------------------------------
# code - annotate + write doc file. place a few check points 
#       e.g. chisqr_err_arr_IQUV0.shape[0] > initial_n0 and accordingly fact**3
@njit()
def pCMEx_get_min_chisqr_fit_with_hierarchy(chisqr_err_arr_IQUV0=None, 
                                            obs_IQUV=None,
                                            initial_n0=200, fact=4., 
                                            i_bestmodel=None, 
                                            ):
    '''

    Questions:
        - To take (IQUV/Imax - IQUV_mean) or IQUV (in actual units) for obs_IQUV

    '''
    
    # -------------------------------
    # get order in which to perform finding best chisqr
    # in decreasing order of importance
    obs_dIQUV = np.zeros(4) + np.nan
    for k in range(4):
        obs_dIQUV[k] = np.max(obs_IQUV[:,k]) - np.min(obs_IQUV[:,k])
    iS_sorted = np.argsort(obs_dIQUV)[::-1]
    
    # -------------------------------
    i0_sorted= np.array([-1])
    chisqr_err_arr_IQUV = chisqr_err_arr_IQUV0+0. # keep orig untouched
    n2 = int(initial_n0)

    for k in iS_sorted[:]:
        tS_chisqr = chisqr_err_arr_IQUV[:,k] # get the chisqr for this Stokes param
        i1_sorted = np.argsort(tS_chisqr)[:n2] # sort chisqr and cut out n2 best fits

        if i0_sorted[0] == -1:
            i0_sorted = i1_sorted 
        else:      
            i0_sorted = i0_sorted[i1_sorted] # rearrange i0_sorted according to this Stokes param

        chisqr_err_arr_IQUV = chisqr_err_arr_IQUV0[i0_sorted, :]+0. # cut out chisqr_err corres. to i0_sorted
        # print(n2, chisqr_err_arr_IQUV.shape, n2/fact, '\n')
        n2= round(n2/fact) # update n2

    # -------------------------------
    i0_best = i0_sorted[0]    
    chisqr_best = chisqr_err_arr_IQUV0[i0_best, :]
    
    return i0_best, chisqr_best

# --------------------------------------
# code status: red, incomplete
# @njit()
def INCOMPLETE_pCMEx_remove_duplicate_Stokes_profiles(Stokes_2Darr=None):
    '''

    '''
    # Assumption- that pre-processing is already done
    # SI SQ SU.. proper normalization factors is already applied

    new_Stokes_2Darr = Stokes_2Darr + np.nan
    tStokes_2Darr = Stokes_2Darr +0.

    for k in range(tStokes_2Darr.shape[1]-1):
        tStokes1 = new_Stokes_2Darr[:,k]

        # for k1 in range()

# --------------------------------------
# code status: green. Matched dict values with input 2Darr
def pCMEx_RCasini_model_2Darr_to_1Darr(RCasini_model_2Darr=None):
    '''
    written on Jan 11 2024
    Goal: convert RCasini_model_2Darr into a dictionary - that is more readable and we
          understand what different variables are.
    
    Input: 
    
    RCasini_model_2Darr format:
    
        [h,     anired, tau,    -    - ]  ; (height,anisotropy reduction, optical depth) 
        [theta, phi,    gamma,  -    - ]  ; (LOS information)
        [B,     thetaB, phiB,  TBlos, PBlos]; (Bfield information) 
        [T,     vlos,    -      -    - ]  ; (temperature and LOS velocity within a pixel)
    
    
    Output:
        RCasini_model_1Darr: RCasini_model_2Darr data stored in a dictionary 
        [0, 1, 2] = h, anired, tau
        [3, 4, 5] = theta, phi, gamma
        [6, 7, 8, 9, 10] = B, thetaB, phiB,  TBlos, PBlos
        [12, 13] = T, vlos
    '''

    RCasini_model_1Darr = np.zeros(13) + np.nan
    RCasini_model_1Darr[:3]    = RCasini_model_2Darr[0,:3]
    RCasini_model_1Darr[3:6]   = RCasini_model_2Darr[1,:3]
    RCasini_model_1Darr[6:11]  = RCasini_model_2Darr[2,:]
    RCasini_model_1Darr[11:13] = RCasini_model_2Darr[3,:2]
    
    return RCasini_model_1Darr


# code status: green. Matched dict values with input 2Darr
def pCMEx_RCasini_model_1Darr_to_2Darr(RCasini_model_1Darr=None):
    '''
    written on Jan 11 2024
    Goal: convert RCasini_model_2Darr into a dictionary - that is more readable and we
          understand what different variables are.
    
    Input: 
    
    RCasini_model_2Darr format:
    
        [h,     anired, tau,    -    - ]  ; (height,anisotropy reduction, optical depth) 
        [theta, phi,    gamma,  -    - ]  ; (LOS information)
        [B,     thetaB, phiB,  TBlos, PBlos]; (Bfield information) 
        [T,     vlos,    -      -    - ]  ; (temperature and LOS velocity within a pixel)
    
    
    Output:
        RCasini_model_1Darr: RCasini_model_2Darr data stored in a dictionary 
        [0, 1, 2] = h, anired, tau
        [3, 4, 5] = theta, phi, gamma
        [6, 7, 8, 9, 10] = B, thetaB, phiB,  TBlos, PBlos
        [12, 13] = T, vlos
    '''

    RCasini_model_2Darr = np.zeros((4,5)) + np.nan
    RCasini_model_2Darr[0,:3]    = RCasini_model_1Darr[:3]
    RCasini_model_2Darr[1,:3]    = RCasini_model_1Darr[3:6]
    RCasini_model_2Darr[2,:]    = RCasini_model_1Darr[6:11]
    RCasini_model_2Darr[3,:2]    = RCasini_model_1Darr[11:13]

    # RCasini_model_2Darr[3:6]   = RCasini_model_1Darr[1,:3] [3:6]
    # RCasini_model_2Darr[6:11]  = RCasini_model_1Darr[2,:]
    # RCasini_model_2Darr[11:13] = RCasini_model_1Darr[3,:2]
    
    return RCasini_model_2Darr

# --------------------------------------
# code status: green. Matched dict values with input 2Darr
def pCMEx_RCasini_model_to_dict(RCasini_model_2Darr=None):
    '''
    written on Jan 11 2024
    Goal: convert RCasini_model_2Darr into a dictionary - that is more readable and we
          understand what different variables are.
    
    Input: 
    
    RCasini_model_2Darr format:
    
        [h,     anired, tau,    -    - ]  ; (height,anisotropy reduction, optical depth) 
        [theta, phi,    gamma,  -    - ]  ; (LOS information)
        [B,     thetaB, phiB,  TBlos, PBlos]; (Bfield information) 
        [T,     vlos,    -      -    - ]  ; (temperature and LOS velocity within a pixel)
    
    
    Output:
        RCasini_model_dict: RCasini_model_2Darr data stored in a dictionary 
    '''
    
    h, anired, tau0                 = RCasini_model_2Darr[0,:3] # (height,anisotropy reduction, optical depth) 
    theta, phi, gamma               = RCasini_model_2Darr[1,:3] # (LOS information)
    B, thetaB, phiB, T_Blos, P_Blos = RCasini_model_2Darr[2,:] # (Bfield information) 
    T_K, vlos                       = RCasini_model_2Darr[3,:2] # (temperature and LOS velocity within a pixel)
    
    RCasini_model_dict= dict(h=h, anired=anired, tau0=tau0, 
                             theta=theta, phi=phi, gamma=gamma,
                             B=B, thetaB=thetaB, phiB=phiB, T_Blos=T_Blos, P_Blos=P_Blos,
                             T_K=T_K, vlos=vlos, 
                             RCasini_model_2Darr = RCasini_model_2Darr)
    
    return RCasini_model_dict

# --------------------------------------
def pCMEx_Jan30_24_combine_PCAinv_results(fold_fnm=None):
    '''
    written on Jan 30 2024
    Goal: combine indiv results from pCMEx_PCA_inv_1_profile_main to a single file.


    '''
    print(1)

    # get file list
    # sort them in order - assume that they have same naming and have k value to them
    # now read and combine them to file..!!?
    # store them in the same format as that in the .ipynb file

    file_list = np.array(glob.glob('{}/*.npz'.format(fold_fnm)))
    for k in range(file_list.size):
        tfile = file_list[k]

        # with 

# --------------------------------------
# @njit() - no need as such - leads to same time as no-njit..!!
def pCMEx_PCA_inv_1_profile(
        # dtime=None, # to get around Joblib cachce issue
        key_return_invmodel=0, 

        # ----------------------- needed to save PCAinv result to file
        key_save_inv_result_to_file=0,
        stats_savefold='./', 
        index_k=0, 

        # ----------------------- when using parallelized version
        k0=None, 
        obs_IQUV_all=None,                            
        obs_model_all=None, 

        obs_IQUV=None,
        obs_model_2Darr=None, # needed to be saved in savefile

        obs_IQUV0=None, # needed for key_apr26_find_models_within_COF=1
        obs_IQUV_all0=None, # needed for key_apr26_find_models_within_COF=1

        # -----------------------
        PCA_dbase_coeff_all=None, 
        PCA_dbase_model_all=None, 
        PCA_eigenbasis_IQUV=None, 
        PCA_IQUV_mean0=None, 
        key_normalize_obs_IQUV_to_Imax=1, 
        key_subtract_PCA_IQUV_mean0=1, 

        # -----------------------
        key_search_using_PCAbarcodes=0, 
            PCA_dbase_barcodes=np.zeros((1,1))+np.nan, 
            PCA_barcode_nmode=0, # barcode calc corresponding to how many modes

        nmodes_to_use_arr=np.array([-1,-1,-1,-1]), # nmodes for chisqr for each IQUV

        # -----------------------
        key_mar15_restrict_hproj_search_inPCAdbase=0, 
            param_dh_range=1e-3, 
            PCAdbase_proj_h_arr=None, 
            ll_to_nmodels_restrct_hproj_search=10, 

        # -----------------------
        key_apr11_filter_dbase_search_below_NL=0, 
            noise_level_wrt_obs_Imax=1e-3, 
            PCA_dbase_C_L2n_all=np.zeros((1,1))+np.nan,

        
        key_use_IQUV_weights_for_chisqr=1, # 0 = weights are all 1s
            # key_use_L2p_as_weights=0, 
            key_use_obsIQUV_ratio_as_wts_for_chisqr=0, 
            key_use_obsIQUV_L2n_as_wts_for_chisqr=0, 
            key_use_obsIQUV_L1n_as_wts_for_chisqr=0, 
            key_use_IQUV_L2product_as_wts_for_chisqr=0,
            max_chisqr_wt_UnSq=np.nan, # NaN = weights are unounded

        key_use_GM_for_chisqr=0, 
        key_print_PAnotes=0, 
        chisqr_wts_IQUV_fact0=np.array([1,1,1,1]), 

        # --------------------------
        # May 3
        key_use_rCnl=0, 
        obs_C_IQUV0=np.zeros([2,2])+np.nan, 
        # rCnl_diff_thresh_percent=10., # removed on May 13 2024, (successor) rCnl_diff_thresh_percent_arr
        rCnl_diff_thresh_percent_arr=np.zeros(4)+np.nan, # added on May 13 2024
        rCnl_mask_value=np.nan,
        obs_C_IQUV_all0=None, 

        # noise_level_wrt_obs_Imax=np.nan, 

        

        # key_use_IQUonly_for_chisqr_calc=0, 

    
        # --------------------------------
        # Feb 21 - it seems like coeff less than noise level cause bad inversions
        # so check if discarding them from chisqr helps or not
        # key_feb21_discard_obs_C_IQUV_lt_noiselevel=0, 
        # obs_C_IQUV_threshold=0., 

        # ----------------- Feb 20 2024
        # temporary keys - to manually amplify SV 
        # key_manually_amplify_SV_weight=0, 
        # SV_wt_amp_factor=5.,

        # n1=7, 

        # Mar5  - if obsIQUV < noise , then do stuff
        # key_mar5_inc_obsIQUVnoise_in_chisqr_calc=0, 
        # key_mar5_est_IQUVnoise_with_all_wvlnths=1, 
        # key_mar5_rmv_obsIQUV_if_below_noise=0,
        # noise_level_wrt_obs_Imax=1e-3, 
        # key_mar12_PCAdbase_search_below_noise_only=0, 
        # key_use_avg_obsS_as_noise_ul=0, 
        # -------------------------------
        # reweight chisqr to address chisqr imbalance
        # key_mar10_reweight_chisqr=0, # discarded- reweighting to equalize chisqr contribution 
        
                    ):

    '''
    Goal: invert 1 profile - and 
    
        
    Input:
        - max_chisqr_wt_UnSq
            - UnSq = un squared weights


    '''
    # def temp_func():
    #     return
    # void = temp_func() # does nothing


    if k0 is not None:
       obs_IQUV = obs_IQUV_all[:,:,k0] +0.
       index_k= k0
       obs_model_2Darr = obs_model_all[:,:,k0]
       obs_IQUV0 = obs_IQUV_all0[:,:,k0]

       obs_C_IQUV0 = obs_C_IQUV_all0[:,:,k0]
       


    # ----------------------------------------------
    # compute chisqr against each in PCA database spectra
    indx_arr_ALL, chisqr_err_arr, chisqr_err_wtd_arr_IQUV, chisqr_wts_arr_IQUV, \
            obsIQUV_abv_noise_arr, obs_C_IQUV, obs_IQUV_rescaled, NL_mask_IQUV = \
                pCMEx_calc_chisqr_err_arr_for_1_obs_profile(
            # needed to conv obs_IQUV to obs_PCAcoeff
            obs_IQUV=obs_IQUV, 
            PCA_eigenbasis_IQUV=PCA_eigenbasis_IQUV, 
            PCA_IQUV_mean0=PCA_IQUV_mean0, 
            key_normalize_obs_IQUV_to_Imax=key_normalize_obs_IQUV_to_Imax, 
            key_subtract_PCA_IQUV_mean0=key_subtract_PCA_IQUV_mean0,

            # --------------------
            key_search_using_PCAbarcodes=key_search_using_PCAbarcodes, 
            PCA_dbase_barcodes=PCA_dbase_barcodes, 
            PCA_barcode_nmode=PCA_barcode_nmode, # barcode calc corresponding to how many modes

            # --------------------
            nmodes_to_use_arr=nmodes_to_use_arr, # how many modes to use for chisqr for each IQUV
            # nmodes_to_use=nmodes_to_use, 

            # --------------------
            key_mar15_restrict_hproj_search_inPCAdbase=key_mar15_restrict_hproj_search_inPCAdbase, 
            obs_model_2Darr=obs_model_2Darr, 
            param_dh_range=param_dh_range, 
            PCAdbase_proj_h_arr=PCAdbase_proj_h_arr, 
            ll_to_nmodels_restrct_hproj_search=ll_to_nmodels_restrct_hproj_search, 

            # --------------------
            key_apr11_filter_dbase_search_below_NL=key_apr11_filter_dbase_search_below_NL, 
            noise_level_wrt_obs_Imax=noise_level_wrt_obs_Imax, 
            PCA_dbase_C_L2n_all=PCA_dbase_C_L2n_all, # size = (nmodels,4); when comp. dbase C_L2n with thresh_NL

            # --------------------
            key_use_IQUV_weights_for_chisqr=key_use_IQUV_weights_for_chisqr, 
            # key_use_L2p_as_weights=key_use_L2p_as_weights, 
            key_use_obsIQUV_ratio_as_wts_for_chisqr=key_use_obsIQUV_ratio_as_wts_for_chisqr, 
            key_use_obsIQUV_L2n_as_wts_for_chisqr=key_use_obsIQUV_L2n_as_wts_for_chisqr, 
            key_use_obsIQUV_L1n_as_wts_for_chisqr=key_use_obsIQUV_L1n_as_wts_for_chisqr, 
            key_use_IQUV_L2product_as_wts_for_chisqr=key_use_IQUV_L2product_as_wts_for_chisqr,
            max_chisqr_wt_UnSq=max_chisqr_wt_UnSq, 
            chisqr_wts_IQUV_fact0=chisqr_wts_IQUV_fact0, 

            # --------------------
            # needed for chisqr calculation 
            PCA_dbase_coeff_all=PCA_dbase_coeff_all,
            key_use_GM_for_chisqr=key_use_GM_for_chisqr, # GM = geometric mean

            key_print_PAnotes=key_print_PAnotes,

            # --------------------
            key_use_rCnl=key_use_rCnl,             
            # rCnl_diff_thresh_percent=rCnl_diff_thresh_percent, 
            rCnl_diff_thresh_percent_arr=rCnl_diff_thresh_percent_arr, # added on May 13 2024
            rCnl_mask_value=rCnl_mask_value,
            obs_C_IQUV0=obs_C_IQUV0, 
            # key_use_IQUonly_for_chisqr_calc=key_use_IQUonly_for_chisqr_calc, 
            # key_feb21_discard_obs_C_IQUV_lt_noiselevel=key_feb21_discard_obs_C_IQUV_lt_noiselevel, 
            # obs_C_IQUV_threshold=obs_C_IQUV_threshold, 

            # key_manually_amplify_SV_weight=key_manually_amplify_SV_weight, 
            # SV_wt_amp_factor=SV_wt_amp_factor,

            # Mar5  - if obsIQUV < noise , then do stuff
            # key_mar5_inc_obsIQUVnoise_in_chisqr_calc=key_mar5_inc_obsIQUVnoise_in_chisqr_calc, 
            # key_mar5_est_IQUVnoise_with_all_wvlnths=key_mar5_est_IQUVnoise_with_all_wvlnths, 
            # key_mar5_rmv_obsIQUV_if_below_noise=key_mar5_rmv_obsIQUV_if_below_noise,
            # noise_level_wrt_obs_Imax=noise_level_wrt_obs_Imax, 
            # key_mar12_PCAdbase_search_below_noise_only=key_mar12_PCAdbase_search_below_noise_only, 
            # key_use_avg_obsS_as_noise_ul=key_use_avg_obsS_as_noise_ul, 
                
                 )   

    # ----------------------------------------------
    if chisqr_err_arr[~np.isnan(chisqr_err_arr)].size ==0: # all values in chisqr are nans
        # print(chisqr_err_arr[~np.isnan(chisqr_err_arr)].size) 
        return 

    # ----------------------------------------------
    # Get PCA database model with min chisqr_err (best fit spectra) - RCasini way
    i_bestmodel = np.argwhere(chisqr_err_arr == np.nanmin(chisqr_err_arr))[:,0]
    if i_bestmodel.size != 1:
        print(i_bestmodel, np.nanmax(chisqr_err_arr), np.nanmin(chisqr_err_arr))
        # raise NameError('PAerr: multiple best fit models exist in PCA database.')
    i_bestmodel = i_bestmodel[0]

    indx_PCA_bestmodel = indx_arr_ALL[i_bestmodel] # get the dbase location for best fit model
    inv_model_2Darr = PCA_dbase_model_all[:,:,indx_PCA_bestmodel][:4,:] # best fit model in PCA database
    
    # ----------------------------------------------
    # convert inv_model_2Darr to 1D array. Will later think if it's better to store as dict or not    
    inv_model = pCMEx_RCasini_model_2Darr_to_1Darr(RCasini_model_2Darr=inv_model_2Darr[:4,:]) # 1D arr
    obs_model=None
    if obs_model_2Darr is not None:
        obs_model = pCMEx_RCasini_model_2Darr_to_1Darr(RCasini_model_2Darr=obs_model_2Darr) # 1D arr
    
    # ----------------------------------------------
    best_chisqr_err = chisqr_err_arr[i_bestmodel]
    best_chisqr_wtd_err_IQUV = chisqr_err_wtd_arr_IQUV[i_bestmodel, :] # weighted chisqr IQUV err

    if key_use_IQUV_L2product_as_wts_for_chisqr==1:
        best_chisqr_wts_IQUV = chisqr_wts_arr_IQUV[i_bestmodel,:].flatten()
    else:
        best_chisqr_wts_IQUV = chisqr_wts_arr_IQUV.flatten() # 1Darr

    # print(best_chisqr_wtd_err_IQUV, i_bestmodel, )

    # ----------------------------------------------
    # ----------------------------------------------
    # COC_stats_GM = None
    # COC_stats_AM = None
    COC_radius = -1
    COC_ix_all = -1  # index of 
    COC_ix_all3 = -1 # one that is n times away from 
    # indx_arr_ALL # list of index for which we calculated chisqr
    # chisqr_err_arr
    # obs_C_IQUV0=None

    key_apr26_COCwork=1 # COC = circle of confusion
    if key_apr26_COCwork==1:
        # get the number of models that are within the Circle of confusion radius...
        
        # obs_C_IQUV0
        # nmodes_to_use_arr
        # chisqr_wts_arr_IQUV
        # obs_C_IQUV - noised up
        # indx_arr_ALL, chisqr_err_arr - to help find models within circle of confusion
        # get COC radius

        if key_use_IQUV_L2product_as_wts_for_chisqr ==1:
            raise NameError('this wont work with key_use_IQUV_L2product_as_wts_for_chisqr=1')

        # if obs_IQUV0 is None:
        # #     obs_IQUV0 = obs_IQUV_all0[:,:,k0]
        # obs_C_IQUV0, void = pCMEx_decomp_IQUV_to_PCA_coeff(IQUV=obs_IQUV0,
        #                 PCA_eigenbasis_IQUV=PCA_eigenbasis_IQUV, 
        #                 PCA_IQUV_mean0=PCA_IQUV_mean0,
        #                 key_normalize_obs_IQUV_to_Imax=key_normalize_obs_IQUV_to_Imax,
        #                 key_subtract_PCA_IQUV_mean0=key_subtract_PCA_IQUV_mean0, )

        # COC_stats_GM = np.zeros((1,)) # n within COF, 
        # COC_stats_AM = np.zeros((1,)) # n within COF, 

        # print(obs_C_IQUV0)

        if key_use_GM_for_chisqr ==1:
            COC_radius = pCMEx_apr26_calc_GM_chisqr_err_bn_C_IQUV_pair(C1_IQUV=obs_C_IQUV0, C2_IQUV=obs_C_IQUV, 
                                      chisqr_wts_IQUV=chisqr_wts_arr_IQUV.flatten(), 
                                      nmodes_to_use_arr=nmodes_to_use_arr )
        else:
            COC_radius = pCMEx_apr26_calc_AM_chisqr_err_bn_C_IQUV_pair(C1_IQUV=obs_C_IQUV0, C2_IQUV=obs_C_IQUV, 
                                      chisqr_wts_IQUV=chisqr_wts_arr_IQUV.flatten(), 
                                      nmodes_to_use_arr=nmodes_to_use_arr )

        COC_ix_all1 = np.argwhere(chisqr_err_arr <= COC_radius)[:,0]
        COC_ix_all = indx_arr_ALL[COC_ix_all1] # PCA dbase indx within COC (includes orig, no noise model)

        key_print1 = 0
        if key_print1 ==1:
            print('\n COC_radius = ', COC_radius)
            print('\n all indx within COC = ', COC_ix_all, indx_PCA_bestmodel)
            print('\n chsiqr err (for models within COC) = ', chisqr_err_arr[COC_ix_all1])
            print('\n sorted chisqr best = ', sorted(chisqr_err_arr)[:5])

            print('\n',obs_model_2Darr[:4,:])
            print('\n % diff = ')
            print((1-inv_model_2Darr[:4,:]/obs_model_2Darr[:4,:])*100.)
            # print('\n',PCA_dbase_model_all[:4,:, 1084])
            print('\n best PCA coeff = \n', PCA_dbase_coeff_all[:,:,indx_PCA_bestmodel]*NL_mask_IQUV, '\n')


        COC_radiusn = 3*COC_radius
        COC_ix_alln = np.argwhere(chisqr_err_arr <= COC_radiusn)[:,0]
        COC_ix_alln = indx_arr_ALL[COC_ix_alln] # PCA dbase indx within COC (includes orig, no noise model)

        # if ixx0.size > 0:
        #     print('ixx0.size = ', ixx0.size)

        # print('\n', COC_ix_all, COC_radius, chisqr_err_arr.shape)
        # # print(obs_C_IQUV0)
        # # print(chisqr_wts_arr_IQUV)
        # print(np.nanmin(chisqr_err_arr))
        # print(indx_PCA_bestmodel)
        # # print(indx_arr_ALL[np.argwhere(chisqr_err_arr <= COC_radius)[:,0]])
        # ixx02 = np.argwhere(chisqr_err_arr == np.min(chisqr_err_arr))[:,0]
        # print(indx_arr_ALL[ixx02], chisqr_err_arr[ixx02])



    # # ----------------------------------------------

    # ----------------------------------------------
    # save stuff to file - needed when parallelizing
    if key_save_inv_result_to_file ==1:
        # save_fnm1 = '{}/{}_{}.pickle'.format(stats_savefold, prefix_savefnm, index_k)
        save_fnm1 = '{}/k={:05d}.pickle'.format(stats_savefold, index_k)

        # # check if save_fnm1 already exists - if yes, raise error
        # if os.path.isfile(save_fnm1):
        #     raise NameError('PAerr: \n {} \n already exists. Be careful not to overwrite something'.format(save_fnm1))

        tdict= dict(
                    obs_model=obs_model,    
                    inv_model=inv_model,                                        
                    best_chisqr_err=best_chisqr_err, # sum of weighted chisqr - for best fit
                    best_chisqr_err_IQUV=best_chisqr_wtd_err_IQUV, # indiv chisqr IQUV
                    best_chisqr_wts_IQUV=best_chisqr_wts_IQUV, # weights used
                    nmodes_to_use_arr=nmodes_to_use_arr, 
                    obs_C_IQUV=obs_C_IQUV, 
                    inv_C_IQUV=PCA_dbase_coeff_all[:,:,indx_PCA_bestmodel],
                    
                    obs_IQUV=obs_IQUV, 
                    obsIQUV_abv_noise_arr=obsIQUV_abv_noise_arr, 

                    # COC stuff below
                    # chisqr_wts_arr_IQUV=chisqr_wts_arr_IQUV, # = best_chisqr_wts_IQUV
                    # chisqr_err_arr=chisqr_err_arr, 
                    # obs_C_IQUV0=obs_C_IQUV0, 
                    indx_PCA_bestmodel=indx_PCA_bestmodel, # best index models
                    COC_ix_all=COC_ix_all, # index of all within COC
                    COC_radius=COC_radius,                     
                    # COC_radiusn=COC_radiusn, 
                    # COC_ix_alln=COC_ix_alln,




                    # coeff_inv_IQUV=PCA_dbase_coeff_all[:nmodes_to_use,:,i_bestmodel], 
                    # indx_arr_ALL=indx_arr_ALL, 
                    # nmodes_to_use_arr=nmodes_to_use_arr, 

                    # # chisqr_mean_IQUV=chisqr_mean_IQUV, # mean IQUV chisqr over dbase model, 
                    # chisqr_wtd_err_arr_IQUV=chisqr_wtd_err_arr_IQUV, 
                    # IQUV_noise_or_not=IQUV_noise_or_not, 
                   )

        with open(save_fnm1, 'wb') as f:
            pickle.dump(tdict, f)
        tdict = None 

    # ----------------------------------------------
    if key_return_invmodel ==1:
        return inv_model, obs_model, \
               chisqr_err_arr, chisqr_err_arr_IQUV, \
               obs_C_IQUV, i_bestmodel        
    else: 
        inv_model, obs_model = None, None
        chisqr_err_arr, chisqr_err_arr_IQUV = None, None
        obs_C_IQUV, i_bestmodel = None, None

        return # None, None, None, None, None, None
# --------------------------------------
# code status: green, triple checked on Jan 16 2024
@njit()
def pCMEx_calc_chisqr_err_arr_for_1_obs_profile(
        obs_IQUV=None, 
        PCA_eigenbasis_IQUV=None, 
        PCA_IQUV_mean0=None, 
        key_normalize_obs_IQUV_to_Imax=1, 
        key_subtract_PCA_IQUV_mean0=1,

        # --------------------
        key_search_using_PCAbarcodes=0, 
        PCA_dbase_barcodes=np.zeros((1,1))+np.nan, 
        PCA_barcode_nmode=0, # barcode calc corresponding to how many modes

        # --------------------
        nmodes_to_use_arr=np.array([-1,-1,-1,-1]), # nmodes for chisqr for each IQUV
        # nmodes_to_use=nmodes_to_use, 

        # --------------------
        key_mar15_restrict_hproj_search_inPCAdbase=0, 
        obs_model_2Darr=None, 
        param_dh_range=1e-3, 
        PCAdbase_proj_h_arr=None, 
        ll_to_nmodels_restrct_hproj_search=10, 

        # --------------------
        key_apr11_filter_dbase_search_below_NL=0, 
        # key_apr11_restrict_dbase_search_below_NL=0, 
        noise_level_wrt_obs_Imax=1e-3, 
        PCA_dbase_C_L2n_all=np.zeros((1,1))+np.nan, # size = (nmodels,4); when comp. dbase C_L2n with thresh_NL

        # --------------------
        key_use_IQUV_weights_for_chisqr=1, 
        # key_use_L2p_as_weights=key_use_L2p_as_weights, 
        key_use_obsIQUV_ratio_as_wts_for_chisqr=0, 
        key_use_obsIQUV_L2n_as_wts_for_chisqr=0, 
        key_use_obsIQUV_L1n_as_wts_for_chisqr=0, 
        key_use_IQUV_L2product_as_wts_for_chisqr=0,
        max_chisqr_wt_UnSq=np.nan, 
        chisqr_wts_IQUV_fact0=np.array([1,1,1,1]),

        # --------------------
        # needed for chisqr calculation 
        PCA_dbase_coeff_all=None,
        key_use_GM_for_chisqr=0, # GM = geometric mean

        # --------------------
        # May 3 2024 : learning: using only those IQUV modes that are above noise level
        # when we know what exactly is the noise contribution 
        # (unlike key_apr11_filter_dbase_search_below_NL=1)
        key_use_rCnl=0, 
        obs_C_IQUV0=np.zeros([2,2])+np.nan, 
        # rCnl_diff_thresh_percent=10.,  
        rCnl_diff_thresh_percent_arr=np.zeros(4)+np.nan, # added on May 13 2024
        rCnl_mask_value=np.nan,



        key_print_PAnotes=0,

        

        # key_use_IQUonly_for_chisqr_calc=0, # No Stokes V in chisqr 
        # max_chisqr_wt_UnSq=np.nan, # thresholding limit

        # # -----------------
        # # temporary keys - to manually amplify SV 
        # # key_manually_amplify_SV_weight=0, 
        # # SV_wt_amp_factor=5.,

        # # -----------------
        # # Mar5  - if obsIQUV < noise , then do stuff
        # # key_mar5_inc_obsIQUVnoise_in_chisqr_calc=0, 
        # # key_mar5_est_IQUVnoise_with_all_wvlnths=1, 
        # # key_mar5_rmv_obsIQUV_if_below_noise=0,
        # noise_level_wrt_obs_Imax=1e-3,

            ):
    '''
    Successor to: "OLD_pCMEx_calc_chisqr_err_arr_for_1_obs_profile_used_b4_Apr12"
    This code re-written on: Apr 18 2024

    Written on Jan 11 2024
    Goal to do PCA inversion for a single profile and return chisqr_err_arr 
            (measure of how close are the eigen coeff for the obs_IQUV wrt each profile in the database)

    Input:
        - obs_IQUV: nlambda by 4 matrix, in IQUV order
        - PCA_eigenbasis_IQUV : (4 by nlambda by nmodes) matrix containing eigen modes of the PCA basis
        - PCA_IQUV_mean0: nlambda by 4 matrix containing mean value at each wavelength for each IQUV
                            - these are likely normalized to obs_Imax
        - PCA_dbase_coeff_all: nmodes by 4 by number of profiles in the database
                            - it is this database to which coeff for obs_IQUV would be compared to

        - nmodes_to_use = total number of modes to use when decomposing obs_IQUV onto PCA_eigenbasis_IQUV
            - -1 implies using all the modes in PCA_eigenbasis_IQUV

        - max_chisqr_wt_UnSq:np.nan, max chisqr weight for a given say Stokes V to be used to amplify its relative contribution to chisqr calculation
            - If np.nan - then weights are unbounded

        - keywords
            - key_normalize_obs_IQUV_to_Imax =1 to normalize all values in obs_IQUV with obs_Imax
            - key_subtract_IQUV_mean0 =1 to subtract PCA_IQUV_mean0 from obs_IQUV. A standard in PCA
            - key_use_IQUV_weights_for_chisqr =1 to use weighted chisqr
                - weights are calculated using pCMEx_get_IQUV_weights_for_chisqr_calc, which also
                    requires noise_level_wrt_obs_Imax. 
                    - noise_level_wrt_obs_Imax is more a factor. noise_level_wrt_obs_Imax*obs_Imax is the 
                      noise level in the obs_IQUV

            - key_print_PAnotes =1 to pring some PA notes

    Output:
        - chisqr_err_arr : Euclidean distance of coeff_obs_IQUV1 wrt each coeff in PCA_dbase_coeff_all
        - coeff_obs_IQUV1 : eigen coeff for obs_IQUV
        - obs_IQUV_rescaled: Note this could be normalized or not depending on if key_normalize_obs_IQUV_to_Imax=1 or not.

    - PA concerns:
        - does it make sense to use noise_level as upper limit in chisqr_wts calculation

    Steps:
        1) normalize obs_IQUV data by Imax
        2) subtract PCA_IQUV_mean0 from obs_IQUV
        3) decompose obs_IQUV1 on PCAbasis and calc obs_coeff_IQUV
        4) get chisqr weights for IQUV
        5) calc chisqr_err_arr: for obs_coeff_IQUV and coeff for each profile 
           in the PCA databbase weighted by respective chisqr weights
        6) return chisqr_err_arr

    Updates:
        [Jan 18 2024]: PCA_eigenbasis_IQUV shape changed to (4, nlambda, nmodes) from (nlambda, nmodes, 4).
                            - Reason: Prevents numba error: @ matrix multiplication is faster on 
                                      contiguous arrays.

    '''

    # print('key_use_IQUV_weights_for_chisqr =', key_use_IQUV_weights_for_chisqr)

    if nmodes_to_use_arr[0] ==0:
        raise NameError('PAerr (May1 2024) - I-component 0 leads to all weights = NaN when using L2n of Coeffs.')

    nmodels_in_PCAdbase = PCA_dbase_coeff_all.shape[2]
    # --------------------------------------------------------
    # decompose observed IQUV into Coeff IQUV
    # obs_C_IQUV = PCA Coeffs for observed IQUV
    # obs_IQUV_rescaled = rescaled using peak SI, PCAdbase mean subtracted
    obs_C_IQUV, obs_IQUV_rescaled = pCMEx_decomp_IQUV_to_PCA_coeff(IQUV=obs_IQUV,
                        PCA_eigenbasis_IQUV=PCA_eigenbasis_IQUV, 
                        PCA_IQUV_mean0=PCA_IQUV_mean0,
                        key_normalize_obs_IQUV_to_Imax=key_normalize_obs_IQUV_to_Imax,
                        key_subtract_PCA_IQUV_mean0=key_subtract_PCA_IQUV_mean0, )
    # --------------------------------------------------------
    indx_arr_ALL = np.arange(nmodels_in_PCAdbase)+0. # dbase indices for which we compute chisqr
    # --------------------------------------------------------
    # May 3 2024 work
    NL_mask_IQUV = np.ones(obs_C_IQUV.shape) +0.
    if key_use_rCnl ==1:
        max_ePCA_nmodes = PCA_eigenbasis_IQUV.shape[2] # max no. of ebasis
        max_cPCA_nmodes = PCA_dbase_coeff_all.shape[0] # max no. of modes in PCA_dbase_coeff_all
        


        if ((key_search_using_PCAbarcodes ==1) | 
            ((nmodes_to_use_arr != max_ePCA_nmodes).any()) | 
            ((nmodes_to_use_arr != max_cPCA_nmodes).any()) | 
            (key_apr11_filter_dbase_search_below_NL ==1) |
            ((chisqr_wts_IQUV_fact0 != 1).any())

            ):

            raise NameError('PAerr: Recheck these conditions.')

            # if key_use_rCnl ==1:
            #     - nmodes_to_use_arr = all are PCA_dbase_coeff_all.shape[0] # as removal is done based on which coeffs are < noise
            #     - key_search_using_PCAbarcodes =0
            #     - key_apr11_filter_dbase_search_below_NL=0 # to avoid mixing things
            #     - chisqr_wts_IQUV_fact0 = 1 for all IQUV - masking done based on noise level   
            #     - key_use_IQUV_L2product_as_wts_for_chisqr =0

        # print(obs_C_IQUV, '\n', )
        obs_C_IQUV, NL_mask_IQUV = pCMEx_may3_rCnl_mask_coeff_if_diff_gt_NL(
                                    C_IQUV0=obs_C_IQUV0, C_IQUV=obs_C_IQUV, 
                                    # rCnl_diff_thresh_percent=rCnl_diff_thresh_percent, 
                                    rCnl_diff_thresh_percent_arr=rCnl_diff_thresh_percent_arr, 
                                    rCnl_mask_value=rCnl_mask_value)

        
        key_print1 = 0
        if key_print1 ==1:
            # print(obs_C_IQUV0, '\n', )
            print(obs_C_IQUV, '\n')



    # --------------------------------------------------------
    if key_search_using_PCAbarcodes ==1:
        ''' 
        Apr 1 2024 work
        - Added this key to search only those dbase profiles for which barcode/index matches that of obs_profile
        where the dbase barcode matches with that of the obs_profile
        '''
        raise NameError('PAerr:Apr15- needs to be rechecked/tested for accuracy.')
        obs_C_barcode, str_barcode_binary = pCMEx_get_PCAcoeff_barcode_index(PCA_C_IQUV_2Darr=obs_C_IQUV, 
                                                                         max_nmodes=PCA_barcode_nmode)
        obs_C_barcode = obs_C_barcode[PCA_barcode_nmode-1]+0.

        # indx_arr_ALL1 = np.argwhere(PCA_dbase_barcodes == obs_C_barcode)[:,0]
        # indx_arr_ALL = indx_arr_ALL[PCA_dbase_barcodes == obs_C_barcode] # before Apr 11
        indx_arr_ALL[PCA_dbase_barcodes != obs_C_barcode] = np.nan # Apr 11 # set index to NaN where barcodes don't match
        # raise NameError('check above line')

        # print(PCA_dbase_barcodes[indx_arr_ALL].max(), PCA_dbase_barcodes[indx_arr_ALL].min(), obs_C_barcode)
        # print(obs_C_barcode, barcode_binary, indx_arr_ALL.size, obs_C_IQUV[:2,:])

        # if indx_arr_ALL.size ==0:
        #     raise NameError('No model in PCAdbase with this obs_profile barcode.')

    # --------------------------------------------------------
    # Use only a certain set of modes for chisqr calculation for each IQUV
    # these number for each IQUV can be different, (before Apr 18, nmodes were equal for all IQUV])
    # set obs_C_IQUV with index > nmodes_to_use_arr to 0.
    # this needs to be done after barcodes have been created.
    tk=-1
    for tn in nmodes_to_use_arr:
        tk+=1        
        if ((tn > PCA_eigenbasis_IQUV.shape[2]) | (tn < 0)):
            raise NameError('PAerror: nmodes_to_use is either > nmodes in PCA_eigenbasis_IQUV or is < 0.')    
        obs_C_IQUV[tn:, tk] = np.nan # if tn=0, then all coeffs for that tk gets set to 0.
    
    # --------------------------------------------------------
    if key_mar15_restrict_hproj_search_inPCAdbase==1:
        '''
        See Mar 15 conversation with RCasini. He thinks that we can get around a lot of uncertainty in inversion result by restricting the search to +/- .002R of the obs_projected h
        # projected height = h_proj = (1 + h) sin(theta) - 1; Assumes h in proj units
        # Apr 2 - calculate chisqr for only those models that fall within the projH range.
        #         This should speed up calculation!
        '''
        pi180 = np.pi/180.

        # ---------- get obs projected height and height range
        obs_h = obs_model_2Darr[0,0]
        obs_theta = obs_model_2Darr[1,0]*pi180        
        obs_proj_h = (1+obs_h) * np.sin(obs_theta) -1 # projected height formula

        dh = param_dh_range # in Rsun units
        proj_hmax = obs_proj_h + dh
        proj_hmin = obs_proj_h - dh

        # ----------- in PCAdbase get outside of +/- dh range index and set them to NaNs
        ix1 = np.argwhere((PCAdbase_proj_h_arr < proj_hmin) | \
                          (PCAdbase_proj_h_arr > proj_hmax))[:,0]
        indx_arr_ALL[ix1] = np.nan

        # ----------- constraint on number of models within height proj range
        non_NaN_dbase_models = nmodels_in_PCAdbase-ix1.size
        if non_NaN_dbase_models < ll_to_nmodels_restrct_hproj_search: # 10
            raise NameError('PAerr: Too small number of models within this height range.')      
    
    # --------------------------------------------------------
    if key_apr11_filter_dbase_search_below_NL==0: # NL = noise level
        obsIQUV_abv_noise_arr = np.ones(4) # 1 for signal above noise, 0 for below noise
                                       # to help remove Stokes IQUV (below noise) cont. to chisqr
                                       # if key_apr11_filter_dbase_search_below_NL=1
    else:
        obsIQUV_abv_noise_arr = np.ones(4)
        # this is a successor to "key_mar5_inc_obsIQUVnoise_in_chisqr_calc"
        # coded on Apr 18 2024
        # now do the following:
        # 1. compute noise threshold (thresh_NL) as per RCasnin proposal 3
        # 2. find obs IQUV L2norm of coeff and see which of IQUV is smaller than thresh_NL
        # 3. set obsIQUV_abv_noise_arr for that to 0.
        # 4. Go in PCAdbase, and set index for profile with PCA_dbase_C_L2n_all[k,:] > thresh_NL/2.
        #    this means, we will compute chisqr from only those profiles in dbase which have signal < thresh_NL/2.

        if 0 in nmodes_to_use_arr:
            raise NameError('PAerr: Can only either artifically calc chisqr for a certain IQUV where nmodes_to_use_arr!=0, or let key_apr11_filter_dbase_search_below_NL=1 determine which IQUV are above noise and use them in chisqr calc.')
            # Note: nmodes_to_use_arr = 0 only removes a given IQUV contribution to chisqr
            # while, key_apr11_filter_dbase_search_below_NL =1, removes IQUV contribution to chisqr and 
            # also filters those models in the database that are above the noise threshold limit.

        # -----------------
        # find which obs C stokes component is below/above the noise level
        # obs C L2norm of each Stokes component has to be larger than thresh_nl to ensure that it has signal in it.
        for k in range(4):
            # get noise level threshold based on number of modes and noise_level_wrt_obs_Imax
            # Following what RCasini said in his proposal.
            if nmodes_to_use_arr[k] <= 3:
                raise NameError('Beware: following formulation for thresh_nl doesnt apply for small nmodes (<3).')
                
            # --------------------------
            # above is Based on RCasini assessment that mean noise level in 
            # PCA IQUV coeff is given by noise_level * √(nC-0.5). 
            # PA still needs to confirm this.
            thresh_NL = 2 * np.sqrt(nmodes_to_use_arr[k] - 0.5) * noise_level_wrt_obs_Imax 
                    
            # ----------------------------
            # compute L2 norm of the coefficents, for a given Stokes
            tc_L2n = np.sqrt(np.nansum(obs_C_IQUV[:,k]**2)) # l2 norm of a given Stokes Coeff

            # ----------------------------
            # compare tc_L2n with thresh_NL, and donot include their contribution to chisqr if < thresh_NL
            # and also restrict chisqr calc. to only those models in the database 
            # with ||coeff|| <= thresh_NL/2

            if tc_L2n <= thresh_NL:
                obsIQUV_abv_noise_arr[k] = 0 # 0 = below noise; 0 removes this k's contribution from chisqr                
                # ----------------------------------------
                # implement RCasini proposal: to restrict_dbase_search_below_NL
                # for that k for which signal is below thresh_NL 
                # so, set index for those with signal > noise.
                ix1 = np.argwhere(PCA_dbase_C_L2n_all[:,k] > thresh_NL/2)[:,0]
                indx_arr_ALL[ix1] = np.nan
                # ----------------------------------------
                # PA: what i don't get is why when signal is above noise level
                # we are not similarly restricting databse search to those 
                # above the noise level.
    # ----------------------------------------
    # search in PCAdbase for only those locations for which indx_arr_ALL is not NaN
    # remove NaNs from indx_arr_ALL
    indx_arr_ALL = indx_arr_ALL[~np.isnan(indx_arr_ALL)].astype('int')

    # --------------------------------------------------------
    # chi_square calculation = Euclidean distance against each coeff in PCA_dbase_coeff_all
    # --------------------------------------------------------
    # compare Euclidean distances: between obs_C_IQUV and PCA_dbase_C_IQUV
    # errors made before Jan16 2024 calculations: 
        # 1. used resquared chisqr_wts
        # 2. used tsum += np.linalg.norm(chisqr_wts[k2] * tdiff**2, ord=2). 
        #      This is summing Root of Squared sum over IQUV, ∑_IQUV √ ∑(c-c_i)**2 while we want 
        #      summing of squared sums ∑_IQUV √ ∑(c-c_i)**2
    # ---------------
    # place holders
    chisqr_err_arr = np.zeros(indx_arr_ALL.size) + np.nan # stores total chisqr
    chisqr_err_wtd_arr_IQUV = np.zeros((indx_arr_ALL.size, 4)) + np.nan # indiv IQUV chisqr weighted err
    if key_use_IQUV_L2product_as_wts_for_chisqr ==1:
        chisqr_wts_arr_IQUV = np.zeros((indx_arr_ALL.size, 4)) + np.nan # indiv IQUV chisqr weighted err
    else:
        chisqr_wts_arr_IQUV = np.zeros((1, 4)) + np.nan

    # ---------------
    key_recalc_chisqr_wts=1 # helps implement obsIQUV_abv_noise_arr and nmodes_to_use_arr[k] to chisqr_wts_IQUV 
    chisqr_wts_IQUV = np.ones(4) # Initialize to 1s. Helpful when key_use_IQUV_weights_for_chisqr=0
                                  # do not put this inside PCA dbase loop as then chisqr wts get updated
                                  # to 1 unless key_recalc_chisqr_wts is always 1. 
    # if key_use_IQUV_weights_for_chisqr ==0:
    #     raise NameError('PAerr: unsure what to do if key_use_IQUV_weights_for_chisqr=0.')

    # ----------------
    # loop over non-NaN indx_arr_ALL PCA database, 
    # and calculate corresponding chisqr

    for k1 in range(indx_arr_ALL.size): 
        tk1 = indx_arr_ALL[k1] # location in PCA dbase
        k1_PCA_dbase_coeff_all = PCA_dbase_coeff_all[:,:,tk1]+0. # shape = (nmodes, 4, nprofiles in dbase)


        # --------------------------------------------------------
        # calc weights for chisqr calculation
        # --------------------------------------------------------   
        if key_recalc_chisqr_wts ==1:
            
            tPCA_coeff = np.zeros((1,1))+np.nan
            if key_use_IQUV_L2product_as_wts_for_chisqr ==1:
                tPCA_coeff = k1_PCA_dbase_coeff_all+0.
                tk=-1
                for tn in nmodes_to_use_arr:
                    tk+=1        
                    tPCA_coeff[tn:, tk] = np.nan # if tn=0, then all coeffs for that tk gets set to 0.


            if key_use_IQUV_weights_for_chisqr==1:
                # calc chisqr weights
                chisqr_wts_IQUV = pCMEx_get_IQUV_sqr_wts_for_chisqr_calc(
                            key_use_obsIQUV_ratio_as_wts_for_chisqr=key_use_obsIQUV_ratio_as_wts_for_chisqr, 
                            key_use_obsIQUV_L2n_as_wts_for_chisqr=key_use_obsIQUV_L2n_as_wts_for_chisqr, 
                            key_use_obsIQUV_L1n_as_wts_for_chisqr=key_use_obsIQUV_L1n_as_wts_for_chisqr, 
                            key_use_IQUV_L2product_as_wts_for_chisqr=key_use_IQUV_L2product_as_wts_for_chisqr,
                            obs_IQUV=obs_IQUV_rescaled, 
                            obs_C_IQUV = obs_C_IQUV,                    
                            # obs_C_IQUV_L2n = obs_C_IQUV_L2n,
                            pca_C_IQUV = tPCA_coeff, 
                            max_chisqr_wt_UnSq=max_chisqr_wt_UnSq, )  

            # --------------------------------------------------------
            # set chisqr_wts_IQUV to 0, for those Stokes depending on 
            #      - if key_apr11_filter_dbase_search_below_NL =1, set wts=0 for  obs profile < noise threshold
            #      - if key_apr11_filter_dbase_search_below_NL =0, set wts =0 for which nmodes_to_use_arr =0
            chisqr_wts_IQUV = chisqr_wts_IQUV*obsIQUV_abv_noise_arr*chisqr_wts_IQUV_fact0

            # for nmodes_to_use_arr=0, chisqr is calculated using a subset of IQUV
            for k in range(4):
                if nmodes_to_use_arr[k] ==0:
                    chisqr_wts_IQUV[k] = 0.
            if np.isnan(chisqr_wts_IQUV).any():
                raise NameError('PAerr: chisqr_wts_IQUV has NaNs in it. ')
            elif (chisqr_wts_IQUV < 0).any():
                raise NameError('PAerr: chisqr_wts_IQUV < 0.')

            if key_use_IQUV_L2product_as_wts_for_chisqr ==0:  
                key_recalc_chisqr_wts=0
                chisqr_wts_arr_IQUV[0,:]  = chisqr_wts_IQUV
            else:
                chisqr_wts_arr_IQUV[k1,:] = chisqr_wts_IQUV
                
        # --------------------------------- 
        # loop over IQUV and calc total chisqr (weighted or un-weighted)
        tchisqr_err = np.nan # reset

        for k in range(4): # IQUV loop
            if chisqr_wts_IQUV[k] == 0: # no contribution from this k
                continue

            t_nm = nmodes_to_use_arr[k] # nmodes to use

            if t_nm <=0:
                continue

            t_wtd_diff_sqr_sum = np.nansum((obs_C_IQUV[:t_nm,k] - k1_PCA_dbase_coeff_all[:t_nm,k])**2) * chisqr_wts_IQUV[k]
            chisqr_err_wtd_arr_IQUV[k1,k] = t_wtd_diff_sqr_sum # IQUV wtd chisqr err
                
            if t_wtd_diff_sqr_sum ==0: # would happen if obs_C_IQUV[:t_nm,k] are all NaNs or 0s
                continue
            elif np.isnan(tchisqr_err):
                tchisqr_err = t_wtd_diff_sqr_sum # wts needed in GM when key_use_IQUV_L2product_as_wts_for_chisqr=1
                continue

            if key_use_GM_for_chisqr ==1: # GM = Geometric mean
                tchisqr_err = tchisqr_err * t_wtd_diff_sqr_sum
            else:            
                tchisqr_err = tchisqr_err + t_wtd_diff_sqr_sum
            
        chisqr_err_arr[k1] = tchisqr_err+0. # store total chisqr err
        # if tchisqr_err == -1: # if -1, then NaN in chisqr_err_arr
        #     chisqr_err_arr[k1] = tchisqr_err+0. # store total chisqr err
        # else:
            # chisqr_err_arr[k1] = np.nan

    # print(np.nanmax(chisqr_err_arr), np.nanmin(chisqr_err_arr))
    # # print(t_wtd_diff_sqr_sum)
    # # print(tchisqr_err)
    # print(chisqr_wts_IQUV)

        
    # if key_search_using_PCAbarcodes==0:
    #     if np.isnan(chisqr_err_arr).any():
    #         raise NameError('PAerr: chisqr_err_arr has NaNs in it.')

    return indx_arr_ALL, chisqr_err_arr, \
            chisqr_err_wtd_arr_IQUV, chisqr_wts_arr_IQUV, obsIQUV_abv_noise_arr, \
            obs_C_IQUV, obs_IQUV_rescaled, NL_mask_IQUV, 


# --------------------------------------
# code status
@njit()
def pCMEx_apr26_calc_GM_chisqr_err_bn_C_IQUV_pair(C1_IQUV=None, C2_IQUV=None, 
                                                  chisqr_wts_IQUV=np.array([1.,1.,1.,1.]), 
                                                  nmodes_to_use_arr=np.array([10,10,10,10]) ):
    '''
    written on Ap 26 2024
    Goal to compute GM = Geometric mean distance between coeff vectors
        GM = (dC_I_L2n * wt_I) * (dC_Q_L2n*wt_Q)

        chisqr_wts_IQUV[k] = 0 for a given IQUV component means to not use it for 
        nmodes_to_use_arr[k] = number of modes to use for this IQUV component
        C1_IQUV = shape = (nmodes, 4)
            - same for C2_IQUV


    '''
    max_nmodes1 = C1_IQUV.shape[0]
    if (nmodes_to_use_arr > max_nmodes1).any():
        raise NameError('max nmodes is < nmodes_to_use_arr')

    GM_chisqr_err = 1.
    k1 = -1 # an indicator that we actually computed something
    for k in range(4): # IQUV loop
        nm1 = nmodes_to_use_arr[k] # nmodes to use
        if ((chisqr_wts_IQUV[k] == 0) | (nm1 <1)): # no contribution from this k, if you proceed, then GM would be 0
            continue

        k1 = 0 
        
        wtd_C_diff_L2n_sqr = np.nansum((C1_IQUV[:nm1,k] - C2_IQUV[:nm1,k])**2) * chisqr_wts_IQUV[k]
        GM_chisqr_err = GM_chisqr_err*wtd_C_diff_L2n_sqr

    if k1 == -1:
        return np.nan
    else:
        return GM_chisqr_err

# --------------------------------------
# code status
@njit()
def pCMEx_apr26_calc_AM_chisqr_err_bn_C_IQUV_pair(C1_IQUV=None, C2_IQUV=None, 
                                                  chisqr_wts_IQUV=None, 
                                                  nmodes_to_use_arr=None ):
    '''
    written on Ap 26 2024
    Goal to compute GM = Geometric mean distance between coeff vectors
        GM = (dC_I_L2n * wt_I) * (dC_Q_L2n*wt_Q)

        chisqr_wts_IQUV[k] = 0 for a given IQUV component means to not use it for 
        nmodes_to_use_arr[k] = number of modes to use for this IQUV component
        C1_IQUV = shape = (nmodes, 4)
            - same for C2_IQUV


    '''

    max_nmodes1 = C1_IQUV.shape[0]
    if (nmodes_to_use_arr > max_nmodes1).any():
        raise NameError('max nmodes is < nmodes_to_use_arr')

    AM_chisqr_err = 0 
    k1 = -1 

    for k in range(4): # IQUV loop
        nm1 = nmodes_to_use_arr[k] # nmodes to use
        if ((chisqr_wts_IQUV[k] == 0) | (nm1 <1)):
            continue

        k1 = 0 

        wtd_C_diff_L2n_sqr = np.nansum((C1_IQUV[:nm1,k] - C2_IQUV[:nm1,k])**2) * chisqr_wts_IQUV[k]
        AM_chisqr_err = AM_chisqr_err + wtd_C_diff_L2n_sqr

    if k1 == -1:
        return np.nan
    else:
        return AM_chisqr_err



# --------------------------------------
# code status - Green
@njit()
def pCMEx_apr19_calc_PCAdbase_C_L2n(PCA_dbase_coeff_all=None, nmodes_to_use_arr=None):
    '''
    Goal - to calc Coeff L2n norm for each coeff (for each model) in PCAbdase
           with dominant nmodes dictated by nmodes_to_use_arr
    Written on Apr 19 2024

    '''

    PCA_dbase_C_L2n_all=np.zeros((PCA_dbase_coeff_all.shape[2], 4))

    for k in range(4):
        t_nm = nmodes_to_use_arr[k]
        
        for k1 in range(PCA_dbase_coeff_all.shape[2]):
            PCA_dbase_C_L2n_all[k1,k] = np.linalg.norm(PCA_dbase_coeff_all[:t_nm,k,k1], ord=2)

    return PCA_dbase_C_L2n_all

# --------------------------------------
# code status - Green
@njit()
def pCMEx_apr19_calc_projected_h(h_1Darr=None, theta_in_deg_1Darr=None):
    '''
    Goal - to calc projected heights
    Written on Apr 19 2024
    '''
    pi180 = np.pi/180
    proj_h_arr = (1+h_1Darr) * np.sin(theta_in_deg_1Darr*np.pi/180) -1

    return proj_h_arr


# --------------------------------------
# @njit() - no need as such - leads to same time as no-njit..!!
def OLD_pCMEx_PCA_inv_1_profile_used_b4_Apr12_2024(
        dtime=None, # to get around Joblib cachce issue
        key_return_invmodel=0, 

        obs_IQUV=None,
        obs_model_2Darr=None, # needed to be saved in savefile

        # ----------------------- needed when using parallelized version
        k0=None, 
        obs_IQUV_all=None,                            
        obs_model_all=None, 
        # -----------------------

        PCA_eigenbasis_IQUV=None, 
        PCA_IQUV_mean0=None, 
        PCA_dbase_coeff_all=None, 
        PCA_dbase_model_all=None, 

        nmodes_to_use=-1, 
        key_normalize_obs_IQUV_to_Imax=1, 
        key_subtract_PCA_IQUV_mean0=1, 

        key_use_IQUV_weights_for_chisqr=0, # 0 = weights are all 1s
            # key_use_L2p_as_weights=0, 
            key_use_obsIQUV_ratio_as_wts_for_chisqr=0, 
            key_use_obsIQUV_L2n_as_wts_for_chisqr=0, 
            key_use_obsIQUV_L1n_as_wts_for_chisqr=0, 
            key_use_IQUV_L2product_as_wts_for_chisqr=0,
        max_chisqr_wt_UnSq=np.nan, # NaN = weights are unounded

        # noise_level_wrt_obs_Imax=np.nan, 

        # needed to save PCAinv result to file-----------------------
        key_save_inv_result_to_file=0,
        stats_savefold='./', 
        index_k=0, 

        key_use_IQUonly_for_chisqr_calc=0, 

        key_print_PAnotes=0, 

        # --------------------------------
        # Feb 21 - it seems like coeff less than noise level cause bad inversions
        # so check if discarding them from chisqr helps or not
        # key_feb21_discard_obs_C_IQUV_lt_noiselevel=0, 
        # obs_C_IQUV_threshold=0., 

        # ----------------- Feb 20 2024
        # temporary keys - to manually amplify SV 
        # key_manually_amplify_SV_weight=0, 
        # SV_wt_amp_factor=5.,

        # n1=7, 

        # Mar5  - if obsIQUV < noise , then do stuff
        key_mar5_inc_obsIQUVnoise_in_chisqr_calc=0, 
        key_mar5_est_IQUVnoise_with_all_wvlnths=1, 
        key_mar5_rmv_obsIQUV_if_below_noise=0,
        noise_level_wrt_obs_Imax=1e-3, 
        key_mar12_PCAdbase_search_below_noise_only=0, 
        # key_use_avg_obsS_as_noise_ul=0, 

        # -------------------------------
        key_mar15_restrict_hproj_search_inPCAdbase=0, 
        param_dh_range=2e-3, 

        # -------------------------------
        # reweight chisqr to address chisqr imbalance
        # key_mar10_reweight_chisqr=0, # discarded- reweighting to equalize chisqr contribution 
        # -------------------------------
        key_search_using_PCAbarcodes=0, 
        PCA_dbase_barcodes=np.zeros((2,2))+np.nan, 
        PCA_barcode_nmode=0, 
                            

                                ):

    '''
    Goal: invert 1 profile - and 
    
        
    Input:
        - max_chisqr_wt_UnSq
            - UnSq = un squared weights


    '''
    # def temp_func():
    #     return
    # void = temp_func() # does nothing


    if k0 is not None:
       obs_IQUV = obs_IQUV_all[:,:,k0] +0.
       index_k= k0
       obs_model_2Darr = obs_model_all[:,:,k0]
    # ----------------------------------------------

    # compute chisqr against each spectra in PCA database
    chisqr_err_arr, coeff_obs_IQUV, tobs_IQUV1, chisqr_err_arr_IQUV, \
        chisqr_wts_arr_IQUV, chisqr_wtd_err_arr_IQUV, IQUV_noise_or_not = \
            OLD_pCMEx_calc_chisqr_err_arr_for_1_obs_profile_used_b4_Apr12(obs_IQUV=obs_IQUV, 
                                                                                 
                PCA_eigenbasis_IQUV=PCA_eigenbasis_IQUV, 
                PCA_IQUV_mean0=PCA_IQUV_mean0, 
                PCA_dbase_coeff_all=PCA_dbase_coeff_all,

                nmodes_to_use=nmodes_to_use, 
                key_normalize_obs_IQUV_to_Imax=key_normalize_obs_IQUV_to_Imax, 
                key_subtract_PCA_IQUV_mean0=key_subtract_PCA_IQUV_mean0,

                key_use_IQUV_weights_for_chisqr=key_use_IQUV_weights_for_chisqr, 
                # key_use_L2p_as_weights=key_use_L2p_as_weights, 
                key_use_obsIQUV_ratio_as_wts_for_chisqr=key_use_obsIQUV_ratio_as_wts_for_chisqr, 
                key_use_obsIQUV_L2n_as_wts_for_chisqr=key_use_obsIQUV_L2n_as_wts_for_chisqr, 
                key_use_obsIQUV_L1n_as_wts_for_chisqr=key_use_obsIQUV_L1n_as_wts_for_chisqr, 
                key_use_IQUV_L2product_as_wts_for_chisqr=key_use_IQUV_L2product_as_wts_for_chisqr, 


                key_use_IQUonly_for_chisqr_calc=key_use_IQUonly_for_chisqr_calc, 
                max_chisqr_wt_UnSq=max_chisqr_wt_UnSq, 

                # key_feb21_discard_obs_C_IQUV_lt_noiselevel=key_feb21_discard_obs_C_IQUV_lt_noiselevel, 
                # obs_C_IQUV_threshold=obs_C_IQUV_threshold, 

                key_print_PAnotes=key_print_PAnotes, 

                # key_manually_amplify_SV_weight=key_manually_amplify_SV_weight, 
                # SV_wt_amp_factor=SV_wt_amp_factor,

                # Mar5  - if obsIQUV < noise , then do stuff
                key_mar5_inc_obsIQUVnoise_in_chisqr_calc=key_mar5_inc_obsIQUVnoise_in_chisqr_calc, 
                key_mar5_est_IQUVnoise_with_all_wvlnths=key_mar5_est_IQUVnoise_with_all_wvlnths, 
                key_mar5_rmv_obsIQUV_if_below_noise=key_mar5_rmv_obsIQUV_if_below_noise,
                noise_level_wrt_obs_Imax=noise_level_wrt_obs_Imax, 
                # key_mar12_PCAdbase_search_below_noise_only=key_mar12_PCAdbase_search_below_noise_only, 
                # key_use_avg_obsS_as_noise_ul=key_use_avg_obsS_as_noise_ul, 

                key_search_using_PCAbarcodes=key_search_using_PCAbarcodes, 
                PCA_dbase_barcodes=PCA_dbase_barcodes, 
                PCA_barcode_nmode=PCA_barcode_nmode, # barcode calc corresponding to how many modes

                key_mar15_restrict_hproj_search_inPCAdbase=key_mar15_restrict_hproj_search_inPCAdbase, 
                obs_model_2Darr=obs_model_2Darr, 
                PCA_dbase_model_all=PCA_dbase_model_all, 
                param_dh_range=param_dh_range, 
                 )   

    # time.sleep(1)

    # chisqr_mean_IQUV = np.zeros(4)+np.nan
    # if key_mar10_reweight_chisqr ==1:
    #     if key_use_IQUonly_for_chisqr_calc!=0:
    #         raise NameError('Need all 4 IQUV for this to work.')

    #     # i11 = np.argwhere(chisqr_err_arr == np.nanmin(chisqr_err_arr))[:,0]
    #     # print(chisqr_wtd_err_arr_IQUV[i11,:])
    #     # print(chisqr_wtd_err_arr_IQUV[i11,:]/np.sum(chisqr_wtd_err_arr_IQUV[i11,:])*100.)

    #     # print('>>', chisqr_err_arr[0])

        
    #     # i11 = np.argwhere(chisqr_err_arr == np.nanmin(chisqr_err_arr))[:,0][0]
    #     # print(chisqr_wtd_err_arr_IQUV[i11, :])

    #     # print(chisqr_wtd_err_arr_IQUV[10000,:])
    #     chisqr_mean_IQUV = np.mean(chisqr_wtd_err_arr_IQUV, axis=0)
    #     # chisqr_mean_IQUV = chisqr_mean_IQUV
    #     chisqr_wtd_err_arr_IQUV= chisqr_wtd_err_arr_IQUV/chisqr_mean_IQUV
    #     # print(chisqr_wtd_err_arr_IQUV[i11,:], '\n')
    #     # print(chisqr_mean_IQUV)

    #     # updated chisqr_err_arr


    #     chisqr_err_arr = np.sum(chisqr_wtd_err_arr_IQUV, axis=1)
    #     chisqr_wts_arr_IQUV = chisqr_wts_arr_IQUV/chisqr_mean_IQUV

    #     i11 = np.argwhere(chisqr_err_arr == np.nanmin(chisqr_err_arr))[:,0]
    #     # print(chisqr_wtd_err_arr_IQUV[i11,:]/np.sum(chisqr_wtd_err_arr_IQUV[i11,:])*100.)
    #     # print(chisqr_mean_IQUV)

    #     # print(chisqr_err_arr[0])



    # -----------------------------------------------
    if key_mar12_PCAdbase_search_below_noise_only==1:
        '''
        Mar12: Goal: if a given Stokes, say SV is noise dominatend, then we know that including its contribution in chisqr results in bad inversion. So, what we are doing here is removing SV contribution to chisqr. In addition, it makes sense to only look for profiles in the database with signal below the noise level (else SV wouldn't be noise-dominated on the 1st place). So, here, we only search those values in chisqr_err_arr, for which PCA dbase SV signal content is less than the noise level. At 1e-3 noise level, most of SV would be noise dominated. So, this would affect SU contrbutions to chisqr.
        '''

        if key_mar5_inc_obsIQUVnoise_in_chisqr_calc != 1:
            raise NameError('PAerr: key_mar12_PCAdbase_search_below_noise_only and key_mar5_inc_obsIQUVnoise_in_chisqr_calc both need to be 1.')

        if ((key_mar5_rmv_obsIQUV_if_below_noise != 1) & (key_mar5_est_IQUVnoise_with_all_wvlnths!=0)):
            raise NameError('PAerr: for key_mar12_PCAdbase_search_below_noise_only=1, chisqr calc should remove contribution from those obsIQUV which are dominated by noise + Noise upper limit should be based on line core region only (based on wavelengths for which PCAebasis is non-zero) and not the entire wavelength range.')

        chisqr_err_arr0 = chisqr_err_arr+0. # save original
        C_L2n_ul_arr = IQUV_noise_or_not[1,:] # noise upper limit in stokes based on sqrt(Nλ) * noise_level_wrt_Imax

        # if key_use_avg_obsS_as_noise_ul==0:
        #     C_L2n_ul_arr = IQUV_noise_or_not[1,:] # noise upper limit in stokes based on sqrt(Nλ) * noise_level_wrt_Imax
        # else:
        #     C_L2n_ul_arr = IQUV_noise_or_not[2,:] # avg signal content (No noise) |I_nonoise|**2 ~ (|I_noisy|**2 - |err|**2)
        # discarded |I_nonoise|**2 ~ (|I_noisy|**2 - |err|**2) as √(|I_noisy|**2 - |err|**2) could be a NaN

        # get common IQUV indx_arr in PCAdbase where Cnorm is < C_L2n_ul_arr
        # print(IQUV_noise_or_not)
        indx_arr_all, void = pCMEx_find_PCAdbase_profile_indx_below_an_upplim(
                                            PCA_dbase_coeff_all=PCA_dbase_coeff_all, 
                                              IQUV_noise_or_not=IQUV_noise_or_not[0,:], 
                                              noise_ul_arr=C_L2n_ul_arr, )

        # restrict the index where we are gonna find best chisqr
        chisqr_err_arr = chisqr_err_arr*indx_arr_all 


    # # -----------------------------------------------
    # # moved this to inside of "pCMEx_calc_chisqr_err_arr_for_1_obs_profile"
    # 
    # if key_mar15_restrict_hproj_search_inPCAdbase==1:
    #     '''
    #     See Mar 15 conversation with RCasini. He thinks that we can get around a lot of uncertainty in inversion result by restricting the search to +/- .002R of the obs_projected h

    #     # h_proj = (1 + h) sin(theta) - 1; Assumes h in proj units
    #     # projected height

    #     # Apr 2 - calculate chisqr for only those models that fall within the projH range.
    #     #         This should speed up calculation!

    #     '''

    #     pi180 = np.pi/180.

    #     obs_h = obs_model_2Darr[0,0]
    #     obs_theta = obs_model_2Darr[1,0]*pi180        
    #     obs_proj_h = (1+obs_h) * np.sin(obs_theta) -1 # projected height formula
    #     #---
    #     PCA_h = PCA_dbase_model_all[0,0,:]
    #     PCA_theta = PCA_dbase_model_all[1,0,:]*pi180
    #     PCA_proj_h = (1+PCA_h) * np.sin(PCA_theta) -1

    #     indx_arr_all = np.zeros(PCA_dbase_model_all.shape[2]) + np.nan

    #     dh = param_dh_range # in R units
    #     ix1 = np.argwhere((PCA_proj_h > obs_proj_h - dh) & \
    #                       (PCA_proj_h < obs_proj_h + dh))[:,0]
    #     if ix1.size > 0:
    #         if ix1.size < 1e2:
    #             raise NameError('?? too small number of models within this height range')
    #         indx_arr_all[ix1] = 1. 

    #         # use only those chisqr for which the h_proj within limits criteria is satisfied
    #         chisqr_err_arr = chisqr_err_arr*indx_arr_all

    # ----------------------------------------------
    key_apr7_chisqr_IQUVindx_below_noise_for_each=0
    if key_apr7_chisqr_IQUVindx_below_noise_for_each==1:
        nl_IQUV_threshold = 4* np.sqrt(nmodes_to_use - 0.5) * noise_level_wrt_obs_Imax # noise level threshold       

        for k in range(4):
            indx_arr_all = np.zeros(PCA_dbase_model_all.shape[2]) + np.nan
            ix_arr = np.argwhere(chisqr_err_arr_IQUV[:,k] < nl_IQUV_threshold)[:,0]
            # print('ix_arr.size=', ix_arr.size)
            indx_arr_all[ix_arr] = 1.

            chisqr_err_arr = chisqr_err_arr*indx_arr_all

        # print('.----->>>>', chisqr_err_arr[~np.isnan(chisqr_err_arr)].size, chisqr_err_arr.size)

    # if chisqr_err_arr[~np.isnan(chisqr_err_arr)].size < 2:
    #     raise NameError('All values in chisqr_err_arr are NaNs.')




    # ----------------------------------------------
    if np.isnan(np.nanmin(chisqr_err_arr)):
        print(chisqr_err_arr[~np.isnan(chisqr_err_arr)].size) # all values in chisqr are nans
        return 

    # chisqr_err_arr = chisqr_err_arr[~np.isnan(chisqr_err_arr)]

    # Get PCA database model with min chisqr_err (best fit spectra) - RCasini way
    i_bestmodel = np.argwhere(chisqr_err_arr == np.nanmin(chisqr_err_arr))[:,0]
    if i_bestmodel.size != 1:
        print(i_bestmodel, np.nanmax(chisqr_err_arr), np.nanmin(chisqr_err_arr))
        raise NameError('PAerr: multiple best fit models exist in PCA database.')

    i_bestmodel = i_bestmodel[0]
    inv_model_2Darr = PCA_dbase_model_all[:,:,i_bestmodel][:4,:] # best fit model in PCA database
    # ----------------------------------------------
    # convert inv_model_2Darr to 1D array. Will later think if it's better to store as dict or not    
    inv_model = pCMEx_RCasini_model_2Darr_to_1Darr(RCasini_model_2Darr=inv_model_2Darr[:4,:]) # 1D arr
    obs_model=None
    if obs_model_2Darr is not None:
        obs_model = pCMEx_RCasini_model_2Darr_to_1Darr(RCasini_model_2Darr=obs_model_2Darr) # 1D arr
    # ----------------------------------------------
    best_chisqr_err = chisqr_err_arr[i_bestmodel]
    best_chisqr_err_IQUV = chisqr_err_arr_IQUV[i_bestmodel, :]

    # print(best_chisqr_err)
    # print('>>>>>', i_bestmodel)

    if chisqr_wts_arr_IQUV.shape[0] > 1:
        best_chisqr_wts_IQUV = chisqr_wts_arr_IQUV[i_bestmodel, :].flatten() +0.
    else:
        best_chisqr_wts_IQUV = chisqr_wts_arr_IQUV.flatten()+0.
    # ----------------------------------------------
    # save stuff to file - needed when parallelizing
    if key_save_inv_result_to_file ==1:
        # save_fnm1 = '{}/{}_{}.pickle'.format(stats_savefold, prefix_savefnm, index_k)
        save_fnm1 = '{}/k={:05d}.pickle'.format(stats_savefold, index_k)

        # # check if save_fnm1 already exists - if yes, raise error
        # if os.path.isfile(save_fnm1):
        #     raise NameError('PAerr: \n {} \n already exists. Be careful not to overwrite something'.format(save_fnm1))

        # print(chisqr_wts_arr_IQUV.shape, chisqr_err_arr_IQUV.shape)

        tdict= dict(
                    obs_model=obs_model,    
                    inv_model=inv_model,                                        
                    best_chisqr_err=best_chisqr_err, # sum of weighted chisqr - for best fit
                    best_chisqr_err_IQUV=best_chisqr_err_IQUV, # indiv chisqr IQUV unweights
                    best_chisqr_wts_IQUV=best_chisqr_wts_IQUV, # weights used
                    coeff_obs_IQUV=coeff_obs_IQUV[:nmodes_to_use,:], 
                    coeff_inv_IQUV=PCA_dbase_coeff_all[:nmodes_to_use,:,i_bestmodel]

                    # # chisqr_mean_IQUV=chisqr_mean_IQUV, # mean IQUV chisqr over dbase model, 
                    # chisqr_wtd_err_arr_IQUV=chisqr_wtd_err_arr_IQUV, 
                    # IQUV_noise_or_not=IQUV_noise_or_not, 
                   )

        with open(save_fnm1, 'wb') as f:
            pickle.dump(tdict, f)
        tdict = None 

    # ----------------------------------------------
    if key_return_invmodel ==1:
        return inv_model, obs_model, \
               chisqr_err_arr, chisqr_err_arr_IQUV, \
               coeff_obs_IQUV, i_bestmodel        
    else: 
        inv_model, obs_model = None, None
        chisqr_err_arr, chisqr_err_arr_IQUV = None, None
        coeff_obs_IQUV, i_bestmodel = None, None

        return # None, None, None, None, None, None

# -------------------------------------------------------
# code status: green, triple checked on Jan 16 2024
@njit()
def OLD_pCMEx_calc_chisqr_err_arr_for_1_obs_profile_used_b4_Apr12(
            obs_IQUV=None, 

            PCA_eigenbasis_IQUV=None, 
            PCA_IQUV_mean0=None, 
            PCA_dbase_coeff_all=None, 

            nmodes_to_use=-1, 

            key_normalize_obs_IQUV_to_Imax=1, 
            key_subtract_PCA_IQUV_mean0=1, 

            key_use_IQUV_weights_for_chisqr=0, 
            # key_use_L2p_as_weights=0, 
            key_use_obsIQUV_ratio_as_wts_for_chisqr=0, 
            key_use_obsIQUV_L2n_as_wts_for_chisqr=0, 
            key_use_obsIQUV_L1n_as_wts_for_chisqr=0, 
            key_use_IQUV_L2product_as_wts_for_chisqr=0, 

            key_use_IQUonly_for_chisqr_calc=0, # No Stokes V in chisqr 
            max_chisqr_wt_UnSq=np.nan, # thresholding limit

            key_print_PAnotes=1,

            # --------------------------------
            # Feb 21 - it seems like coeff less than noise level cause bad inversions
            # so check if removing them from 
            # key_feb21_discard_obs_C_IQUV_lt_noiselevel=0, 
            # obs_C_IQUV_threshold=0., 
            # -----------------
            # temporary keys - to manually amplify SV 
            # key_manually_amplify_SV_weight=0, 
            # SV_wt_amp_factor=5.,

            # -----------------
            # Mar5  - if obsIQUV < noise , then do stuff
            key_mar5_inc_obsIQUVnoise_in_chisqr_calc=0, 
            key_mar5_est_IQUVnoise_with_all_wvlnths=1, 
            key_mar5_rmv_obsIQUV_if_below_noise=0,
            noise_level_wrt_obs_Imax=1e-3, 

            # -------------------------------
            key_search_using_PCAbarcodes=0, 
            PCA_dbase_barcodes=np.zeros((2,2))+np.nan, 
            PCA_barcode_nmode=0, 

            key_mar15_restrict_hproj_search_inPCAdbase=0,
            obs_model_2Darr=None, 
            PCA_dbase_model_all=None, 
            param_dh_range=None, 



                ):
    '''
    Written on Jan 11 2024
    Goal to do PCA inversion for a single profile and return chisqr_err_arr 
            (measure of how close are the eigen coeff for the obs_IQUV wrt each profile in the database)

    Input:
        - obs_IQUV: nlambda by 4 matrix, in IQUV order
        - PCA_eigenbasis_IQUV : (4 by nlambda by nmodes) matrix containing eigen modes of the PCA basis
        - PCA_IQUV_mean0: nlambda by 4 matrix containing mean value at each wavelength for each IQUV
                            - these are likely normalized to obs_Imax
        - PCA_dbase_coeff_all: nmodes by 4 by number of profiles in the database
                            - it is this database to which coeff for obs_IQUV would be compared to

        - nmodes_to_use = total number of modes to use when decomposing obs_IQUV onto PCA_eigenbasis_IQUV
            - -1 implies using all the modes in PCA_eigenbasis_IQUV

        - max_chisqr_wt_UnSq:np.nan, max chisqr weight for a given say Stokes V to be used to amplify its relative contribution to chisqr calculation
            - If np.nan - then weights are unbounded

        - keywords
            - key_normalize_obs_IQUV_to_Imax =1 to normalize all values in obs_IQUV with obs_Imax
            - key_subtract_IQUV_mean0 =1 to subtract PCA_IQUV_mean0 from obs_IQUV. A standard in PCA
            - key_use_IQUV_weights_for_chisqr =1 to use weighted chisqr
                - weights are calculated using pCMEx_get_IQUV_weights_for_chisqr_calc, which also
                    requires noise_level_wrt_obs_Imax. 
                    - noise_level_wrt_obs_Imax is more a factor. noise_level_wrt_obs_Imax*obs_Imax is the 
                      noise level in the obs_IQUV

            - key_print_PAnotes =1 to pring some PA notes

    Output:
        - chisqr_err_arr : Euclidean distance of coeff_obs_IQUV1 wrt each coeff in PCA_dbase_coeff_all
        - coeff_obs_IQUV1 : eigen coeff for obs_IQUV
        - obs_IQUV_normalized_PCAmean_subtracted: Note this could be normalized or not depending on if key_normalize_obs_IQUV_to_Imax=1 or not.

    - PA concerns:
        - does it make sense to use noise_level as upper limit in chisqr_wts calculation

    Steps:
        1) normalize obs_IQUV data by Imax
        2) subtract PCA_IQUV_mean0 from obs_IQUV
        3) decompose obs_IQUV1 on PCAbasis and calc obs_coeff_IQUV
        4) get chisqr weights for IQUV
        5) calc chisqr_err_arr: for obs_coeff_IQUV and coeff for each profile 
           in the PCA databbase weighted by respective chisqr weights
        6) return chisqr_err_arr

    Updates:
        [Jan 18 2024]: PCA_eigenbasis_IQUV shape changed to (4, nlambda, nmodes) from (nlambda, nmodes, 4).
                            - Reason: Prevents numba error: @ matrix multiplication is faster on 
                                      contiguous arrays.

    '''



    # --------------------------------------------------------
    # decompose observed IQUV into Coeff IQUV
    # obs_IQUV1 is normalized and has PCA_IQUV_mean0 subtracted from it
    # obs_C_IQUV = PCA Coeffs for observed IQUV
    obs_C_IQUV, obs_IQUV_normalized_PCAmean_subtracted = pCMEx_decomp_IQUV_to_PCA_coeff(IQUV=obs_IQUV,
                        PCA_eigenbasis_IQUV=PCA_eigenbasis_IQUV, 
                        key_normalize_obs_IQUV_to_Imax=key_normalize_obs_IQUV_to_Imax,
                        key_subtract_PCA_IQUV_mean0=key_subtract_PCA_IQUV_mean0, 
                        PCA_IQUV_mean0=PCA_IQUV_mean0, )
    # --------------------------------------------------------
    # use only up until :nmodes_to_use in inversion
    if nmodes_to_use ==-1:
        nmodes_to_use = PCA_eigenbasis_IQUV.shape[2] # shape = (4, nlambda, nmodes)
    elif ((nmodes_to_use > PCA_eigenbasis_IQUV.shape[2]) | (nmodes_to_use <= 0)):
        raise NameError('PAerror: nmodes_to_use is either > nmodes in PCA_eigenbasis_IQUV or is <= 0.')    
    obs_C_IQUV = obs_C_IQUV[:nmodes_to_use, :]
    # --------------------------------------------------------
    indx_arr_ALL = np.arange(PCA_dbase_coeff_all.shape[2])
    # --------------------------------------------------------
    # obs_C_barcode = -1
    if key_search_using_PCAbarcodes ==1:
        ''' 
        Apr 1 2024 work
        - Added this key to search only those dbase profiles for which barcode/index matches that of obs_profile
        where the dbase barcode matches with that of the obs_profile
        '''
        obs_C_barcode, barcode_binary = pCMEx_get_PCAcoeff_barcode_index(PCA_C_IQUV_2Darr=obs_C_IQUV, 
                                                                         max_nmodes=PCA_barcode_nmode)
        obs_C_barcode = obs_C_barcode[PCA_barcode_nmode-1]+0.

        # indx_arr_ALL1 = np.argwhere(PCA_dbase_barcodes == obs_C_barcode)[:,0]
        indx_arr_ALL = indx_arr_ALL[PCA_dbase_barcodes == obs_C_barcode]

        # print(PCA_dbase_barcodes[indx_arr_ALL].max(), PCA_dbase_barcodes[indx_arr_ALL].min(), obs_C_barcode)
        # print(obs_C_barcode, barcode_binary, indx_arr_ALL.size, obs_C_IQUV[:2,:])

        if indx_arr_ALL.size ==0:
            raise NameError('No model in PCAdbase with this obs_profile barcode.')
    # --------------------------------------------------------
    if key_mar15_restrict_hproj_search_inPCAdbase==1:
        '''
        See Mar 15 conversation with RCasini. He thinks that we can get around a lot of uncertainty in inversion result by restricting the search to +/- .002R of the obs_projected h

        # h_proj = (1 + h) sin(theta) - 1; Assumes h in proj units
        # projected height

        # Apr 2 - calculate chisqr for only those models that fall within the projH range.
        #         This should speed up calculation!

        '''

        pi180 = np.pi/180.

        obs_h = obs_model_2Darr[0,0]
        obs_theta = obs_model_2Darr[1,0]*pi180        
        obs_proj_h = (1+obs_h) * np.sin(obs_theta) -1 # projected height formula
        #---
        PCA_h = PCA_dbase_model_all[0,0,indx_arr_ALL]
        PCA_theta = PCA_dbase_model_all[1,0,indx_arr_ALL]*pi180
        PCA_proj_h = (1+PCA_h) * np.sin(PCA_theta) -1

        indx_arr_all1 = np.zeros(indx_arr_ALL.size) + np.nan

        dh = param_dh_range # in R units
        ix1 = np.argwhere((PCA_proj_h > obs_proj_h - dh) & \
                          (PCA_proj_h < obs_proj_h + dh))[:,0]
        if ix1.size > 0:
            if ix1.size < 1e2:
                raise NameError('?? too small number of models within this height range')
            indx_arr_all1[ix1] = 1. 

        indx_arr_ALL = indx_arr_ALL[~np.isnan(indx_arr_all1)] # cut out indx_arr_ALL which are within projected height range.
        # print('...........', indx_arr_ALL.size)

                                    
    # key_mar5_prioritize_chisqrwt_basedon_obsIQUV_gt_noiselevel=1
    # key_remove_obsIQUV_frm_chisqr_if_below_noise
    # param_wt_th_obsIQUV_below_noise_threshold
    # param_wt_th_based_on_obsC_L2n
    # param_wt_th_based_on_obsIQUV_L2n_linecore


    # Discard coefficients less than noise level. turn them into NaNs
    # key_feb28_discard_obs_IQUV_lt_nlevel

    # if key_feb21_discard_obs_C_IQUV_lt_noiselevel ==1:
    #     if key_use_obsIQUV_L2n_as_wts_for_chisqr != 1:
    #         raise NameError('PAerr: testing this key only in the presence of obs_L2n as weights key.')        
    #     for k1 in range(obs_C_IQUV.shape[0]):
    #         for k2 in range(obs_C_IQUV.shape[1]):
    #             if abs(obs_C_IQUV[k1,k2]) < obs_C_IQUV_threshold:
    #                 obs_C_IQUV[k1,k2] = np.nan


    # --------------------------------------------------------
    # print(obs_C_IQUV, '\n\n', np.cumsum(abs(obs_C_IQUV), axis=0)/np.sum(abs(obs_C_IQUV), axis=0)*100., '\n\n')
    # --------------------------------------------------------
    # calc weights for chisqr calculation
    # --------------------------------------------------------
    # chi_squared calculation 
    #     - compare Euclidean distances: between obs_C_IQUV and PCA_dbase_C_IQUV
    # errors made before Jan16 2024 calculations: 
        # 1. used resquared chisqr_wts
        # 2. used tsum += np.linalg.norm(chisqr_wts[k2] * tdiff**2, ord=2). 
        #      This is summing Root of Squared sum over IQUV, ∑_IQUV √ ∑(c-c_i)**2 while we want 
        #      summing of squared sums ∑_IQUV √ ∑(c-c_i)**2
        #                                                                 

    # calc chisqr Euclidean distance against each coeff in PCA database
    # PCA_dbase_coeff_all.shape = (nmodes, 4, nprofiles_in_database)
    # ----------------------------
    # if using IQU for chisqr_err calculation. (Ignore STokes V contribution)
    # Assumes 1st 3 k-index in obs_C_IQUV[:, k] 
    if key_use_IQUonly_for_chisqr_calc ==1: # IQU only
        tn1 = 3
    elif key_use_IQUonly_for_chisqr_calc ==2: # IQ only
        tn1 = 2
    elif key_use_IQUonly_for_chisqr_calc ==0:
        tn1 = 4 # IQUV all
    else:
        raise NameError()
    # ----------------------------
    # chisqr_err_arr size = number of PCA database models/profiles
    # chisqr for each coeff_arr in PCA database
    key_recalc_chisqr_wts = 1 # needed if key_use_IQUV_L2product_as_wts_for_chisqr =1 # geometric mean = 1
    chisqr_err_arr = np.zeros(PCA_dbase_coeff_all.shape[2]) + np.nan # place holder, 
    chisqr_err_arr_IQUV = np.zeros((chisqr_err_arr.size, 4)) + np.nan # indiv IQUV chisqr error place holder
    chisqr_wtd_err_arr_IQUV = np.zeros((chisqr_err_arr.size, 4)) + np.nan # weighted IQUV chisqr error place holder

    if key_use_IQUV_L2product_as_wts_for_chisqr==1:
        chisqr_wts_arr_IQUV = np.ones((chisqr_err_arr.size, 4)) + np.nan
    else:
        chisqr_wts_arr_IQUV = np.ones((1, 4)) + np.nan # (1,4) for Numba to not throw error
    chisqr_wts_IQUV = np.zeros(4) + 1. # Initialize to 1s. Helpful when key_use_IQUV_weights_for_chisqr=0
                                  # do not put this inside PCA dbase loop as then chisqr wts get updated
                                  # to 1 unless key_recalc_chisqr_wts is always 1. 
    # -------------------------------------------
    # get obs_C_IQUV_L2n   
    obs_C_IQUV_L2n = np.zeros(4) + np.nan # Initialize
    for k in range(4):    
        obs_C_IQUV_L2n[k] = np.sqrt(np.nansum(obs_C_IQUV[:,k]**2))
    # if key_use_IQUV_L2product_as_wts_for_chisqr ==1: # pre-compute obs_C_IQUV_L2n
    #     for k in range(4): # loop over IQUV and get C_L2n as numba doesn't support axis=0
    #         obs_C_IQUV_L2n[k] = np.sqrt(np.nansum(obs_C_IQUV[:,k]**2))
    #         # obs_C_IQUV_L2n[k] = np.linalg.norm(obs_C_IQUV[:,k], ord=2)
    # -------------------------------------------
    # -------------------------------------------
    # loop over each C_IQUV in PCA database and calc chisqr value agains obsC_IQUV
    # for k1 in range(chisqr_err_arr.size): # loop over database
    for k1 in indx_arr_ALL: # range(chisqr_err_arr.size): # loop over PCA database (or only dbase profiles with same obs barcode)
                                   # and calculate corresponding chisqr
        # if obs_C_barcode != -1:
        #     if PCA_dbase_barcodes[k1] != obs_C_barcode:
        #         continue

        t_PCA_dbase_C_IQUV = PCA_dbase_coeff_all[:nmodes_to_use,:,k1]+0. # shape = (nmodes, 4, nprofiles)
        # ---------------------------------
        # calc chisqr weights
        IQUV_noise_or_not = np.zeros((4,4))+ np.nan # initialize
        if ((key_use_IQUV_weights_for_chisqr ==1) & (key_recalc_chisqr_wts==1)): 
            chisqr_wts_IQUV = pCMEx_get_IQUV_sqr_wts_for_chisqr_calc(
                key_use_obsIQUV_ratio_as_wts_for_chisqr=key_use_obsIQUV_ratio_as_wts_for_chisqr, 
                key_use_obsIQUV_L2n_as_wts_for_chisqr=key_use_obsIQUV_L2n_as_wts_for_chisqr, 
                key_use_obsIQUV_L1n_as_wts_for_chisqr=key_use_obsIQUV_L1n_as_wts_for_chisqr, 
                key_use_IQUV_L2product_as_wts_for_chisqr=key_use_IQUV_L2product_as_wts_for_chisqr, 
                obs_IQUV=obs_IQUV_normalized_PCAmean_subtracted, 
                obs_C_IQUV = obs_C_IQUV, 
                obs_C_IQUV_L2n = obs_C_IQUV_L2n,
                pca_C_IQUV = t_PCA_dbase_C_IQUV, 
                max_chisqr_wt_UnSq=max_chisqr_wt_UnSq, )
            
            if np.isnan(chisqr_wts_IQUV).any():
                raise NameError('PAerr: chisqr_wts_IQUV has NaNs in it. ')
            # --------------------------------            
            # if key_manually_amplify_SV_weight ==1:
            #     chisqr_wts_IQUV[3] = chisqr_wts_IQUV[3] * (1+SV_wt_amp_factor)
            # raise NameError()
            # --------------------------------
            if key_mar5_inc_obsIQUVnoise_in_chisqr_calc ==1:
                # MAr 5 2024 work
                # identify if any of obs IQUV below noise level. If so then either
                #       - remove obs_C for that IQUV from chisqe 
                #       - or set obs_C = 0 so that min chisqr finds min dbase_C profile
                # - if IQUV < noise then
                # - key_mar5_rmv_obsIQUV_if_below_noise =1, the Stokes prof that is below now
                #     - remove IQUV completely from chisqr calc
                # - key_mar5_rmv_obsIQUV_if_below_noise=0
                #     - set obs_C_IQUV to 0 (the Stokes prof that is below now)
                #     - apply a threshold to chisqr_wts_IQUV based on noise*√(nmodes-0.5) : see Gchat notes on Feb 28

                # this will only work if: max_chisqr_wt_UnSq =NaN & 
                # key_use_obsIQUV_L2n_as_wts_for_chisqr =1
                if key_use_obsIQUV_L2n_as_wts_for_chisqr ==0:
                    raise NameError('PAerr: Can use key_mar5_inc_obsIQUVnoise_in_chisqr_calc=1 ONLY with key_use_obsIQUV_L2n_as_wts_for_chisqr=1.')
                elif ~np.isnan(max_chisqr_wt_UnSq):
                    raise NameError('PAerr: max_chisqr_wt_UnSq needs to be NaN for key_mar5_inc_obsIQUVnoise_in_chisqr_calc=1 to work.')

                # --------------------------
                # determine if/which obsIQUV is < noise
                # Note this uses obs IQUV to determine if signal above noise or not
                # and doesn't use obs_C_IQUV > [√(nC-0.5)] * noise_level_wrt_obs_Imax
                # as suggested by RCasini
                IQUV_noise_or_not, void = pCMEx_check_if_IQUV_above_noise_level(IQUV=obs_IQUV_normalized_PCAmean_subtracted, nmodes=nmodes_to_use, 
                    noise_level_wrt_Imax=noise_level_wrt_obs_Imax, 
                    PCA_eigenbasis_IQUV=PCA_eigenbasis_IQUV, 
                    key_mar5_est_IQUVnoise_with_all_wvlnths=key_mar5_est_IQUVnoise_with_all_wvlnths,
                    # PCA_nonzero_threshold=5e-3, 
                    )

                # for k9 in range(4): # Immaterial if key_mar5_rmv_obsIQUV_if_below_noise =1  
                #     obs_C_IQUV[:,k9] = obs_C_IQUV[:,k9] * IQUV_noise_or_not[0,k9]

                if key_mar5_rmv_obsIQUV_if_below_noise==1:
                    chisqr_wts_IQUV = chisqr_wts_IQUV * IQUV_noise_or_not[0,:] # set weights to 0
                    # No need to threshold weights for IQUV that are above noise level
                else:
                    tweight_threshold = obs_C_IQUV_L2n[0]/(1* (nmodes_to_use-0.5) * noise_level_wrt_obs_Imax**2)
                    # threshold chisqr weight
                    # print(tweight_threshold, chisqr_wts_IQUV)
                    for k9 in range(3, 4): 
                        if chisqr_wts_IQUV[k9] > tweight_threshold:
                            chisqr_wts_IQUV[k9] = tweight_threshold
            # --------------------------------
            if key_use_IQUV_L2product_as_wts_for_chisqr ==0:
                key_recalc_chisqr_wts=0 
                # chisqr_wts_IQUV only depend on obs_IQUV and not on t_PCA_dbase_C_IQUV
                # so no point in recalculating the weights
            # --------------------------------
            if key_recalc_chisqr_wts==1:
                chisqr_wts_arr_IQUV[k1, :] = chisqr_wts_IQUV+0. # store the weights
            else:
                chisqr_wts_arr_IQUV[0, :] = chisqr_wts_IQUV+0. # store the weights


        # --------------------------------- 
        # loop over IQUV and calc total chisqr (weighted or un-weighted)
        tchisqr_err =0
        for k2 in range(tn1): # IQUV loop
            tdiff = obs_C_IQUV[:,k2] - t_PCA_dbase_C_IQUV[:,k2]
            tdiff_sqr_sum = np.nansum(tdiff**2)  # Jan 17
            chisqr_err_arr_IQUV[k1,k2] = tdiff_sqr_sum # individual sums 
            chisqr_wtd_err_arr_IQUV[k1,k2] = tdiff_sqr_sum * chisqr_wts_IQUV[k2]
            # --------------------------------------
            # chisqr_wts_IQUV = all 1s, if key_use_IQUV_weights_for_chisqr=0
            # Note chisqr_wts_IQUV are already squared and are consistent with units of tdiff_sqr_sum
            tchisqr_err += tdiff_sqr_sum * chisqr_wts_IQUV[k2]
            
        chisqr_err_arr[k1] = tchisqr_err+0.
        # print(tchisqr_err)
        
    # if key_search_using_PCAbarcodes==0:
    #     if np.isnan(chisqr_err_arr).any():
    #         raise NameError('PAerr: chisqr_err_arr has NaNs in it.')

    return chisqr_err_arr, obs_C_IQUV, obs_IQUV_normalized_PCAmean_subtracted, \
           chisqr_err_arr_IQUV, chisqr_wts_arr_IQUV, chisqr_wtd_err_arr_IQUV, \
           IQUV_noise_or_not

# --------------------------------------
# CodeStatus: to be tested.
@njit()
def pCMEx_calc_proj_height(h_arr=None, theta_arr=None):
    '''
    written on Apr 15 2024
    Goal to calculate projected heights

    Input:
        theta_arr - Must be in radians
    '''

    raise NameError('Has not been tested yet.')

    proj_h_arr = (1+h_arr) * np.sin(theta_arr) -1 # projected height formula

    return proj_h_arr

# --------------------------------------
# Code status: green, documentation Complete
@njit()
def pCMEx_get_IQUV_sqr_wts_for_chisqr_calc(
                key_use_obsIQUV_ratio_as_wts_for_chisqr=0, # dI/dV
                key_use_obsIQUV_L2n_as_wts_for_chisqr=0,  # |obs_I_C|_2/|obs_V_C|_2
                key_use_obsIQUV_L1n_as_wts_for_chisqr=0,  # |obs_I_C|_1/|obs_V_C|_1
                key_use_IQUV_L2product_as_wts_for_chisqr=0, # 1/|obs_I_C|_2 * 1/|pca_I_C|_2

                obs_IQUV= np.zeros((1,1))+np.nan, # when key_use_obsIQUV_ratio_as_wts_for_chisqr=1
                obs_C_IQUV = np.zeros((1,1))+np.nan, # when key_use_obsIQUV_L2n_as_wts_for_chisqr=1
                max_chisqr_wt_UnSq=np.nan, # = np.nan => no thresholding

                # following when key_use_IQUV_L2product_as_wts_for_chisqr=1
                pca_C_IQUV = np.zeros((1,1))+np.nan, 
                obs_C_IQUV_L2n = np.array([np.nan]),  # precompute to save computing time
                ):
    '''
    written on Jan 10 2024
    Updated on Feb 19 2024

    Goal: Calc the weights for IQUV to be used in chisqr calc
          to put e.g ∑ (obs_coeff_Q - database_coeff_Q)**2 on the same amplitude as
                     ∑ (obs_coeff_I - database_coeff_I)**2 - so that the 

                     Goodness of fit is not dominated by say - I measurements only
                     Q has some say in it too.

    Input:
        - key_use_obsIQUV_ratio_as_wts_for_chisqr=0, # dI/dV as weights
            - only requires obs_IQUV and max_chisqr_wt_UnSq
            - mimics RCasini method: e.g. Q_weight = Imax/(Qmax-Qmin)

        - key_use_obsIQUV_L2n_as_wts_for_chisqr=0,  # |obs_I_C|/|obs_V_C|
            - obs_C_IQUV = PCA coeff of obs IQUV

        - key_use_IQUV_L2product_as_wts_for_chisqr=0, 
            - weights = 1/|obs_I_C| * 1/|pca_I_C|
            - requires obs_C_IQUV_L2n, 
                       pca_C_IQUV, 
                       obs_C_IQUV (if obs_C_IQUV_L2n is NaN)
                       max_chisqr_wt_UnSq


        - obs_IQUV = obs_IQUV: column1= SI, column 4 = Stokes V
            - Caution: this should be normalized to Imax=1, and have pca mean subtracted

        - obs_C_IQUV = PCA coeff of obs IQUV
        - pca_C_IQUV = PCA coeff of pca IQUV (the one in the database)

        - obs_C_IQUV_L2n - L2norm of obs_C_IQUV, 
            - 4-element array 

        - max_chisqr_wt_UnSq = maximum unsquared weights; UnSq = un squared
            if = np.nan - then weights are unbounded
    
    Output:
        chisqr_wts = 4 element array 
            - carrying weights for IQUV to be used 
            - with a ceiling of max_chisqr_wt_UnSq (if it is not NaN)
            - to be used as is in chisqr calculation - no squaring needed of the weights outputted 

    
    '''
    if key_use_obsIQUV_ratio_as_wts_for_chisqr + \
        key_use_obsIQUV_L2n_as_wts_for_chisqr  + \
        key_use_IQUV_L2product_as_wts_for_chisqr + \
        key_use_obsIQUV_L1n_as_wts_for_chisqr !=1:
        raise NameError('PAErr: pick only 1 approach to chisqr wt calculation.')
    # ----------------------------------------------
    wts_arr = np.zeros(4) + np.nan # place holder
    # ----------------------------------------------
    
    if key_use_obsIQUV_ratio_as_wts_for_chisqr ==1: # intensity ratio of obs_IQUV as weights e.g. dI/dV 
                                                    # Old: used by RCasini
        wts_arr[0] = 1. # intensity weight =1 
        obs_dI = np.max(obs_IQUV[:,0]) - np.min(obs_IQUV[:,0])

        for k in range(1,4): # loop over QUV and get weights
            tStokes_diff = np.max(obs_IQUV[:,k]) - np.min(obs_IQUV[:,k])   
            tweight = abs(obs_dI/tStokes_diff)
            wts_arr[k] = np.nanmin([tweight, max_chisqr_wt_UnSq]) # threshold wt, unless max_chisqr_wt_UnSq = NaN        
        return wts_arr**2. # <<<<<<<<<<< 

    # ----------------------------------------------
    elif key_use_obsIQUV_L2n_as_wts_for_chisqr==1: # I_wt = 1/L2n(C_I) # this makes the most sense to me (PA)
        for k in range(4): # loop over QUV
            tL2n = np.sqrt(np.nansum(obs_C_IQUV[:,k]**2)) # 0 if all nans in obs_C_IQUV[:,k]
            if tL2n ==0:
                tweight = 0. # else tweight -> infinity
            else:
                tweight = 1./tL2n                
            wts_arr[k] = np.nanmin([tweight, max_chisqr_wt_UnSq]) # threshold wt, unless max_chisqr_wt_UnSq = NaN
        return wts_arr**2 # squaring is needed here 

        # -------------- old code, used before May 6 2024
        # wts_arr[0] = 1. # intensity weight = 1
        # I_C_L2n = np.sqrt(np.nansum(obs_C_IQUV[:,0]**2)) # Intensity Coeff L2norm
        # # I_C_L2n = np.linalg.norm(obs_C_IQUV[:,0], ord=2) # Intensity Coeff L2norm

        # for k in range(1,4): # loop over QUV
        #     tL2n = np.sqrt(np.nansum(obs_C_IQUV[:,k]**2)) # 0 if all nans in obs_C_IQUV[:,k]
        #     if tL2n !=0:
        #         tweight = I_C_L2n/tL2n
        #     else:
        #         tweight = 0.
        #     # tweight = I_C_L2n/np.linalg.norm(obs_C_IQUV[:,k], ord=2)
        #     wts_arr[k] = np.nanmin([tweight, max_chisqr_wt_UnSq]) # threshold wt, unless max_chisqr_wt_UnSq = NaN
        # return wts_arr**2 # squaring is needed here 

    # ----------------------------------------------
    elif key_use_obsIQUV_L1n_as_wts_for_chisqr==1: # I_wt = 1/L2n(C_I) # this makes the most sense to me (PA)
        for k in range(4): # loop over QUV
            tL1n = np.nansum(np.abs(obs_C_IQUV[:,k]))
            if tL1n ==0:
                tweight =0 # else tweight -> infinity
            else:
                tweight = 1./tL1n
            wts_arr[k] = np.nanmin([tweight, max_chisqr_wt_UnSq]) # threshold wt, unless max_chisqr_wt_UnSq = NaN
        return wts_arr**2 # squaring is needed here 

        # -------------- old code, used before May 6 2024
        # wts_arr[0] = 1. # intensity weight = 1
        # I_C_L1n = np.nansum(np.abs(obs_C_IQUV[:,0])) # Intensity Coeff L1norm

        # for k in range(1,4): # loop over QUV
        #     tweight = I_C_L1n/np.nansum(np.abs(obs_C_IQUV[:,k]))
        #     wts_arr[k] = np.nanmin([tweight, max_chisqr_wt_UnSq]) # threshold wt, unless max_chisqr_wt_UnSq = NaN
        # return wts_arr**2 # squaring is needed here 

    # ----------------------------------------------
    elif key_use_IQUV_L2product_as_wts_for_chisqr==1: # this is what Roberto computed
        # raise NameError('PAerr (Apr18, 2024): Do not use this key.')
        if np.isnan(obs_C_IQUV_L2n[0]): # np.isnan(obs_C_IQUV_L2n).any():
            obs_C_IQUV_L2n = np.zeros(4) + np.nan # Initialize
            for k in range(4):
                obs_C_IQUV_L2n[k] = np.sqrt(np.nansum(obs_C_IQUV[:,k]**2)) 
                # obs_C_IQUV_L2n[k] = np.linalg.norm(obs_C_IQUV[:,k], ord=2)

        for k in range(4):  # loop over all IQUV; wt for chisqr_I is not 1
            pca_tC_L2n = np.sqrt(np.nansum(pca_C_IQUV[:,k]**2))
            # pca_tC_L2n = np.linalg.norm(pca_C_IQUV[:,k], ord=2)

            obs_twt = np.nanmin([1/obs_C_IQUV_L2n[k], max_chisqr_wt_UnSq]) # get obs wt and compare that to threshold
            pca_twt = np.nanmin([1/pca_tC_L2n, max_chisqr_wt_UnSq])
            wts_arr[k] = obs_twt*pca_twt

        return wts_arr # Note: Do not square: as wts_arr = 1/|obs_C|_2  * 1/|pca_C|_2 = product of L2 norms

    else:
        raise NameError('PAErr (Feb 19 2024): Select atleast 1 key to calc wts according to it.')

# --------------------------------------
# code status: Green
@njit()
def pCMEx_reconstruct_IQUV_from_coeff(PCA_eigenbasis_IQUV=None, 
                                      coeff_IQUV=None, 
                                      PCA_IQUV_mean0=None,                                        
                                      ):
    '''
    written on Jan 11 2024
    Goal: for a given coeff set, reconstruct Stokes IQUV i.e. U @ C = I
        - U = eigen basis
        - C = coefficient set
        - I = reconstructed stokes profiles 

    Input:
        - PCA_eigenbasis_IQUV = PCA eigen basis. 
            - shape: 4 by nlambda by nmodes
            - each column = 1 basis

        - coeff_IQUV = 
            - shape: nmodes by 4

        - PCA_IQUV_mean0 = mean IQUV (at each wavelength) subtracted from IQUV before 
                           computing the PCA eigen basis

    Output:
        reconstructed_IQUV with PCA_IQUV_mean0 added to the profiles

    [Jan 18 2024]: PCA_eigenbasis_IQUV shape changed to (4, nlambda, nmodes) from (nlambda, nmodes, 4).
                   

    '''
    ncmodes = coeff_IQUV.shape[0]
    reconstructed_IQUV = np.zeros((PCA_eigenbasis_IQUV.shape[1], 4)) + np.nan # shape: nlambda by 4 
    for j in range(4):
        reconstructed_IQUV[:,j] = PCA_eigenbasis_IQUV[j, :, :ncmodes] @ coeff_IQUV[:,j]

    reconstructed_IQUV = reconstructed_IQUV + PCA_IQUV_mean0 # add mean IQUV

    return reconstructed_IQUV

# ------------------------------------------------------
# code status: Seems to be working
@njit()
def pCMEx_decomp_IQUV_to_PCA_coeff(IQUV=None,
                                    PCA_eigenbasis_IQUV=None,    
                                    PCA_IQUV_mean0=None,                                   
                                    key_normalize_obs_IQUV_to_Imax=1,  
                                    key_subtract_PCA_IQUV_mean0=1, 
                                    ):
    '''
    written on Feb 10 2024
    This code follows RCasini's "decomp_Stokes.pro"

    Goal: to decompose Stokes IQUV into PCA coefficients

    Steps:
        - take IQUV
        - Normalize by Imax
        - subtract PCA mean IQUV
        - solve for c in U @ c = IQUV => c = U.T @ IQUV
        - return C

    Input:
        - IQUV: (nlambda, 4): contains Stokes IQUV to be decomposed
        - PCA_eigenbasis_IQUV: (4, nlambda, nmodes): PCA basis for each IQUV
        - PCA_IQUV_mean0: (nlambda. 4): PCA mean IQUV
        - key_normalize_obs_IQUV_to_Imax: 1 to normalize IQUV to Imax
        - key_subtract_IQUV_mean0:1 to subtract PCA_IQUV_mean0 before decomposing

    Output:
        C_IQUV (nmodes, 4): PCA coefficients for IQUV 
            - nmodes = number of modes in PCA_eigenbasis_IQUV
            - Note: IQUV[:,i]/Imax = PCA_eigenbasis_IQUV[i,:,:] @ C_IQUV[:,i] + PCA_IQUV_mean0

    '''
    # ----------------------------
    # normalize all obs_IQUV1 with Imax
    if key_normalize_obs_IQUV_to_Imax ==1:
        Imax0 = np.nanmax(np.abs(IQUV[:,0]))
        if Imax0 ==0:
            raise NameError('PAerr: Imax0 is 0.')
        IQUV = IQUV/Imax0
    else:
        raise NameError('PA: key_normalize_obs_IQUV_to_Imax should be 1.')
    # ----------------------------
    # subtract PCA_IQUV_mean0 from obs_IQUV1
    # PCA_eigenbasis_IQUV eigenbasis was created from a set of Stokes IQUV 
        # after subtracting PCA_IQUV_mean0
        # so, we need to do the same with obs_IQUV1 before decomposing it to eigen basis
    if key_subtract_PCA_IQUV_mean0 ==1:         
        IQUV = IQUV - PCA_IQUV_mean0 # subtracts corresponding IQUV profiles
    else:
        raise NameError('PA: key_subtract_IQUV_mean0 should be 1.')
    
    # ----------------------------
    # decompose IQUV on PCAbasis and calc coeff
    nmodes = PCA_eigenbasis_IQUV.shape[2] # PCA_eigenbasis_IQUV.shape = (4, nlambda, nmodes)
    C_IQUV = np.zeros((nmodes, 4)) + np.nan # placeholder    
    for j in range(4): # loop over IQUV
        C_IQUV[:,j] = np.dot(PCA_eigenbasis_IQUV[j,:,:].T, IQUV[:,j])
        # coeff_obs_IQUV1[:,j] = PCA_eigenbasis_IQUV[:,:nmodes_to_use,j].T @ obs_IQUV1[:,j] # reshaped post Jan 18 2024.
    return C_IQUV, IQUV # set of coefficient for a given Stokes type - for a given set of database profiles
                             # IQUV = normalized tp Imax, PCA mean IQUV subtracted

# ------------------------------------------------------
# code status: ??: Checked - U-values are similar to what Roberto sent me
@njit()
def pCMEx_calc_PCA_eigen_profiles(Stokes_2Darr=None, 
                                 
                                 key_normalize_by_respective_SImax=0, 
                                 key_normalize_by_SImax0=0, 

                                 SImax_arr=np.array([np.nan]), 
                                 SImax0=np.nan,

                                 key_PCA_calc_eigen_stuff_via_covar_matrix=0, 
                                 key_subtract_Stokes_avg_profile=1, 
                                 ):
    '''
    Written on Dec 29 2023- Jan 1 2024 
    Goal - To perform PCA on a given set of Stokes profiles (for only 1 kind say all are SI or SQ..). 
           This code follows: RCasini's "pca_Stokes.pro"

    Constraints: e.g. Stokes_2Darr: - each column = 1 complete Stokes profile (can be either of IQUV)
                               - each row stores values for different profile at a given wavelength

    Input:
        - Stokes_2Darr = 2D-array of Stokes I (or Q U V)
            - each column = 1 complete Stokes profile, 
            - each row has Stokes I for a given wavelength

        - Keywords:
            - key_normalize_by_respective_SImax = 1, needs SImax_arr
                to normalize each IQUV profile by its respective SI max

            - key_normalize_by_SImax0 =1, 
                to normalize each IQUV profile by a single number SImax0

            - key_PCA_calc_eigen_stuff_via_covar_matrix =1 
                - to calculate U-vector and singular values from covariance matrix: A@A.T, instead of directly
                - U-vectors are same (up until you go to super small singular values - WHY?)
                - Singular values of A is np.sqrt of the singular valus of A@A.T
            - key_subtract_Stokes_avg_profile =1 [default =1], subtract the mean value for each wavelength from each wavelength/profile

    Output:
        - u_SI, s_SI, Stokes_avg_profile

    '''
    # --------------------------------------
    # Normalize each Stokes profile by... 
    # ?? why though ?? 
    if key_normalize_by_SImax0 ==1: # by a common SImax0
        if np.isnan(SImax0):
            raise NameError('SImax0 is NaN.')
        Stokes_2Darr = Stokes_2Darr/SImax0
    elif key_normalize_by_respective_SImax ==1: # by respective profile's SImax
        # ------------------------
        # Divide each profile by respective SImax - that peak SI is 1 for each profile
        # But, you lose relative SImax difference doing this
        # Question - why do we want to do this one the 1st place?
        if np.isnan(SImax_arr[0]):# in None: 
            raise NameError('SImax_arr is None.')
        Stokes_2Darr = Stokes_2Darr/SImax_arr # each profile divided by it's respective SImax
                                              # Note QUV are also divided by SImax and not say SQmax

    # --------------------------------------
    # PCA default step: subtract the mean (at each lambda) from the profiles 
    Stokes_avg_profile = np.zeros(Stokes_2Darr.shape[0])+np.nan 
    if key_subtract_Stokes_avg_profile ==1:
        for k in range(Stokes_2Darr.shape[0]):
            Stokes_avg_profile[k] = np.mean(Stokes_2Darr[k,:])
            Stokes_2Darr[k,:] = Stokes_2Darr[k,:] - Stokes_avg_profile[k] # subtract mean (at kth lambda)      
                                                                       # from each indiv profile             
    else:
        raise NameError('PCA default is to subtract mean. Why not subtracting it?')

    # --------------------------------------
    # get covariance matrix = A@A.T 
    # the way matrix is ordered: A@A.T is shaped: nlambda by nlambda
    if key_PCA_calc_eigen_stuff_via_covar_matrix==1:
        Stokes_2Darr = np.dot(Stokes_2Darr, Stokes_2Darr.T) # A @ A.T

    # --------------------------------------
    # do SVD decomposition and get the eigen value and U-vector matrix
    u, s, void = np.linalg.svd(Stokes_2Darr, full_matrices=False) # by default gives UV matrix
    void= None

    return u, s, Stokes_avg_profile

# # ------------------------------------------------------
# # code status: Green: Checked - U-values are similar to what Roberto sent me
# # @njit()
# def DISCARD_pCMEx_calc_PCA_eigen_profiles(SI_2Darr=None, SQ_2Darr=None, 
#                                  SU_2Darr=None, SV_2Darr=None, 

#                                  key_normalize_by_respective_SImax=0, 
#                                  key_normalize_by_SImax0=0, SImax0=np.nan,

#                                  key_PCA_calc_eigen_stuff_via_covar_matrix=0, 

#                                  key_subtract_mean=1, 


#                                  ):
#     '''
#     Written on Dec 29 2023- Jan 1 2024 
#     Goal - to perform PCA on Stokes profiles. This code follows: RCasini's "pca_Stokes.pro"

#     Constraints: e.g. SI_2Darr: - each column = 1 complete Stokes profile, 
#                                - each row has Stokes I for a given wavelength

#     Input:
#         - SI_2Darr = 2D-array of Stokes I
#             - each column = 1 complete Stokes profile, 
#             - each row has Stokes I for a given wavelength

#             - Similar for Stokes QUV 2Darr

#         - Keywords:
#             - key_normalize_by_respective_SImax = 1 
#                 to normalize each IQUV profile by its Stokes I max

#             - key_normalize_by_SImax0 =1, 
#                 to normalize each IQUV profile by a single number SImax0

#             - key_PCA_calc_eigen_stuff_via_covar_matrix =1 
#                 - to calculate U-vector and singular values from covariance matrix: A@A.T, instead of directly
#                 - U-vectors are same (up until you go to super small singular values - WHY?)
#                 - Singular values of A is np.sqrt of the singular valus of A@A.T


#     Output:
#         - u_SI, s_SI, SImean_arr, u_SQ, s_SQ, SQmean_arr, \
#           u_SU, s_SU, SUmean_arr, u_SV, s_SV, SVmean_arr

#     Assumption:
#         - All Stokes IQUV are given and are of same shape



#     '''
#     # --------------------------------------
#     # divide each Stokes IQUV by SImax for that line's profile
#     # ?? why though ?? 
#     if key_normalize_by_SImax0 ==1:
#         if np.isnan(SImax0):
#             raise NameError('SImax0 is NaN.')

#         SI_2Darr = SI_2Darr/SImax0
#         SQ_2Darr = SQ_2Darr/SImax0 
#         SU_2Darr = SU_2Darr/SImax0 
#         SV_2Darr = SV_2Darr/SImax0 

#     elif key_normalize_by_respective_SImax ==1:
#         # ------------------------
#         # calculate SImax for each profile
#         SImax_arr = np.zeros(SI_2Darr.shape[1])+np.nan
#         for k in range(SI_2Darr.shape[1]):
#             SImax_arr[k] = np.max(SI_2Darr[:,k])
#         # ------------------------
#         # Divide each profile by SImax - that peak SI is 1 for each profile
#         # you lose relative SImax difference doing this
#         # Question - why do we want to do this one the 1st place?

#         SI_2Darr = SI_2Darr/SImax_arr # each column divided by it's max
#         SQ_2Darr = SQ_2Darr/SImax_arr
#         SU_2Darr = SU_2Darr/SImax_arr
#         SV_2Darr = SV_2Darr/SImax_arr

#     # --------------------------------------
#     # PCA default step: subtract the mean (at each lambda) from the profiles 
#     SImean_arr = np.zeros(SI_2Darr.shape[0])+np.nan 
#     SQmean_arr = SImean_arr + 0. 
#     SUmean_arr = SImean_arr + 0. 
#     SVmean_arr = SImean_arr + 0. 

#     if key_subtract_mean ==1:
#         for k in range(SI_2Darr.shape[0]):
#             SImean_arr[k] = np.mean(SI_2Darr[k,:])
#             SQmean_arr[k] = np.mean(SQ_2Darr[k,:])
#             SUmean_arr[k] = np.mean(SU_2Darr[k,:])
#             SVmean_arr[k] = np.mean(SV_2Darr[k,:])

#             SI_2Darr[k,:] = SI_2Darr[k,:] - SImean_arr[k] # subtract mean (at kth lambda) 
#             SQ_2Darr[k,:] = SQ_2Darr[k,:] - SQmean_arr[k] # from each indiv profile 
#             SU_2Darr[k,:] = SU_2Darr[k,:] - SUmean_arr[k]
#             SV_2Darr[k,:] = SV_2Darr[k,:] - SVmean_arr[k]
#     else:
#         raise NameError('PCA default is to subtract mean. Why not subtracting it?')

#     # --------------------------------------
#     # get covariance matrix = A@A.T 
#     # the way matrix is ordered: A@A.T is shaped: nlambda by nlambda
#     if key_PCA_calc_eigen_stuff_via_covar_matrix==1:
#         SI_2Darr = np.dot(SI_2Darr, SI_2Darr.T) # SI_2Darr @ SI_2Darr.T
#         SQ_2Darr = np.dot(SQ_2Darr, SQ_2Darr.T) # SQ_2Darr @ SQ_2Darr.T
#         SU_2Darr = np.dot(SU_2Darr, SU_2Darr.T) # SU_2Darr @ SU_2Darr.T
#         SV_2Darr = np.dot(SV_2Darr, SV_2Darr.T) # SV_2Darr @ SV_2Darr.T

#     # --------------------------------------
#     # do SVD decomposition and get the eigen value and U-vector matrix
#     u_SI, s_SI, void = np.linalg.svd(SI_2Darr, full_matrices=False) # by default gives UV matrix
#     u_SQ, s_SQ, void = np.linalg.svd(SQ_2Darr, full_matrices=False)
#     u_SU, s_SU, void = np.linalg.svd(SU_2Darr, full_matrices=False)
#     u_SV, s_SV, void = np.linalg.svd(SV_2Darr, full_matrices=False)
#     void= None

#     return u_SI, s_SI, SImean_arr, \
#            u_SQ, s_SQ, SQmean_arr, \
#            u_SU, s_SU, SUmean_arr, \
#            u_SV, s_SV, SVmean_arr


# ------------------------------------------------------
# code status: to be checked
def pCMEx_7jan24_read_RCasini_PCA_dbase_file(filename=None):
    '''
    written on Jan 7 2024
    Goal: to read Roberto Casini's PCA database file

    Input:
        - filename = string

    Ouput:
        - lambda_arr, SI_modes, SQ_modes, SU_modes, SV_modes, \
           IQUV_mean, IQUV_sval, lambda0

    '''
    # ---------------------------------------------

    # if nprofiles0 ==-1:
    #     nprofiles = 10001 # set to some super larger number 
    # else:
    #     nprofiles = nprofiles0
    # nmodes, dbase_size, 

    # -------------------------------------
    # get lambda0 and nlambda0 - from the header
    print('PA: there might be some inconsistency from 1 header to another in dbase file.')
    with open(filename, "r") as text_file:
        for tline in islice(text_file, None): # read all lines in file
            tdata = np.float64(np.array(tline.split())) 

            if tdata.size ==1: # jan 12 2024 - missing no. of profiles in header file
                if '500k' in filename:
                    dbase_size = 600000
                else:                    
                    dbase_size = 1500005
                print('PA: Be aware. Using {:d} for n_models in PCA database.'.format(dbase_size))
                ncoeffs = int(tdata)
            else:
                dbase_size = int(tdata[0])
                ncoeffs = int(tdata[1]) # number of coeffs for each IQUV
            break

            # j+=1
            # # if j==n:
            # #     lambda0 = np.float64(list(tline.split())[1]) # lambda0 (center of gravity)
            # if j==n+1:
            #     tdata = np.float64(np.array(tline.split())) 
            #     lambda0 = tdata[0] # lambda0 (center of gravity)
            #     nlambda0 = int(tdata[1]) # int(list(tline.split())[1]) # number of wavelength points
            #     break # break out of the loop
            
    # place holders
    tcoeff_arr0 = np.zeros((ncoeffs, 4)) + np.nan
    Stokes_coeff_all = np.zeros((ncoeffs, 4, dbase_size)) + np.nan # index is the model number
    Stokes_model_all = np.zeros((5, 5, dbase_size)) + np.nan # index is the model number
    # print('.......', dbase_size)

    # lambda_arr = np.zeros(nlambda0) + lambda0
    # tStokes_IQUV_arr0 = np.zeros((nlambda0,4)) # place holder for indiv IQUV profile 

    # Stokes_IQUV_all = np.zeros((nlambda0, 4, nprofiles)) + np.nan
    # Stokes_model_all = np.zeros((4, 5, nprofiles)) + np.nan
    # SQ_all = np.zeros((nlambda0, 4, nprofiles)) + np.nan
    # SU_all = np.zeros((nlambda0, 4, nprofiles)) + np.nan
    # SV_all = np.zeros((nlambda0, 4, nprofiles)) + np.nan
    # ---------------------------------------------
    n = 5 # counter - to read through header - get lambda0, nlambda
    k =-1 # counter for nth dbase
    m=-1
    # -------------------------------------
    # loop over and read the files
    with open(filename, "r") as text_file:
        for tline in islice(text_file, 1, None): # starting at model no., read all lines in file
            m+=1

            tdata = np.array(tline.split())
            tdata_size = tdata.size

            
            if tdata_size ==1: # the line only has the model number
                # if tline == '\n': # reset counters when new spectral line is encountered
                j=0  # counter: for all lines in a profile (including header)
                j1= -1 # counter: for lines that make a spectral profile 
                j2 = -1 # counter: for lines that make a spectral profile 
                tcoeff_arr = tcoeff_arr0+0.
                k+=1
            elif ((j>0) & (j<=n)): # store model parameters
                j2+=1
                # tdata = np.float64(np.array(tline.split()))
                Stokes_model_all[j2, :tdata_size, k] = tdata

            # elif j ==n:
            #     tlambda0 = np.float64(list(tline.split())[1])
            #     if tlambda0 != lambda0:
            #         raise NameError('lambda0 for current read line is not same as 1st line lambda0')
            # elif j==n+1:
            #     tlambda0 = np.float64(list(tline.split())[0])
            #     if tlambda0 != lambda0:
            #         raise NameError('lambda0 for current read line is not same as 1st line lambda0')
            #     if nlambda0 != int(list(tline.split())[1]):
            #         raise NameError('no. of lambda for current read line is not same as nlambda0')
            elif j > n:
                j1+=1

                try:
                    Stokes_coeff_all[j1, :, k] = tdata[:4]
                except:                
                    print('potential issue at: data no. {}, at line no. {}. tdata_size={}'.format(k, m+2, tdata_size))

                if j1 ==12:
                    break
                # if j1 == ncoeffs-1: # when reached at the end of this spectral profile

                #     # store the profile in respective matrix
                #     Stokes_coeff_all[:, :, k] = tcoeff_arr[:,:]


                    # --------------------------
                    # k+=1
                    # if k >= nprofiles: # break out if a given number of profiles is read
                    #     break
            #     j1+=1
            j+=1


    Stokes_coeff_all = Stokes_coeff_all[:,:,:k+1]
    Stokes_model_all = Stokes_model_all[:,:,:k+1]

    print(Stokes_coeff_all.shape)
    print(Stokes_model_all.shape)

    return Stokes_coeff_all, Stokes_model_all # SI_all, SQ_all, SU_all, SV_all

# ------------------------------------------------------
# code status: to be checked
def pCMEx_7jan24_read_RCasini_PCA_basis_file(filename=None, key_plot_IQUV_modes=0):
    '''
    written on Jan 7 2024
    Goal: to read Roberto Casini's PCA basis file

    Input:
        - filename = string

    Ouput:
        - lambda_arr, SI_modes, SQ_modes, SU_modes, SV_modes, \
           IQUV_mean, IQUV_sval, lambda0

    '''
    # ---------------------------------------------
    # Get lambda_arr
    j=-1
    k=-1 # loop over rows and collect wavelength info
    with open(filename, "r") as text_file:
        for tline in islice(text_file, None): # read all lines in file
            j+=1
            if j==0:
                nlambda = int(list(tline.split())[0]) # number of wavelength points
                nmodes = int(list(tline.split())[1])# number of modes in the file (excluding mean)
                lambda_arr = np.zeros(nlambda) + np.nan # initialize
            elif j==1:
                lambda0 = np.float64(tline) # central wavelength
            elif j>1:
                k+=1
                lambda_arr[k] = lambda0 + np.float64(tline)
            if j ==nlambda+1:
                break

    # ---------------------------------------------
    # place holders
    IQUV_mean = np.zeros((nlambda,4)) + np.nan # mean array
    IQUV_sval = np.zeros((nmodes,4)) + np.nan # singular values array
    IQUV_modes = np.zeros((nlambda, nmodes, 4)) + np.nan # [Jan 18 2024] Now, later reshaped as (4, nlambda, nmodes). 
                                                         # this helps with matrix inner product later on

    # SI_modes = np.zeros((nlambda, nmodes)) + np.nan
    # SQ_modes = np.zeros((nlambda, nmodes)) + np.nan
    # SU_modes = np.zeros((nlambda, nmodes)) + np.nan
    # SV_modes = np.zeros((nlambda, nmodes)) + np.nan

    # ------------------------------------
    j = -1  # counter for each row in the file
    im = -1 # counter for a given mode
    with open(filename, "r") as text_file:
        for tline in islice(text_file, None):
            j+=1

            if j<nlambda+2: # up until here, it's all lambda_arr info
                continue
            
            # reset the loop param + increase counter
            if (j-1) % (nlambda+1) ==0: 
                il = -1 # reinitialize, counter for stuff within a mode
                im += 1 
                if im>0: # store singular values
                    IQUV_sval[im-1] = np.array((tline.split()))
                continue
                
            il+=1
            if im==0: # store the mean IQUV
                IQUV_mean[il,:] = np.array((tline.split()))
            else: # store IQUV modes
                tarr = np.array((tline.split()))
                IQUV_modes[il, im-1, 0] = tarr[0]
                IQUV_modes[il, im-1, 1] = tarr[1]
                IQUV_modes[il, im-1, 2] = tarr[2]
                IQUV_modes[il, im-1, 3] = tarr[3]

                # SI_modes[il,im-1] = tarr[0]
                # SQ_modes[il,im-1] = tarr[1]
                # SU_modes[il,im-1] = tarr[2]
                # SV_modes[il,im-1] = tarr[3]

    if key_plot_IQUV_modes==1:
        linewidth= 1
        fig, axx =plt.subplots(1,4, figsize=(15,1.5))
        axx[0].plot(lambda_arr, IQUV_mean[:,0], linewidth=linewidth)
        axx[1].plot(lambda_arr, IQUV_mean[:,1], linewidth=linewidth)
        axx[2].plot(lambda_arr, IQUV_mean[:,2], linewidth=linewidth)
        axx[3].plot(lambda_arr, IQUV_mean[:,3], linewidth=linewidth)

        for i in range(nmodes):
            fig, axx =plt.subplots(1,4, figsize=(15,1.5))
            axx[0].plot(lambda_arr, IQUV_modes[:,i,0], linewidth=linewidth)
            axx[1].plot(lambda_arr, IQUV_modes[:,i,1], linewidth=linewidth)
            axx[2].plot(lambda_arr, IQUV_modes[:,i,2], linewidth=linewidth)
            axx[3].plot(lambda_arr, IQUV_modes[:,i,3], linewidth=linewidth)

            # axx[0].plot(lambda_arr, SI_modes[:,i], linewidth=linewidth)
            # axx[1].plot(lambda_arr, SQ_modes[:,i], linewidth=linewidth)
            # axx[2].plot(lambda_arr, SU_modes[:,i], linewidth=linewidth)
            # axx[3].plot(lambda_arr, SV_modes[:,i], linewidth=linewidth)

    if np.isnan(IQUV_modes).any():
        raise NameError('PAerror: values in the modes_2Darr is NaN. Check it.')
    if np.isnan(IQUV_mean).any():
        raise NameError('PAerror: values in the IQUV_mean is NaN. Check it.')
    if np.isnan(IQUV_sval).any():
        raise NameError('PAerror: values in the IQUV_sval is NaN. Check it.')
    if np.isnan(lambda_arr).any():
        raise NameError('PAerror: values in the lambda_arr is NaN. Check it.')


    # ----------------------
    # Jan 18 2024 - reshaping IQUV_modes to (4,nlambda,nmodes). 
    # This helps with inner produce calculation down the line.
    IQUV_modes1 = np.zeros((4, IQUV_modes.shape[0], IQUV_modes.shape[1])) + np.nan

    for k in range(4):
        IQUV_modes1[k,:,:] = IQUV_modes[:,:,k]
    IQUV_modes= IQUV_modes1+0.
    IQUV_modes1=None # delete var
    # ----------------------


    return lambda_arr, IQUV_modes, IQUV_mean, IQUV_sval, lambda0

    # return lambda_arr, SI_modes, SQ_modes, SU_modes, SV_modes, \
    #        IQUV_mean, IQUV_sval, lambda0

# ------------------------------------------------------
# code status: to be checked
def pCMEx_7jan24_read_RCasini_LHS_profs_file(filename=None, nprofiles0=-1, 
                                            ):
    '''
    written on Jan7 2024
    Goal- to read the profiles file created by Roberto Casini
        Copied from 'pCMEx_read_RCasini_spectralprofiles_file'

    Input: 
        - filename (string)
        - nprofiles0 = determines how many profiles (starting from 1st line)
            to return. If -1, then all profiles in the file are returned

    Output:
        - lambda_arr, SI_all, SQ_all, SU_all, SV_all

    Assumptions:
        lambda_arr - is the same for all the profiles in the file
            - lambda_arr corresponding to the 1st profile in the file is returned

    Note:
        - line in this code = line number in the file (not a spectral line)
    '''
    if nprofiles0 ==-1:
        nprofiles = 30001 # set to some super larger number 
    else:
        nprofiles = nprofiles0

    # -------------------------------------
    n = 5 # counter - to read through header - get lambda0, nlambda
    # -------------------------------------
    # get lambda0 and nlambda0 - from the header
    j=0
    with open(filename, "r") as text_file:
        for tline in islice(text_file, None): # read all lines in file
            # if j==n:
            #     lambda0 = np.float64(list(tline.split())[1]) # lambda0 (center of gravity)
            if j==n+1:
                tdata = np.float64(np.array(tline.split())) 
                lambda0 = tdata[0] # lambda0 (center of gravity)
                nlambda0 = int(tdata[1]) # int(list(tline.split())[1]) # number of wavelength points
                break # break out of the loop
            j+=1
    # ---------------------------------------------
    # place holders
    lambda_arr = np.zeros(nlambda0) + lambda0
    tStokes_IQUV_arr0 = np.zeros((nlambda0,4)) # place holder for indiv IQUV profile 

    Stokes_IQUV_all = np.zeros((nlambda0, 4, nprofiles)) + np.nan
    Stokes_model_all = np.zeros((4, 5, nprofiles)) + np.nan
    # SQ_all = np.zeros((nlambda0, 4, nprofiles)) + np.nan
    # SU_all = np.zeros((nlambda0, 4, nprofiles)) + np.nan
    # SV_all = np.zeros((nlambda0, 4, nprofiles)) + np.nan
    # ---------------------------------------------
    # loop over and read the files
    k =0 # counter for number of profiles read
    with open(filename, "r") as text_file:
        for tline in islice(text_file, None): # read all lines in file
            if tline == '\n': # reset counters when new spectral line is encountered
                j=0  # counter: for all details for a given profile (including header)
                j1=0 # counter: for lines that make a spectral profile 
                tStokes_IQUV_arr = tStokes_IQUV_arr0+0.
            elif ((j>1) & (j<=n)): # store model parameters
                tdata = np.float64(np.array(tline.split()))
                Stokes_model_all[j-2, :len(tdata), k] = tdata

            # elif j ==n:
            #     tlambda0 = np.float64(list(tline.split())[1])
            #     if tlambda0 != lambda0:
            #         raise NameError('lambda0 for current read line is not same as 1st line lambda0')
            elif j==n+1:
                tlambda0 = np.float64(list(tline.split())[0])
                if tlambda0 != lambda0:
                    raise NameError('lambda0 for current read line is not same as 1st line lambda0')
                if nlambda0 != int(list(tline.split())[1]):
                    raise NameError('no. of lambda for current read line is not same as nlambda0')
            elif j > n+1:
                ttline = list(tline.split())
                
                if k==0:
                    lambda_arr[j1] = lambda_arr[j1] + np.float64(ttline[0])
                tStokes_IQUV_arr[j1, :] = ttline[1:5]
                
                if j1 == nlambda0-1: # when reached at the end of this spectral profile

                    # store the profile in respective matrix
                    Stokes_IQUV_all[:, :, k] = tStokes_IQUV_arr[:,:]
                    # Stokes_IQUV_all[:, 1, k] = tStokes_IQUV_arr[:,1]
                    # Stokes_IQUV_all[:, 2, k] = tStokes_IQUV_arr[:,2]
                    # Stokes_IQUV_all[:, 3, k] = tStokes_IQUV_arr[:,3]

                    # fig, ax = plt.subplots(figsize=(10,2))
                    # ax.plot(lambda_arr, SI_all[:,k], 'r.-')
                    # fig, ax = plt.subplots(figsize=(10,2))
                    # ax.plot(lambda_arr, SQ_all[:,k], 'r.-')
                    # fig, ax = plt.subplots(figsize=(10,2))
                    # ax.plot(lambda_arr, SU_all[:,k], 'r.-')
                    # fig, ax = plt.subplots(figsize=(10,2))
                    # ax.plot(lambda_arr, SV_all[:,k], 'r.-')
                    # plt.show()

                    # --------------------------
                    k+=1
                    if k >= nprofiles: # break out if a given number of profiles is read
                        break
                j1+=1
            j+=1

    if nprofiles0 ==-1: # cut out the extra piece
        # SI_all = SI_all[:,:k]
        # SQ_all = SQ_all[:,:k]
        # SU_all = SU_all[:,:k]
        # SV_all = SV_all[:,:k]
        Stokes_IQUV_all = Stokes_IQUV_all[:,:,:k]
        Stokes_model_all = Stokes_model_all[:,:,:k]

    return lambda_arr, Stokes_IQUV_all, Stokes_model_all # SI_all, SQ_all, SU_all, SV_all

# ------------------------------------------------------
# code status: checked: the spectral lines in the file are read correctly 
def pCMEx_read_RCasini_spectralprofiles_file(filename=None, nprofiles0=-1, 
                                            ):
    '''
    Goal- to read the profiles file created by Roberto Casini
        Copied from 'Dec12_Roberto_Spectra_Gsigma_histogram.ipynb'

    Input: 
        - filename (string)
        - nprofiles0 = determines how many profiles (starting from 1st line)
            to return. If -1, then all profiles in the file are returned

    Output:
        - lambda_arr, SI_all, SQ_all, SU_all, SV_all

    Assumptions:
        lambda_arr - is the same for all the profiles in the file
            - lambda_arr corresponding to the 1st profile in the file is returned

    Note:
        - line in this code = line number in the file (not a spectral line)
    '''
    if nprofiles0 ==-1:
        nprofiles = 100001 # set to some super larger number 
    else:
        nprofiles = nprofiles0

    # -------------------------------------
    n = 5 # counter - to read through header - get lambda0, nlambda
    # -------------------------------------
    # get lambda0 and nlambda0 - from the header
    j=0
    with open(filename, "r") as text_file:
        for tline in islice(text_file, None): # read all lines in file
            if j==n:
                lambda0 = np.float64(list(tline.split())[1]) # lambda0 (center of gravity)
            elif j==n+1:
                nlambda0 = int(list(tline.split())[1]) # number of wavelength points
                break # break out of the loop
            j+=1
    # ---------------------------------------------
    # place holders
    lambda_arr = np.zeros(nlambda0) + lambda0
    Stokes_IQUV_arr0 = np.zeros((nlambda0,4))

    SI_all = np.zeros((nlambda0, nprofiles)) + np.nan
    SQ_all = np.zeros((nlambda0, nprofiles)) + np.nan
    SU_all = np.zeros((nlambda0, nprofiles)) + np.nan
    SV_all = np.zeros((nlambda0, nprofiles)) + np.nan
    # ---------------------------------------------
    # loop over and read the files
    k =0 # counter for number of profiles read
    with open(filename, "r") as text_file:
        for tline in islice(text_file, None): # read all lines in file
            if tline == '\n': # reset counters when new spectral line is encountered
                j=0  # counter: for all lines in a profile (including header)
                j1=0 # counter: for lines that make a spectral profile 
                tStokes_IQUV_arr = Stokes_IQUV_arr0+0.
            elif j ==n:
                tlambda0 = np.float64(list(tline.split())[1])
                if tlambda0 != lambda0:
                    raise NameError('lambda0 for current read line is not same as 1st line lambda0')
            elif j==n+1:
                if nlambda0 != int(list(tline.split())[1]):
                    raise NameError('no. of lambda for current read line is not same as nlambda0')
            elif j > n+1:
                ttline = list(tline.split())
                
                if k==0:
                    lambda_arr[j1] = lambda_arr[j1] + np.float64(ttline[0])
                tStokes_IQUV_arr[j1, :] = ttline[1:5]
                
                if j1 == nlambda0-1: # when reached at the end of this spectral profile

                    # store the profile in respective matrix
                    SI_all[:,k] = tStokes_IQUV_arr[:,0]
                    SQ_all[:,k] = tStokes_IQUV_arr[:,1]
                    SU_all[:,k] = tStokes_IQUV_arr[:,2]
                    SV_all[:,k] = tStokes_IQUV_arr[:,3]

                    # fig, ax = plt.subplots(figsize=(10,2))
                    # ax.plot(lambda_arr, SI_all[:,k], 'r.-')
                    # fig, ax = plt.subplots(figsize=(10,2))
                    # ax.plot(lambda_arr, SQ_all[:,k], 'r.-')
                    # fig, ax = plt.subplots(figsize=(10,2))
                    # ax.plot(lambda_arr, SU_all[:,k], 'r.-')
                    # fig, ax = plt.subplots(figsize=(10,2))
                    # ax.plot(lambda_arr, SV_all[:,k], 'r.-')
                    # plt.show()

                    # --------------------------
                    k+=1
                    if k >= nprofiles: # break out if a given number of profiles is read
                        break
                j1+=1
            j+=1

    if nprofiles0 ==-1: # cut out the extra piece
        SI_all = SI_all[:,:k]
        SQ_all = SQ_all[:,:k]
        SU_all = SU_all[:,:k]
        SV_all = SV_all[:,:k]

    return lambda_arr, SI_all, SQ_all, SU_all, SV_all
# ------------------------------------------------------
# code - to be checked
def pCMEx_extract_IRIS_MgIIk_SIdata(lambda_arr=None, SI_arr=None, 
                                 MgIIk_lambda_ll=2794, MgIIk_lambda_ul=2798, 
                                 MgIIk_SImax_ll=30.,
                                 param_dlambda_ul=.25, MgIIk_SImean_ul=20.,
                                 key_print_error_message=0, 



                                 ):
    '''
    writen on Dec 16 2023
    Goal - to extract MgIIk data from SI_arr
        - reject the data if prominence is super noisy (based on MgIIk SImax < MgIIk_SImax_ll)
        - pixel is inside the limb (based on SImean (near MgIIk core) > MgIIk_SImean_ul)
            - the mean region is identified using param_lambda_ul

    Input:
        - lambda_arr, SI_arr = MgII h (red) and k(blue) wavelength and SI data
        - MgIIk_lambda_ll, ul = lower limit and upper limit to wavelengths that identify MgIIk line
        - MgIIk_SImax_ll = lower limit to SImax - to identify noisy pixels (too far away from prominence)
        - param_lambda_ul - wavelength upper limit to identiy wavelengths smaller than left wing of MgIIk line
            - idea is - for pixels that are within solar limb are in absorption and have too large 
                        values for SI. So, the SImean would likely be a large value - as compared to when the 
                        the pixel is on prominence (outside solar limb), where the line is in emission for which
                        SImean for these wavelengths ~ 0. So, checking this SImean with MgIIk_SImean_ul and see
                        if the pixel is on the prominence or on the solar limb

    Return
        MgIIk - lambda_Arr and SI_arrr

    '''

    
    xdata0 = lambda_arr
    ydata0 = SI_arr
    lambda0 = MgIIk_lambda_ll
    lambda1 = MgIIk_lambda_ul

    # cut out MgIIk line 
    ydata0 = ydata0[((xdata0 >= lambda0) & (xdata0 <= lambda1))]
    xdata0 = xdata0[((xdata0 >= lambda0) & (xdata0 <= lambda1))]
    ixdata0 = np.arange(xdata0.size)

    param_lambda_ul = MgIIk_lambda_ll + param_dlambda_ul

    # --------------------------------
    # rejection criteria for this xydata0 

    # Criteria 1: point is not on the prominence; super noisy data - where you barely have any Mg II k signal
    if np.max(abs(ydata0)) < MgIIk_SImax_ll:
        if key_print_error_message ==1:
            print('SI_arr is in the noisy region, SI_max = {:.2f}'.format(np.max(ydata0)))
        return None, None

    # Criteria 2: ensure that ydata0 does not correspond to inside limb pixel
    # here mean(abs(ydata0)) for wavelenghts < say 2795.5 A has to be ~ 10 Intensity units
    # reject data when you are inside the limb
    # 1 criteria could be Intensity abs mean inside ~ 2795 A should be < 20 or so
    ix1 = np.argwhere(xdata0 <= param_lambda_ul)[:,0]
    mean_tydata = np.mean(abs(ydata0[xdata0 <= param_lambda_ul]))
    if mean_tydata > MgIIk_SImean_ul:
        if key_print_error_message ==1:
            print('Pixel data0 inside limb. Rejecting it')
        return None, None 

    return xdata0, ydata0
# ------------------------------------------------------
def pCMEx_calc_line_widths_For_RCasini_synthetic_spectra(file_fnm=None, nparam_lines=5, key_scale_SI_to_1=1, key_plot_SI=0):
    '''
    written on Dec 14 2023
    Goal: 1. to read Roberto Casini's synthetic spectra file; 
          2. fit with a Gaussian, 
          3. return Gsigma_arr for each of these synthetic spectral lines

    Input:

        - nparam_lines [default =5] = number of lines for each spectra in file_fnm that correspond to parameters 
                        that went into creating the synthetic spectra
        - key_scale_SI_to_1 = 1 to scale SI to max =1 
        - key_plot_SI = 1 to plot individual spectra + fit to it

    Output:
        Gsigma_arr = array of Gaussian width of the spectral line
    '''

    from itertools import islice
    from scipy.optimize import curve_fit 

    j=0
    j1=0
    Gsigma_arr = np.zeros(500000) + np.nan 
    k=0

    with open(file_fnm, "r") as text_file:
        # initialize parameters 
        for tline in islice(text_file, None): # read all the lines
            if tline == '\n': # the starting line for each spectra is a blank space
                j=0  # iterator for each spectral line 
                j1=0 # iterator for each wavelength (in a spectral line)
                continue
            else:
                j+=1
                if j < nparam_lines:
                    continue # just move to the next line

                elif j == nparam_lines:
                    lambda0 = np.float64(list(tline.split())[1]) # lambda0 (center of gravity of the line )
                    continue

                elif j == nparam_lines+1:
                    nlambda = int(list(tline.split())[1]) # number of wavelength points
                    npts_per_spectra = nlambda + nparam_lines +1 # tot no. of pts = 1 spectra in file
                    dlambda_SI_arr = np.zeros((nlambda,5)) + np.nan # initialize for this spectra; each row = [lambda, SI, SQ, SU, SV]
                    continue

                elif ((j > nparam_lines+1) & (j <= npts_per_spectra)): # read the spectral data 
                    ttline = list(tline.split())
                    dlambda_SI_arr[j1, :] = ttline[:5]
                    j1+=1 
                
                    if j == npts_per_spectra: # now compute stats and fit it
                        lambda_arr = dlambda_SI_arr[:,0] + lambda0
                        SI_arr = dlambda_SI_arr[:,1]
                        
                        if key_scale_SI_to_1 == 1:
                            SI_arr = SI_arr/SI_arr.max()
                            
                        # do the fitting
                        xloc_SImax = lambda_arr[SI_arr == SI_arr.max()][0]

                        try:
                            popt0, pcov0 = curve_fit(func_gauss1D, lambda_arr, SI_arr, p0 = \
                                                 [0., SI_arr.max(), xloc_SImax, 0.2])
                        except RuntimeError:
                            print('for k={}; curvefit failed'.format(k))
                            continue
                        
                        fit_y0 = func_gauss1D(lambda_arr, *popt0)
                        Gsigma_arr[k] = popt0[3]
                        k +=1 # increase number of Spectras you got
                        
                        # do the plotting               
                        if key_plot_SI==1:
                            if k > 100:
                                continue
                            fig, ax = plt.subplots(figsize=(5,2))
                            ax.set_xlabel('wavelength [Angs]')

                            if key_scale_SI_to_1 ==1:
                                ax.set_ylabel('Intensity [photon flux]')
                            else:
                                ax.set_ylabel('Intensity [scaled to max =1]')
                            ax.grid()
                            ax.set_title(' nth_SI={}; Gsigma={:.2f} \n file={}'.format(k, popt0[3], os.path.basename(file_fnm)), fontsize=8, loc='left')

                            ax.plot(lambda_arr, SI_arr, 'r.-')
                            ax.plot(lambda_arr, fit_y0, 'g.-')
                            plt.show()

                    
    Gsigma_arr = Gsigma_arr[:k]
    return Gsigma_arr
# ------------------------------------------------------

# ------------------------------------------------------
# code status: to be checked; func documentation to be written
@njit()
def pCMEx_extract_linecore_region(xdata_arr=None, ydata_arr=None, 
                              y_ll=None, noise_percent_in_line=100.):
    '''
    written on Dec 22 2023
    Goal: to extract the line region (specifically for Mg IIk data)

    Steps: 
        - find ypeak's index location
        - go to the left from peak_ixloc - up until you hit the noise level. 
        - do the same to the right of the peak_xloc
        - count number of pixels that make up this line-region
        - then extend the x,y range to add some noise level pixels.
        - ensure that your x,y range around the line center doesn't not overflow and go

    Caveats:
        - ensure y_ll is small enough that pixels added to when using noise_percent_in_line are 
          actually correspond to noise. Else the extracted line region may be skewed around
          line peak

    '''

    ix_arr = np.arange(xdata_arr.size)
    peak_ix = ix_arr[ydata_arr == np.max(ydata_arr)][0]

    if ydata_arr[peak_ix] < y_ll:
        raise NameError('ydata_arr max < y_ll. We wont any observations that make up a line')
    
    # line's left index limit  
    ix_arr_lpeak = ix_arr[((ix_arr < peak_ix) & (ydata_arr <=  y_ll))]
    l_ix = ix_arr_lpeak[-1]
    l_xdata = xdata_arr[l_ix]

    # line's right index limit  
    ix_arr_rpeak = ix_arr[((ix_arr > peak_ix) & (ydata_arr <=  y_ll))]
    r_ix = ix_arr_rpeak[0]+1 # line's right index limit
    r_xdata = xdata_arr[r_ix]

    # now add some few pixels 
    line_ix_len = r_ix-l_ix+1 # no. of points that make up the line region
    half_n_noisy_pix = round(0.5 * line_ix_len * noise_percent_in_line/100.) # number of pixels that correspond to noise

    l_ix = np.max(np.array([0, l_ix-half_n_noisy_pix]))
    r_ix = np.min(np.array([xdata_arr.size, r_ix+half_n_noisy_pix]))

    # find the xloc corresponding to l_ydata, r_ydata
    n_xdata_arr = xdata_arr[l_ix:r_ix] # new xdata_arr
    n_ydata_arr = ydata_arr[l_ix:r_ix]


    return n_xdata_arr, n_ydata_arr, (l_xdata, r_xdata)


# ------------------------------------------------------
def pCMEx_multigauss_fit(xarr=None, yarr=None, npeaks_to_fit=1, 
                         curvefit_bounds=None, key_plot=0, 
                         key_print_error_message=0):
    '''

    '''

    fit_yarr = xarr+np.nan
    fit_param = np.array(curvefit_bounds[0]) + np.nan

    if np.shape(curvefit_bounds)[1] != 3*npeaks_to_fit:
        raise NameError('npeaks_to_fit and curvefit_bounds shape are not consistent')

    if npeaks_to_fit ==1:
        try:
            fit_param, _ = curve_fit(pCMEx_Gauss1D_1peak, xarr, yarr, bounds=curvefit_bounds)            
            fit_yarr = pCMEx_Gauss1D_1peak(xarr, *fit_param)
        except:
            if key_print_error_message ==1:
                print('Code wasnt able to fit.')
            return fit_yarr, fit_param

    elif npeaks_to_fit==2:
        try:
            fit_param, _ = curve_fit(pCMEx_Gauss1D_2peak, xarr, yarr, bounds=curvefit_bounds)            
            fit_yarr = pCMEx_Gauss1D_2peak(xarr, *fit_param)
        except:
            if key_print_error_message ==1:
                print('Code wasnt able to fit.')
            return fit_yarr, fit_param

    elif npeaks_to_fit==3:
        try:
            fit_param, _ = curve_fit(pCMEx_Gauss1D_3peak, xarr, yarr, bounds=curvefit_bounds)            
            fit_yarr = pCMEx_Gauss1D_3peak(xarr, *fit_param)
        except:
            if key_print_error_message ==1:
                print('Code wasnt able to fit.')
            return fit_yarr, fit_param
    else:
        raise NameError('Myerr: npeaks_to_fit can only be 1, 2, or 3 peaks.')

    if key_plot ==1:
        fig, ax =plt.subplots()
        ax.plot(xarr, yarr, 'k.-')
        ax.plot(xarr, fit_yarr, 'g-')
        ax.grid()
        plt.show()


    return fit_yarr, fit_param
# ------------------------------------------------------
# checked on dec 20 2023
@njit()
def pCMEx_Gauss1D_1peak(xarr, A, mu, Gsigma):
    '''
    written on Dec 13 2023
    Aim - create an 1D Gaussian gunction = C1 * exp^(-((x-x0)^2/(2.*sigma^2)))

    Input:
        - xarr = array of x values of the function
        - A = peak value of the Gaussian
        - mu = peak x-location of the Gaussian
        - Gsigma = width of the Gaussian

    Output:
        yarr [same size as xarr] - Gaussian value at xarr values.

    '''
    return A*np.exp(-(xarr-mu)**2/(2.*Gsigma**2))
# ------------------------------------------------------
# checked on dec 20 2023
@njit()
def pCMEx_Gauss1D_2peak(xarr, A1, mu1, Gsigma1, A2, mu2, Gsigma2):
    '''
    written on Dec 13 2023
    Aim - return a Gaussian 1D array at xarr - that is sum of 2 Gaussians

    Input:
        - xarr = array of x values of the function
        - A1,2 = peak values of the 2 Gaussians
        - mu1,2 = peak x-locations of the Gaussians
        - Gsigma1,2 = widths of the Gaussians

    Output:
        yarr [same size as xarr] - Gaussian value at xarr values.

    '''
    return A1*np.exp(-(xarr-mu1)**2/(2.*Gsigma1**2)) + \
           A2*np.exp(-(xarr-mu2)**2/(2.*Gsigma2**2))
# ------------------------------------------------------
# checked on dec 20 2023
@njit()
def pCMEx_Gauss1D_3peak(xarr, A1, mu1, Gsigma1, A2, mu2, Gsigma2, A3, mu3, Gsigma3):
    '''
    written on Dec 13 2023
    Aim - return a Gaussian 1D array at xarr - that is sum of 3 Gaussians

    Input:
        - xarr = array of x values of the function
        - A1,2 = peak value of the 2 Gaussians
        - mu1, mu2 = peak x-location of the Gaussians
        - Gsigma1,2 = width of the Gaussians

    Output:
        yarr [same size as xarr] - Gaussian value at xarr values.

    '''
    return A1*np.exp(-(xarr-mu1)**2/(2.*Gsigma1**2)) + \
           A2*np.exp(-(xarr-mu2)**2/(2.*Gsigma2**2)) + \
           A3*np.exp(-(xarr-mu3)**2/(2.*Gsigma3**2))
# ------------------------------------------------------

# # ------------------------------------------------------
# # @njit()
# def pCMEx_Gauss1D_multipeak(xarr, *args): # C0, C1_1, x0_1, Gsigma0_1, C1_2, x0_2, Gsigma0_2):
#     '''
#     written on Dec 13 2023
#     Aim - return a Gaussian 1D array at xarr

#     Input:
#         - xarr = array of x values of the function

#         - params_arr = 1D array of parameters

#         - C0 = baseline amplitude of the Gaussian
#         - C1 = peak value of the Gaussian (wrt the C0 baseline)
#         - x0 = peak x-location of the Gaussian
#         - Gsigma0 = width of the Gaussian

#     Output:
#         yarr [same size as xarr] - Gaussian value at xarr values.

#     '''
#     params_arr = args[0] # get the list/array out of tuple
#     C0 = params_arr[0]
#     yarr = np.zeros(xarr.size) + C0
#     for k in range(1, len(params_arr), 3):
#         tC1 = params_arr[k]
#         tx0 = params_arr[k+1]
#         tGsigma0 = params_arr[k+2]

#         yarr += C0 + tC1*np.exp(-(xarr-tx0)**2/(2.*tGsigma0**2))
#     return yarr


# # ------------------------------------------------------
# # @njit() 
# def pCMEx_231213_create_multiple_Gauss(xarr=None, C0=None, params_arr=None): #C0, C1_arr, x0_arr, Gsigma0_arr):
#     '''
#     written on Dec 13 2023
#     Aim - to create function with multiple Gaussians [1D array]

#     Input:
#         - xarr = array of x values of the function
#         - each row of params:

#             - C0 = baseline amplitude of Gaussian
#             - params[i,0] = C1 [array] = peak value of individual Gaussians (wrt the C0 baseline)
#             - params[i,1] = x0 [array] = peak x-location of individual Gaussians
#             - params[i,2] = Gsigma0 [array] = width of individual Gaussians

#     Output:
#         yarr [same size as xarr] - sum of individual Gaussians

#     '''
#     yarr = np.zeros(xarr.size) + C0 # place holder

#     for i in range(params_arr.shape[0]):
#         tC1 = params_arr[i, 0]
#         tx0 = params_arr[i, 1]
#         tGsigma0 = params_arr[i, 2]

#         yarr += pCMEx_Gauss1D_1peak(xarr, 0., tC1, tx0, tGsigma0)

#     return yarr

# @njit(fastmath=True)
# def num_to_binary(number, size_of_bin = 32):
#     out = np.zeros(size_of_bin)
#     num = number
#     index = 31

#     for i in prange(size_of_bin):
#         floatDivide = num // 2
#         divide = num / 2
#         if floatDivide != divide:
#             out[index] = 1

#         num = floatDivide

#         index-= 1
#         if index == -1 or floatDivide == 0:
#             break

#     return out

# ------------------------------------------------------ 
def write_array_to_txt_file(filename=None, format1='%18.9e', arr1=None):
    with open(coeff_to_txt_file, 'a') as f:
        void = np.savetxt(f, arr1, fmt=format1)

def printdone(text=None):
    '''duplicate of done()'''
    if text is None:
        text = ''
    #   print('{} {date:%Y-%m-%d: %H:%M:%S}'.format('Completed@ ', date=datetime.datetime.now())) # note the use of date within {}
    # else:
    print('{}; {} {date:%Y-%m-%d: Done@ %H:%M:%S}'.format(text, 'Completed@ ', date=datetime.now())) # note the use of date within {}
    return None
def printstart(text=None):
    '''duplicate of done()'''
    if text is None:
        text = ''
    #   print('{} {date:%Y-%m-%d: %H:%M:%S}'.format('Started@ ', date=datetime.datetime.now())) # note the use of date within {}
    # else:
    print('{}; {} {date:%Y-%m-%d: %H:%M:%S}'.format(text, 'Started@ ', date=datetime.now())) # note the use of date within {}
        # print('{date:%Y-%m-%d: Start@ %H:%M:%S}'.format(date=datetime.datetime.now())) # note the use of date within {}
    return None
def printmynote(str1):
    '''
    written on Oct 7 2021
    Aim: to quickly make my notes at a location

    '''
    print('My note>>> {}'.format(str1))
    return
# ------------------------------------------------------
# --------------------------------------
# code status: green, triple checked on Jan 16 2024
@njit()
def DISCARD_pCMEx_PCA_inv_1_profile_calc_chisqr_err_arr(
                                    obs_IQUV=None, 
                                    
                                    PCA_eigenbasis_IQUV=None, 
                                    PCA_IQUV_mean0=None, 
                                    PCA_dbase_coeff_all=None, 

                                    nmodes_to_use=-1, 

                                    key_normalize_obs_IQUV_to_Imax=0, 
                                    key_subtract_IQUV_mean0=1, 
                                    key_use_IQUV_weights_for_chisqr=0, 
                                    # key_use_squard_IQUV_weights=1, 
                                    noise_level_wrt_obs_Imax=np.nan, 

                                    key_use_IQUonly_for_chisqr_calc=0, 

                                    key_print_PAnotes=1,
                                    
                                    
                                        ):
    '''
    Written on Jan 11 2024
    Goal to do PCA inversion for a single profile and return chisqr_err_arr 
            (measure of how close are the eigen coeff for the obs_IQUV wrt each profile in the database)

    Input:
        - obs_IQUV: nlambda by 4 matrix, in IQUV order
        - PCA_eigenbasis_IQUV : (4 by nlambda by nmodes) matrix containing eigen modes of the PCA basis
        - PCA_IQUV_mean0: nlambda by 4 matrix containing mean value at each wavelength for each IQUV
                            - these are likely normalized to obs_Imax
        - PCA_dbase_coeff_all: nmodes by 4 by number of profiles in the database
                            - it is this database to which coeff for obs_IQUV would be compared to

        - nmodes_to_use = total number of modes to use when decomposing obs_IQUV onto PCA_eigenbasis_IQUV
            - -1 implies using all the modes in PCA_eigenbasis_IQUV

        - keywords
            - key_normalize_obs_IQUV_to_Imax =1 to normalize all values in obs_IQUV with obs_Imax
            - key_subtract_IQUV_mean0 =1 to subtract PCA_IQUV_mean0 from obs_IQUV. A standard in PCA
            - key_use_IQUV_weights_for_chisqr =1 to use weighted chisqr
                - weights are calculated using pCMEx_get_IQUV_weights_for_chisqr_calc, which also
                    requires noise_level_wrt_obs_Imax. 
                    - noise_level_wrt_obs_Imax is more a factor. noise_level_wrt_obs_Imax*obs_Imax is the 
                      noise level in the obs_IQUV

            - key_print_PAnotes =1 to pring some PA notes

    Output:
        - chisqr_err_arr : Euclidean distance of coeff_obs_IQUV1 wrt each coeff in PCA_dbase_coeff_all
        - coeff_obs_IQUV1 : eigen coeff for obs_IQUV
        - obs_IQUV1: Note this could be normalized or not depending on if key_normalize_obs_IQUV_to_Imax=1 or not.

    - PA concerns:
        - does it make sense to use noise_level as upper limit in chisqr_wts calculation

    Steps:
        1) normalize obs_IQUV data by Imax
        2) subtract PCA_IQUV_mean0 from obs_IQUV
        3) decompose obs_IQUV1 on PCAbasis and calc obs_coeff_IQUV
        4) get chisqr weights for IQUV
        5) calc chisqr_err_arr: for obs_coeff_IQUV and coeff for each profile 
           in the PCA databbase weighted by respective chisqr weights
        6) return chisqr_err_arr

    Updates:
        [Jan 18 2024]: PCA_eigenbasis_IQUV shape changed to (4, nlambda, nmodes) from (nlambda, nmodes, 4).
                            - Reason: Prevents numba error: @ matrix multiplication is faster on 
                                      contiguous arrays.

    '''

    # ----------------------------
    if nmodes_to_use ==-1:
        nmodes_to_use = PCA_eigenbasis_IQUV.shape[2] # shape = (4, nlambda, nmodes)
    elif ((nmodes_to_use > PCA_eigenbasis_IQUV.shape[2]) | (nmodes_to_use <= 0)):
        raise NameError('PAerror: nmodes_to_use is either > nmodes in PCA_eigenbasis_IQUV or is <= 0.')

    # ----------------------------
    obs_IQUV1= obs_IQUV+0. # retain original obs_IQUV just in case
    # ----------------------------
    # normalize all obs_IQUV1 with Imax
    obs_Imax0 = np.max(np.abs(obs_IQUV1[:,0]))
    if key_normalize_obs_IQUV_to_Imax ==1:
        obs_IQUV1 = obs_IQUV1/obs_Imax0

    # ----------------------------
    # subtract PCA_IQUV_mean0 from obs_IQUV1
    if key_subtract_IQUV_mean0 ==1: 
        # U eigenbasis was created from a set of Stokes IQUV after subtracting PCA_IQUV_mean0
        # so, we need to do the same with obs_IQUV1 before decomposing it to eigen basis
        obs_IQUV1 = obs_IQUV1 - PCA_IQUV_mean0 # subtracts corresponding IQUV profiles
    else:
        raise NameError('PA: key_subtract_IQUV_mean0 should be 1.')
    
    # ----------------------------
    # decompose obs_IQUV1 on PCAbasis and calc coeff
    coeff_obs_IQUV1 = np.zeros((nmodes_to_use, 4)) + np.nan # placeholder    
    for j in range(4): # loop over IQUV
        coeff_obs_IQUV1[:,j] = PCA_eigenbasis_IQUV[j,:,:nmodes_to_use].T @ obs_IQUV1[:,j]
        # coeff_obs_IQUV1[:,j] = PCA_eigenbasis_IQUV[:,:nmodes_to_use,j].T @ obs_IQUV1[:,j] # reshaped post Jan 18 2024.
    
    # ----------------------------
    # calc weights for chisqr calculation
    # wts copied from RCasini: e.g. Q_weight = Imax/(Qmax-Qmin)
    # for too small weights - there is an upper limit of 1/noise_level. Why this makes sense?
    # checked the following with Roberto's code
    chisqr_wts = np.zeros(4) + 1.
    if key_use_IQUV_weights_for_chisqr ==1:
        chisqr_wts = pCMEx_get_IQUV_weights_for_chisqr_calc(obs_IQUV=obs_IQUV1,
                        noise_level_wrt_obs_Imax=noise_level_wrt_obs_Imax,)
        if key_print_PAnotes==1:
            print('PA: does it make sense to use noise_level as upper limit in chisqr_wts calculation?')

    # ----------------------------
    # chi_squared calculation - 
    # compare Euclidean distances: between coeff_obs_IQUV1 and those in PCA_dbase_IQUV
    # errors made before Jan16 2024 calculations: 
        # 1. used resquared chisqr_wts
        # 2. used tsum += np.linalg.norm(chisqr_wts[k2] * tdiff**2, ord=2). 
        #      This is summing Root of Squared sum over IQUV, ∑_IQUV √ ∑(c-c_i)**2 while we want 
        #      summing of squared sums ∑_IQUV √ ∑(c-c_i)**2
        #                                                                 

    # calc chisqr Euclidean distance against each coeff in PCA database
    # PCA_dbase_coeff_all.shape = (nmodes, 4, nprofiles_in_database)

    if key_use_IQUonly_for_chisqr_calc ==1: 
        # restricts loop to IQU only in chisqr arr calculation
        # Assumes 1st 3 k-index in coeff_obs_IQUV1[:, k] 
        # correspond to IQU and it is the last index that is for Stokes V
        tn1 = 3
    else:
        tn1 = 4


    # Get number of modesls in the PCA database
    chisqr_err_arr = np.zeros(PCA_dbase_coeff_all.shape[2]) + np.nan # place holder, chisqr for each coeff_arr in PCA database
    chisqr_err_arr_IQUV = np.zeros((chisqr_err_arr.size, 4)) + np.nan
    for k1 in range(chisqr_err_arr.size): # loop over database
        tsum =0
        for k2 in range(tn1): # loop over IQUV and calc total chisqr (weighted or un-weighted)
            tdiff = coeff_obs_IQUV1[:,k2] - PCA_dbase_coeff_all[:nmodes_to_use,k2,k1]
            tsqr_sum1 = np.sum(tdiff**2)  # Jan 17
            chisqr_err_arr_IQUV[k1, k2] = tsqr_sum1 # individual sums 

            #  - to be implemented directly to chisqrs
            # chisqr_wts= all 1s if key_use_IQUV_weights_for_chisqr=0
            tsum +=  tsqr_sum1 * chisqr_wts[k2] # Note chisqr_wts are already squared

        chisqr_err_arr[k1] = tsum

    if key_subtract_IQUV_mean0 ==1:
        return chisqr_err_arr, coeff_obs_IQUV1, obs_IQUV1 + PCA_IQUV_mean0, chisqr_err_arr_IQUV
    else:
        return chisqr_err_arr, coeff_obs_IQUV1, obs_IQUV1, chisqr_err_arr_IQUV



# ----------------------------
# --------------------------------------
# Code status: green, documentation Complete
@njit()
def DISCARD_pCMEx_get_IQUV_sqr_wts_for_chisqr_calc(
                key_use_obsIQUV_ratio_as_wts_for_chisqr=0, # dI/dV
                key_use_obsIQUV_L2n_as_wts_for_chisqr=0,  # |obs_I_C|/|obs_V_C|
                key_use_IQUV_L2product_as_wts_for_chisqr=0, # 1/|obs_I_C| * 1/|pca_I_C|

                obs_IQUV= np.zeros((1,1))+np.nan, # when key_use_obsIQUV_ratio_as_wts_for_chisqr=1
                obs_C_IQUV = np.zeros((1,1))+np.nan, # when key_use_obsIQUV_L2n_as_wts_for_chisqr=1
                max_chisqr_wt_UnSq=np.nan, # = np.nan => no thresholding

                # following when key_use_IQUV_L2product_as_wts_for_chisqr=1
                pca_C_IQUV = np.zeros((1,1))+np.nan, 
                obs_C_IQUV_L2n = np.array([np.nan]),  # precompute to save computing time
                ):
    '''
    written on Jan 10 2024
    Updated on Feb 19 2024

    Goal: Calc the weights for IQUV to be used in chisqr calc
          to put e.g ∑ (obs_coeff_Q - database_coeff_Q)**2 on the same amplitude as
                     ∑ (obs_coeff_I - database_coeff_I)**2 - so that the 

                     Goodness of fit is not dominated by say - I measurements only
                     Q has some say in it too.

    Input:
        - key_use_obsIQUV_ratio_as_wts_for_chisqr=0, # dI/dV as weights
            - only requires obs_IQUV and max_chisqr_wt_UnSq
            - mimics RCasini method: e.g. Q_weight = Imax/(Qmax-Qmin)

        - key_use_obsIQUV_L2n_as_wts_for_chisqr=0,  # |obs_I_C|/|obs_V_C|
            - obs_C_IQUV = PCA coeff of obs IQUV

        - key_use_IQUV_L2product_as_wts_for_chisqr=0, 
            - weights = 1/|obs_I_C| * 1/|pca_I_C|
            - requires obs_C_IQUV_L2n, 
                       pca_C_IQUV, 
                       obs_C_IQUV (if obs_C_IQUV_L2n is NaN)
                       max_chisqr_wt_UnSq


        - obs_IQUV = obs_IQUV: column1= SI, column 4 = Stokes V
            - Caution: this should be normalized to Imax=1, and have pca mean subtracted

        - obs_C_IQUV = PCA coeff of obs IQUV
        - pca_C_IQUV = PCA coeff of pca IQUV (the one in the database)

        - obs_C_IQUV_L2n - L2norm of obs_C_IQUV, 
            - 4-element array 

        - max_chisqr_wt_UnSq = maximum unsquared weights; UnSq = un squared
            if = np.nan - then weights are unbounded
    
    Output:
        chisqr_wts = 4 element array 
            - carrying weights for IQUV to be used 
            - with a ceiling of max_chisqr_wt_UnSq (if it is not NaN)
            - to be used as is in chisqr calculation - no squaring needed of the weights outputted 

    
    '''
    if key_use_obsIQUV_ratio_as_wts_for_chisqr + \
        key_use_obsIQUV_L2n_as_wts_for_chisqr  + \
        key_use_IQUV_L2product_as_wts_for_chisqr !=1:
        raise NameError('PAErr: pick only 1 approach to chisqr wt calculation.')
    # ----------------------------------------------
    wts_arr = np.zeros(4) + np.nan # place holder
    # ----------------------------------------------
    
    if key_use_obsIQUV_ratio_as_wts_for_chisqr ==1: # intensity ratio of obs_IQUV as weights e.g. dI/dV 
                                                    # Old: used by RCasini
        wts_arr[0] = 1. # intensity weight =1 
        obs_dI = np.max(obs_IQUV[:,0]) - np.min(obs_IQUV[:,0])

        for k in range(1,4): # loop over QUV and get weights
            tStokes_diff = np.max(obs_IQUV[:,k]) - np.min(obs_IQUV[:,k])   
            tweight = abs(obs_dI/tStokes_diff)
            wts_arr[k] = np.nanmin([tweight, max_chisqr_wt_UnSq]) # threshold wt, unless max_chisqr_wt_UnSq = NaN
        return wts_arr**2

    # ----------------------------------------------
    elif key_use_obsIQUV_L2n_as_wts_for_chisqr==1: # I_wt = 1/L2n(C_I) # this makes the most sense to me (PA)
        wts_arr[0] = 1. # intensity weight = 1
        I_C_L2n = np.linalg.norm(obs_C_IQUV[:,0], ord=2) # Intensity Coeff L2norm

        for k in range(1,4): # loop over QUV
            tweight = I_C_L2n/np.linalg.norm(obs_C_IQUV[:,k], ord=2)
            wts_arr[k] = np.nanmin([tweight, max_chisqr_wt_UnSq]) # threshold wt, unless max_chisqr_wt_UnSq = NaN
        return wts_arr**2 # squaring is needed here 

    # ----------------------------------------------
    elif key_use_IQUV_L2product_as_wts_for_chisqr==1: # this is what Roberto computed
        if np.isnan(obs_C_IQUV_L2n[0]): # np.isnan(obs_C_IQUV_L2n).any():
            obs_C_IQUV_L2n = np.zeros(4) + np.nan # Initialize
            for k in range(4):
                obs_C_IQUV_L2n[k] = np.linalg.norm(obs_C_IQUV[:,k], ord=2)

        for k in range(0, 4):  # loop over all IQUV; wt for chisqr_I is not 1
            pca_tC_L2n = np.linalg.norm(pca_C_IQUV[:,k], ord=2)

            obs_twt = np.nanmin([1/obs_C_IQUV_L2n[k], max_chisqr_wt_UnSq]) # get obs wt and compare that to threshold
            pca_twt = np.nanmin([1/pca_tC_L2n, max_chisqr_wt_UnSq])
            wts_arr[k] = obs_twt*pca_twt

        return wts_arr # Note: Do not square: as wts_arr = 1/|obs_C|_2  * 1/|pca_C|_2 = product of L2 norms

    else:
        raise NameError('PAErr (Feb 19 2024): Select atleast 1 key to calc wts according to it.')



































