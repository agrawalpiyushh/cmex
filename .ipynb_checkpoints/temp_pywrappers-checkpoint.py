from datetime import datetime, timedelta # helps find date differnce 
import numpy as np
from itertools import islice

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
        nprofiles = 5000000 # set to some super larger number 
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

# code status: Green
# @njit()
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
