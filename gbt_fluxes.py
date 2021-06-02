# script to calculate fluxes given the .out file produced by prepoint.
# should handle all different frequencies.
#
#
# data format in the .out file is the following:
# GBT17A_998_47,    AzF, LCP,    1, 58089.918403, 2209+236, Model 5s - latest, 333.02487500,  23.92792778,   1.66800000, 145.15408791,  73.04577681, 145.08298992,  73.08516315,   0.80664668,   0.04405393, 492.49874348,  31.05816785, -23.30697635,  13.18918951,   0.13078128, 934.57070914,   6.76384341,   1.41209981,  16.25427444

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from scipy.stats import zscore
from math import isnan

"""This script gbt_fluxes.py extracts sources' fluxes from the GBT pointing 
scans performed before RadioAstron observations. 
Actual data are provided upon a reasonable request.

The general logic behind the script is: 
    * match GBT observations codes with those of RadioAstron
    * for each relevant observation, extract the last cross-scans in both 
        polarisations (the last ones should be the best ones)
    * for each frequency, perform basic cleaning of the data, see details in 
        the description of clean_anyband(). Optionally produce plots
    * Write out the cleaned data on a per band basis

"""



__author__ = 'Mikhail Lisakov'
__copyright__ = 'Copyright 2021, GBT pointing scans fluxes for RadioAstron'
__license__ = 'GNU GPL v3.0'
__version__ = '0.9'
__status__ = 'Dev'
__comment__ = 'Data are provided upon a reasonable request'


def read_ragbt(FILE):
    '''read results of correlation on baselines with GBT.
    The goal is to check whether there is any dependence of fringe detection with pointing accuracy
    
    Typical GBT beam FWHM 
    18 cm = L-band : 450 arcsec
    6 cm  = C-band : 150 arcsec
    1.3cm = K-band : 33  arcsec
    
    Let's say that being half-beam off means bad pointing. 
    
    Here is the format : 
 exper_name | band |  source  | polar |     scan_start      |   st1    |   st2    |   snr    | status
------------+------+----------+-------+---------------------+----------+----------+----------+--------
 rafs17     | k    | 0754+100 | RR    | 2012-03-11 21:30:06 | RADIO-AS | GBT-VLBA |    32.11 | y
 rafs17     | k    | 0754+100 | RR    | 2012-03-11 21:40:07 | RADIO-AS | GBT-VLBA |    29.71 | y
    '''
    
    df = pd.read_csv(FILE, delimiter = '\s*\|\s*', engine = 'python')
#    data.rename(columns=lambda x: x.strip(), inplace = True)
    
    
    return df



def reduce_gbt(df):
    '''perform some essential operations on the GBT dataframe:
        - convert obscodes to uppercase short notation, e.g. raks12oz -> RK12OZ
        - make one dataframe with only GBT-RA baselines, taking the highest snr per experiment per correlation (RR,LL, RL?, LR?, or simply only one value?)
        - make another dataframe with ground-GBT baselines, taking the highest snr per experiment
        '''
        
    print(df.columns)
    df.exper_name = df.exper_name.str.upper()
    df.exper_name = df.exper_name.str.replace('RAKS', 'RK')
    df.exper_name = df.exper_name.str.replace('RAGS', 'RG')
    df.exper_name = df.exper_name.str.replace('RAES', 'RE')
    df.exper_name = df.exper_name.str.replace('RAFS', 'RF')

    dra = df[df.st1 == 'RADIO-AS']      # df for RA-GBT baselines only 
    dg  = df[(df.st1 != 'RADIO-AS') & (df.st2 != 'RADIO-AS')]  # groung-GBT baselines only
    
    dra = dra.groupby(['exper_name']).max()
    
    
    



    return dra, dg


def readout(FILE):
    '''read all data from a given file.
    FORMAT: 
ProjectID, Type, Pol, Scan, MJDate, Source, PointingModel, J2000RA, J2000Dec, ObsFreq, Obsc_Az, Obsc_El, Mnt_Az, Mnt_El, Peak, PeakError, Width, WidthError, Offset, OffsetError, BaselineRMS, AzRate, ElRate, TCal, TSys
    '''
    data = pd.read_csv(FILE, delimiter=' *, *', engine='python')
    data.rename(columns=lambda x: x.strip(), inplace = True)
#    data.str(lambda x: x.strip())
    
#    print('Read the following columns from the file {}:\n{}'.format(FILE, data.columns))
    
    return data


def filterout(d):
    '''get the last scans.
    Generally, the last El scans, forward and back, should be the best ones.'''
    
    
    global read_errors
    
    # treat polarisations separately
    l = d[d.Pol == 'LCP']
    r = d[d.Pol == 'RCP']
    
    # RCP 
    lastAzFscanR = r.loc[r.Type == 'AzF', 'Scan'].max()
    lastAzBscanR = r.loc[r.Type == 'AzB', 'Scan'].max()
    lastElFscanR = r.loc[r.Type == 'ElF', 'Scan'].max()
    lastElBscanR = r.loc[r.Type == 'ElB', 'Scan'].max()
    
    
    #LCP
    lastAzFscanL = l.loc[l.Type == 'AzF', 'Scan'].max()
    lastAzBscanL = l.loc[l.Type == 'AzB', 'Scan'].max()
    lastElFscanL = l.loc[l.Type == 'ElF', 'Scan'].max()
    lastElBscanL = l.loc[l.Type == 'ElB', 'Scan'].max()
    
    print('RACODE = {}'.format(d.ProjectID.values[0]))
    print(lastAzFscanR, lastAzBscanR, lastElFscanR, lastElBscanR)
    print(lastAzFscanL, lastAzBscanL, lastElFscanL, lastElBscanL)
    
    
    last_scanR = np.nanmax([lastAzFscanR, lastAzBscanR, lastElFscanR, lastElBscanR])      # RCP: just take the last scan. Usually, this should be ElB. 
    last_scanL = np.nanmax([lastAzFscanL, lastAzBscanL, lastElFscanL, lastElBscanL])      # LCP: just take the last scan. Usually, this should be ElB. 
    
    # if there are valid data for both RCP and LCP
    if (not isnan(last_scanR)) and (not isnan(last_scanL)):
        print('both RCP and LCP are not nan')
        print('RCP: Last scan No. {} type is {}'.format(last_scanR , r.loc[(r.Scan == last_scanR), 'Type'].values[0]))
        print('LCP: Last scan No. {} type is {}'.format(last_scanL , l.loc[(l.Scan == last_scanL), 'Type'].values[0]))
        return d.loc[((d.Scan == last_scanR) & (d.Pol == 'RCP')) | ((d.Scan == last_scanL) & (d.Pol == 'LCP')) ]
    
    # if there are no valid data for RCP
    if isnan(last_scanR):
        print('RCP is NaN')
        read_errors['RCP'] = read_errors['RCP'] + 1
        last_scanR = last_scanL
        t = d.copy(deep=True)
        t.Pol = 'RCP'
        print(d)
        print(t)
        d = pd.concat([d,t])
        return d.loc[((d.Scan == last_scanR) & (d.Pol == 'RCP')) | ((d.Scan == last_scanL) & (d.Pol == 'LCP')) ]
    
    # if there are no valid data for LCP
    if isnan(last_scanL):
        print('LCP is NaN')
        read_errors['LCP'] = read_errors['LCP'] + 1
        last_scanL = last_scanR
        t = d.copy(deep=True)
        t.Pol = 'LCP'
        print(d)
        print(t)
        d = pd.concat([d,t])
        return d.loc[((d.Scan == last_scanR) & (d.Pol == 'RCP')) | ((d.Scan == last_scanL) & (d.Pol == 'LCP')) ]
    

def averout(d):
    '''calculate average values of RA, DEC, azimuth, elevation, observing frequency, date'''
    return  d.loc[:, ['J2000RA', 'J2000Dec', 'Obsc_Az', 'Obsc_El', 'ObsFreq', 'MJDate']].mean()

    

def make_res(d):
    '''this function will average all ElF and ElB scans together (different for different polarisations.)
    So it is better to feed it with output of filterout()
    '''
    res = d.groupby(['Pol', 'Source', 'ProjectID']).mean()
    res.reset_index(level=res.index.names, inplace=True)
#    res.loc[(res.Pol == 'LCP') | (res.Pol == 'RCP'),'DPFU'] = DPFU(res.ObsFreq.drop_duplicates().values) 
    res.loc[(res.Pol == 'LCP') ,'DPFU'],res.loc[(res.Pol == 'RCP') ,'DPFU'] = DPFU(res.ObsFreq.drop_duplicates().values[0]) 
    res.loc[:,'POLY'] = POLY(res.ObsFreq.drop_duplicates().values[0], res.Obsc_El.drop_duplicates().values[0])
    # calculate SEFD, and source flux density
    res['SEFD'] = res['TSys'] / (res['DPFU'] * res['POLY'])
    res['FLUX'] = res['Peak'] * res['TCal'] / (res['DPFU'] * res['POLY'])
    res['FLUXERR'] = res['PeakError'] * res['TCal'] / (res['DPFU'] * res['POLY'])
    
    return res




def DPFU(freq):
    '''return GBT (DPFU_LPOL, DPFU_RPOL) in [K/Jy] for a given frequency to convert Kelvins to Janskys.
    According to https://safe.nrao.edu/wiki/bin/view/GB/Observing/GainPerformance
    Gmax there.
    '''
    if freq > 1.400 and freq < 2.000: # L-band
#        print('USING L-BAND DPFU')
        return (2.06, 1.81)
    elif freq > 4.000 and freq < 7.000: # C-band
        return (1.91, 1.75)
    # should be some more for general purposes. But not for RadioAstron
    elif freq > 18 and freq < 26: # K-band
        return (1.74, 1.74)  # corrected for atmosphere
    elif freq < 1.0 :   # P-band
         return (1e-3, 1e-3)  # no data for P-band
    else:
        return (np.nan, np.nan)
    
def POLY(freq, el):
    ''' calculate poly given frequency and elevation'''
    el = 90-el   # should be zenith angle instead of the elevation.     According to https://safe.nrao.edu/wiki/bin/view/GB/Observing/GainPerformance
    if freq > 1.400 and freq < 2.000: # L-band
#        print('POLY value is {}'.format( 0.96575 + el * 1.8256e-03 + el**2 * (-2.4477e-05)))
        return 0.96575 + el * 1.8256e-03 + el**2 * (-2.4477e-05)
    elif freq > 4.000 and freq < 7.000: # C-band
        return 0.93999 + el * 3.4857e-03 + el**2 * (-5.0846e-05)
    elif freq > 18 and freq < 26: # K-band
        return 0.910 + el * 0.00434 + el**2 * (-5.22e-5)
    elif freq < 1.0:    # P-band
        return 1.0     # no data for P-band
    else:
        return np.nan

def readCODES():
    '''read a file with GBT-RadioAstron codes '''
        
    c = pd.read_csv('gbt2ra_codes.txt', delimiter = '\s+',  engine='python')
    c.rename(columns = lambda x: x[1:] if x[0]=='#' else x, inplace=True)
    return c



def clean_anyband(df, do_plots = False, do_prints = True, band = 'l'):
    """Clean ANY-band data, namely for: 
        + large offsets from the center
        + large difference between RCP and LCP
        + unnatural amplitude (<0, >200)
        + size is much larger or much smaller than the beam
        
        :df:        dataframe with single frequency data
        
        :do_plots:  boolean. If True, make a bunch of plots for all 
                    stages of filtering
        
        :do_prints: boolean. If True(default), print out some useful info 
                    on what happens
                
        :band:      observing band. Namely one of ['l', 'k']
        
        :return:    df with suspicious data flagged => flag = 1
    """
#    dl=df[(df.ObsFreq > 1) & (df.ObsFreq < 2)]  # select only L-band (1.6 GHz) for ease of usage
    # I want simply return the cleaned dataframe. So let require a single-band dataframe to be provided
    dl = df
    
    # define cutoff values that are specific for L-band
    # beam FWHM is 18 cm / 100 m in arcseconds
    BEAM = 18/10000 * 206265 
    BEAM = np.median(dl.Width)  # a better way to get the beam width directly from data
    # maximal offset of the Gaussian center from 0 in BEAMs
    MAX_OFFSET = 1 
    MAX_FLUX = 200
    # min and max sizes in units of the beam
    MIN_WIDTH = 0.7
    MAX_WIDTH = 1.3
    # min and max ratio of RCP FLUX to LCP FLUX. 
    MIN_FLUX_RATIO = 0.5
    MAX_FLUX_RATIO = 1 / MIN_FLUX_RATIO
    
    
    N_original = dl.index.size  # number of points in the original dataset at this frequency
    
    # flags to perform different steps
    do_offset = True
    do_flux = True
    do_width = True
    do_flux_ratio = True
    
    if do_prints:
        print("\n\n")
        print("Start cleaning {}-band data".format(band.upper()))
        print("initially there are {} data points".format(dl.index.size))
    
    if do_plots and False:
        # original data, FLUX(time)
        fig, ax = plt.subplots(1,1)
        ax.plot(dl.MJDate, dl.FLUX, 'ob', label = 'FLUX')
        ax.set_xlabel('MJD')
        ax.set_ylabel('FLUX')
        fig.suptitle('Original data')
        fig.legend()
    
    
    
    # 
    # STEP 1. Remove points with large offset from 0. Should not be the case for the last scans.
    #
    if do_prints:
        print("__1 : Removing points with offset from the center > {} BEAMs ({:.1f} arcsec)".format(MAX_OFFSET, MAX_OFFSET * BEAM))
        print("    Starting with {} points".format(dl.index.size))
    
    if do_plots:
        # original data, OFFSET(time)
        fig, ax = plt.subplots(1,1)
        ax.plot(dl.MJDate, dl.Offset, 'ob', label = 'Offset')
        ax.set_xlabel('MJD')
        ax.set_ylabel('Offset (arcsec)')
        fig.suptitle('Original {}-band data for STEP 1, Offset'.format(band.upper()))
        fig.legend()
    
    if do_offset: 
        N_deleted_offset = dl[dl.Offset.abs() > MAX_OFFSET * BEAM].index.size
        dl = dl[dl.Offset.abs() <= MAX_OFFSET * BEAM]
        if do_prints:
            print("    Deleted {} of original {} points".format(N_deleted_offset, N_original))
        
    if do_plots:
        # cleaned data, OFFSET(time)
        fig, ax = plt.subplots(1,1)
        ax.plot(dl.MJDate, dl.Offset, 'ob', label = 'Offset')
        ax.set_xlabel('MJD')
        ax.set_ylabel('Offset (arcsec)')
        fig.suptitle('STEP 1 (Offset cleaned {}-band), Offset'.format(band.upper()))
        fig.legend()
    
    
    #
    # STEP 2. Remove points with unnatural FLUX
    #
    
    if do_prints:
        print("__2 : Removing points with unnatural Fluxes: either < 0 or > {}".format(MAX_FLUX))
        print("    Starting with {} points".format(dl.index.size))

    if do_plots:
        # original data, FLUX(time)
        fig, ax = plt.subplots(1,1)
        ax.plot(dl.MJDate, dl.FLUX, 'ob', label = 'FLUX')
        ax.set_xlabel('MJD')
        ax.set_ylabel('FLUX (Jy)')
        fig.suptitle('Original {}-band data for STEP 2, FLUX'.format(band.upper()))
        fig.legend()
    
    
    if do_flux: 
        N_original = dl.index.size
        N_deleted_flux = dl[(dl.FLUX > MAX_FLUX) | (dl.FLUX <= 0)].index.size
        dl = dl[(dl.FLUX < MAX_FLUX) & (dl.FLUX > 0)]
        if do_prints:
            print("    Deleted {} of original {} points".format(N_deleted_flux, N_original))
        

    if do_plots:
        # cleaned data, FLUX(time)
        fig, ax = plt.subplots(1,1)
        ax.plot(dl.MJDate, dl.FLUX, 'ob', label = 'FLUX')
        ax.set_xlabel('MJD')
        ax.set_ylabel('FLUX (Jy)')
        fig.suptitle('STEP 2 (FLUX cleaned {}-band), FLUX'.format(band.upper()))
        fig.legend()



    #
    # STEP 3. Remove points with too large or too small size
    #
    
    if do_prints:
        print("__3 : Removing points with width < {} BEAMs or > {} BEAMs".format(MIN_WIDTH, MAX_WIDTH))
        print("    Starting with {} points".format(dl.index.size))

    if do_plots:
        # original data, Width(time)
        fig, ax = plt.subplots(1,1)
        ax.plot(dl.MJDate, dl.Width, 'ob', label = 'Width')
        ax.set_xlabel('MJD')
        ax.set_ylabel('Width (arcsec)')
        fig.suptitle('Original {}-band data for STEP 3, Width'.format(band.upper()))
        fig.legend()

    if do_width:
        N_original = dl.index.size
        N_deleted_width = dl[(dl.Width < MIN_WIDTH * BEAM) | (dl.Width > MAX_WIDTH * BEAM)].index.size
        dl = dl[(dl.Width >= MIN_WIDTH * BEAM) & (dl.Width <= MAX_WIDTH * BEAM)]
        if do_prints:
            print("    Deleted {} of original {} points".format(N_deleted_width, N_original))
        
    if do_plots:
        # cleaned data, Width(time)
        fig, ax = plt.subplots(1,1)
        ax.plot(dl.MJDate, dl.Width, 'ob', label = 'Width')
        ax.set_xlabel('MJD')
        ax.set_ylabel('Width (arcsec)')
        fig.suptitle('STEP 3 (Width cleaned {}-band) data, Width'.format(band.upper()))
        fig.legend()

    #
    # STEP 4. Remove points with too large difference between RCP and LCP
    #
    
    if do_prints:
        print("__4 : Removing points with Flux(RCP) / Flux(LCP) < {} or > {}".format(MIN_FLUX_RATIO, MAX_FLUX_RATIO))
        print("    Starting with {} points".format(dl.index.size))

    # prepare another dataframe with RCP/LCP flux ratio
    dr = pd.pivot_table(dl, index = 'RACODE', columns = 'Pol' , values = 'FLUX' )
    dr.loc[:, 'ratio'] = dr.loc[:, 'RCP'] / dr.loc[:, 'LCP']  # add ratio RCP/ LCP column
    # Add ratio column back to dl
    dl = dl.merge(dr.loc[:, 'ratio'], left_on = 'RACODE', right_index = True, how = 'left')
    
    
    if do_plots:
        # original data, Flux(RCP) / Flux(LCP) (time)
        fig, ax = plt.subplots(1,1)
        ax.plot(dl.MJDate, dl.ratio, 'ob', label = 'RCP/LCP flux ratio')
        ax.set_xlabel('MJD')
        ax.set_ylabel('RCP/LCP Flux ratio')
        fig.suptitle('Original {}-band data for STEP 4, RCP/LCP Flux ratio'.format(band.upper()))
        fig.legend()
    
    if do_flux_ratio:
        N_original = dl.index.size
        N_deleted_flux_ratio = dl[(dl.ratio < MIN_FLUX_RATIO) | (dl.ratio > MAX_FLUX_RATIO)].index.size
        dl = dl[(dl.ratio >= MIN_FLUX_RATIO) & (dl.ratio <= MAX_FLUX_RATIO)]
        if do_prints:
            print("    Deleted {} of original {} points".format(N_deleted_flux_ratio, N_original))

    if do_plots:
        # original data, Flux(RCP) / Flux(LCP) (time)
        fig, ax = plt.subplots(1,1)
        ax.plot(dl.MJDate, dl.ratio, 'ob', label = 'RCP/LCP flux ratio')
        ax.set_xlabel('MJD')
        ax.set_ylabel('RCP/LCP Flux ratio')
        fig.suptitle('STEP 4 (Flux ratio cleaned {}-band), RCP/LCP Flux ratio'.format(band.upper()))
        fig.legend()     



    return df
    
#%%
    # MAIN
    
#(_, _, filenames) = walk('/home/mikhail/sci/gbt_pointing').next()   


read_errors = {'LCP':0 , 'RCP':0 , 'BOTH':0, 'TOTAL':0}

read_data = 1


try:
    FINAL = pd.read_pickle('fluxes_gbt.pkl')
    read_data = 0
except:
    pass





if read_data:
    
    FINAL = FINAL[0:0]
#    PATH = '/home/mikhail/sci/gbt_pointing/out' # all RadioAstron files
    PATH = '/homes/mlisakov/sci/gbt_pointing/out'  # all RadioAstron files on vlb098
    #PATH = '/home/mikhail/sci/gbt_pointing/rg28' # rg28 files
#    PATH = '/homes/mlisakov/sci/gbt_pointing/rg28'   # rg28 files on vlb098
    
    
    print("Reading data from {}".format(PATH))
    
    
    filenames = os.listdir(PATH)
#    filenames = ['AGBT12B_262_76.out']
    
    c= readCODES()
    c['RA'] = c['RA'].apply(lambda x: x.upper())
    
    for f in filenames:
        if not f.endswith('out'):
            continue
        
        f = PATH + '/' + f
        print('FILENAME = {}'.format(f))
        
        try:
            d = readout(f)
        except:
            d = readout('out/{}'.format(f))
        
        d = filterout(d)
        
        try:
            res = make_res(d)
        except:
            continue
            
        final = res        
    #    final = res[['Source', 'ProjectID' , 'MJDate' , 'Pol', 'FLUX', 'FLUXERR', 'TSys', 'SEFD', 'ObsFreq', ]]
        final = pd.merge(final, c, how = 'left', left_on = 'ProjectID', right_on = 'GBT')
        final.rename(columns={'RA':'RACODE'}, inplace = True)
    #    final = final[['RACODE', 'MJDate', 'Source', 'Pol', 'FLUX', 'FLUXERR', 'TSys', 'SEFD']]
        try:
            FINAL = FINAL.append(final)
        except:
            FINAL = final
           
            
            
    print('Saving FINAL to a pickle file')
    FINAL.to_pickle('fluxes_gbt.pkl')
            
#FINAL = FINAL[FINAL.ObsFreq > 1 ]  # to remove P-band measurements, which are mainly useless for survey and are indeed noisy/

# add flag column
FINAL.loc[:, 'flag'] = 0

FINAL_L = clean_anyband(FINAL[(FINAL.ObsFreq > 1 ) & (FINAL.ObsFreq < 2) ], do_plots= True, band = 'l')
FINAL_K = clean_anyband(FINAL[(FINAL.ObsFreq > 20 )], do_plots= True, band = 'k')


#F= FINAL
#F=FINAL[FINAL.ObsFreq > 20]
#F = FINAL[(FINAL.ObsFreq > 2) & (FINAL.ObsFreq < 10)]
#F=FINAL[FINAL.ObsFreq < 2]


#F = F[F.GBT == 'AGBT14B_502_76']
'''
fig,ax = plt.subplots(1,1)
#ax.errorbar(F.MJDate, F.FLUX, F.FLUXERR, fmt = 'x' )
#ax.errorbar(F.MJDate, F.Offset, F.OffsetError, fmt = 'x' )
#ax.plot(F.MJDate, F.FLUX, 'o')
ax.plot(F.MJDate, F.Offset, 'o')
#ax.set_ylabel('Flux density [Jy]')
ax.set_xlabel('MJD')
'''



'''
# read in GBT corr stats
GBTFILE = '/homes/mlisakov/sci/gbt_pointing/GBT_baselines.txt'
df = read_ragbt(GBTFILE)
dra, dg = reduce_gbt(df)
'''




