#!/usr/bin/env python
# coding: utf-8

# In[ ]:


try: import clmm
except:
    import notebook_install
    notebook_install.install_clmm_pipeline(upgrade=False)
    import clmm
import matplotlib.pyplot as plt
import numpy as np
import astropy
from astropy import units
import clmm.dataops as da
import clmm.galaxycluster as gc
import clmm.theory as theory
from clmm import Cosmology
from clmm import support
from clmm.support import mock_data as mock
from clmm.support import sampler
from clmm.support.sampler import *
from clmm.support.sampler import fitters
import random
from scipy.interpolate import interp1d

plt.rcParams['font.family']=['gothambook','gotham','gotham-book','serif']


def massfunction(numsamp,lower_limit,upper_limit,nbinhist=100, 
                     numstep=10**4,M_star = 2.e14,b=4*10**(-5)): 
    if lower_limit < 13:
        raise ValueError("The lower_limit is too small and may cause computational issues.")
    if upper_limit > 15:
        raise ValueError("The upper_limit is too high and may cause computational issues.")
    norm = (upper_limit-lower_limit)/nbinhist

    #define x range 1e12-1e15 in num steps
    x = np.logspace(lower_limit,upper_limit,numstep)
    #define y values from the function 
    y = (b*(1/(x/M_star))*np.exp(-(x)/M_star))

    #calculate cumulative sum of the y values (numpy)
    y_cm = np.cumsum(y)
    y_cm = y_cm-min(y_cm)
    y_cm = y_cm/np.max(y_cm)

    #invert x and y variables (flipping)and interpolate that function 
    f = interp1d(y_cm,x , fill_value=(0,1))
    #Generate N random numbers uniformly between 0 & 1: u_i~U(0,1)
    ynew = np.random.random(numsamp)
    #Using the Inverse of the CDF and the values u_i, compute x_i = F^-1(u_i)
    x_samp = f(ynew) 
    
    return x_samp


def model_reduced_tangential_shear_singlez(r,
                                        logm,
                                       z_src,
                                   cluster_z,
                              concentration,
                                          cosmo):
    m = 10.**logm
    gt_model = clmm.compute_reduced_tangential_shear(r,
                                                     m,
                                         concentration,
                                             cluster_z,
                                                 z_src,
                                                 cosmo,
                                        delta_mdef=200,
                              halo_profile_model='nfw')    
    return gt_model

def model_reduced_tangential_shear_zdistrib(radius, 
                                              logm,
                                              data,
                                           catalog,
                                           profile, 
                                         cluster_z,
                                    concentration,cosmo): 
    m = 10**logm
    gt_model = []
    for i in range(len(radius)):
        
        r = profile['radius'][i]
        galist = profile['gal_id'][i]
        
        z_list = catalog.galcat['z'][galist]
        shear = clmm.compute_reduced_tangential_shear(r,
                                                      m,
                                          concentration,
                                              cluster_z,
                                                 z_list, 
                                                  cosmo, 
                                         delta_mdef=200, 
                               halo_profile_model='nfw')
        if len(galist) == 0:
            gt_model.append(1e-16)
            print("this is bad")
        else:
            gt_model.append(np.mean(shear))

    return gt_model

