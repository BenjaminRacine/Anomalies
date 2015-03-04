import healpy as hp
import numpy as np
from matplotlib import pyplot as plt
import scipy.optimize as op

def filtermap(alm,ell):
    '''
    For a given alm array, returns this array, where all useless multipoles have been filtered out 
    '''
    alm_temp = np.zeros(hp.Alm.getsize(ell))+0*1j#alm.size)+0*1j
    lmax = hp.Alm.getlmax(alm.size)
    index = hp.Alm.getidx(lmax,ell,np.arange(ell+1))
    index_temp = hp.Alm.getidx(ell,ell,np.arange(ell+1))
    alm_temp[index_temp] = alm[index]
    return alm_temp



def ang_mom_disp(alm,ell):
    '''
    return the angular momentum dispersion of a the alm array for a given ell
    see equation (1) of http://arxiv.org/abs/astro-ph/0307282
    '''
    lmax = hp.Alm.getlmax(alm.size)
    index = hp.Alm.getidx(lmax,ell,np.arange(ell+1))
    est = sum([m**2 * np.absolute(alm[index[m]])**2 for m in range(ell+1)])
    return est




def get_angle_gal(theta1,phi1,theta2,phi2):
    """
    Returns the angle between 2 objects, using http://en.wikipedia.org/wiki/Great-circle_distance. Thanks to Julian.
    Alternative to vectorial product method.
    Inputs:
    - right ascension ra (rad), declination dec (rad)
    Output:
    - Angle in radian
    """
    dec1 = np.pi/2 - theta1
    dec2 = np.pi/2 - theta2
    ra1 = phi1 
    ra2 = phi2 
    ra_diff=ra2-ra1
    A=(np.cos(dec2)*np.sin(ra_diff))**2
    B=(np.cos(dec1)*np.sin(dec2) - np.sin(dec1)*np.cos(dec2)*np.cos(ra_diff))**2
    C=np.sin(dec1)*np.sin(dec2) + np.cos(dec1)*np.cos(dec2)*np.cos(ra_diff)
    angle = np.arctan(np.sqrt(A+B)/C)
    if angle<0:
        angle=np.pi+angle
    return angle



def func_to_min(thetaphi,alm,ell):
    '''                                                                                                                                                                       
    function that gives the amd for:                                                                                                                                          
    a given (theta,phi), alm and ell. To be given to a minimizer                                                                                                              
    '''
    alm_temp = alm.copy()
    thetaphi[0] = thetaphi[0]%(np.pi)
    thetaphi[1] = thetaphi[1]%(2*np.pi)
    hp.rotate_alm(alm_temp,thetaphi[1],thetaphi[0],0)
    amd = ang_mom_disp(alm_temp,ell)
    #print "temp",alm_temp[-1],"alm",alm[-1],-amd,thetaphi
    return -amd



def get_angle_multipole_ij(alm,ell_i,ell_j,nside):
    '''
    Give the relative angle between 2 maximum amd directions for a given map
    '''
    #############################################  maybe important to randomly start, and not 0,0 #############################################
    alm_i = filtermap(alm,ell_i)
    alm_j = filtermap(alm,ell_j)
    thetaphi_final_i = op.fmin_powell(func_to_min, [0,0], args=(alm_i,ell_i),xtol=0.0001, ftol=0.00000001,disp=False)
    thetaphi_final_j = op.fmin_powell(func_to_min, [0,0], args=(alm_j,ell_j),xtol=0.0001, ftol=0.00000001,disp=False)
    angle_ij = get_angle_gal(thetaphi_final_i[0],thetaphi_final_i[1],thetaphi_final_j[0],thetaphi_final_j[1])
    return angle_ij


def get_dot_multipole_ij(alm,ell_i,ell_j,nside):
    '''                                                                                                                                                                       
    Give the relative angle between 2 maximum amd directions for a given map                                                                                                  
    '''
    #############################################  maybe important to randomly start, and not 0,0 #############################################                                
    alm_i = filtermap(alm,ell_i)
    alm_j = filtermap(alm,ell_j)
    thetaphi_final_i = op.fmin_powell(func_to_min, [0,0], args=(alm_i,ell_i),xtol=0.0001, ftol=0.00000001,disp=False)
    thetaphi_final_j = op.fmin_powell(func_to_min, [0,0], args=(alm_j,ell_j),xtol=0.0001, ftol=0.00000001,disp=False)
    vec_i = hp.ang2vec(thetaphi_final_i[0],thetaphi_final_i[1])
    vec_j = hp.ang2vec(thetaphi_final_j[0],thetaphi_final_j[1])
    dot = np.dot(vec_i,vec_j)
    return dot