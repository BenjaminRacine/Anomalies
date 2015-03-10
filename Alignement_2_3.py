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



def amd_map(alm,ell,nside=16):
    '''                                                                                                                                                                       
    compute the angular momentum dispersion corresponding to each pixels direction                                                                                           
    '''
    map_amd = np.zeros(hp.nside2npix(nside))
    for i in range(hp.nside2npix(nside)):
        tp = hp.pix2ang(nside,i)
        alm_temp = alm.copy()
        map_amd[i] = func_to_min(np.array(tp),alm_temp,ell)
    return map_amd



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
    a given (theta,phi), alm and ell. 
    To be given to a minimizer, WARNING: With the angle restrictions, the starting point should not be (0,0), or Powell can get stuck. Better use [np.pi/2,np.pi]
    '''
    ##### After tries on restricting, using modulos and trigo rules, here we force the bounds (might be improved by chosing another minimizer)
    if thetaphi[0]<0:
        return 1e30
    elif thetaphi[0]>np.pi:
        return 1e30
    elif thetaphi[1]<0:
        return 1e30
    elif thetaphi[1]>2*np.pi:
        return 1e30
    else:
        ##### We copy alm since rotate_alm is 'active'
        alm_temp = alm.copy()
        ##### Here we rotate first with psi, ie the first arg, then with theta, in order to bring the chosen (theta,phi) direction to the north pole.
        hp.rotate_alm(alm_temp,-thetaphi[1],-thetaphi[0],0)
        amd = ang_mom_disp(alm_temp,ell)
        return -amd



def get_angle_multipole_ij(alm,ell_i,ell_j):
    '''
    Give the relative angle between 2 maximum amd directions for a given map
    '''
    #############################################  maybe important to randomly start, and not 0,0 #############################################
    alm_i = filtermap(alm,ell_i)
    alm_j = filtermap(alm,ell_j)
    thetaphi_final_i = op.fmin_powell(func_to_min, [np.pi/2,np.pi], args=(alm_i,ell_i),xtol=0.0001, ftol=0.00000001,disp=False)
    thetaphi_final_j = op.fmin_powell(func_to_min, [np.pi/2,np.pi], args=(alm_j,ell_j),xtol=0.0001, ftol=0.00000001,disp=False)
    ##### Here we restrict to the cos(angle_ij) in [0,1] to avoid redundancy
    angle_ij = np.arccos(np.abs(np.cos(get_angle_gal(thetaphi_final_i[0],thetaphi_final_i[1],thetaphi_final_j[0],thetaphi_final_j[1]))))
    return angle_ij


def get_dot_multipole_ij(alm,ell_i,ell_j):
    '''                                                                                                                                                                       
    Give the relative angle between 2 maximum amd directions for a given map                                                                                                  
    '''
    #############################################  maybe important to randomly start, and not 0,0 #############################################                               
    alm_i = filtermap(alm,ell_i)
    alm_j = filtermap(alm,ell_j)
    thetaphi_final_i = op.fmin_powell(func_to_min, [np.pi/2,np.pi], args=(alm_i,ell_i),xtol=0.0001, ftol=0.00000001,disp=False)
    thetaphi_final_j = op.fmin_powell(func_to_min, [np.pi/2,np.pi], args=(alm_j,ell_j),xtol=0.0001, ftol=0.00000001,disp=False)
    vec_i = hp.ang2vec(thetaphi_final_i[0],thetaphi_final_i[1])
    vec_j = hp.ang2vec(thetaphi_final_j[0],thetaphi_final_j[1])
    dot = np.dot(vec_i,vec_j)
    return dot


def plot_first_multipoles(map,nside,ell_max):
    '''
    Given a map and a nside, this module shows the ell_max first multipoles (starting at quadrupole), the "axis of maximal power" for each of them, and a map of all directions, 
    '''
    alm = hp.map2alm(map,lmax=300)
    plt.figure()
    thetaphi_list = []
    map_zeros = np.zeros(12*nside**2)
    for i in range(2,ell_max+1):
        alm_filt = filtermap(alm, i)
        thetaphi = op.fmin_powell(func_to_min, [np.pi/2,np.pi], args=(alm_filt,i),xtol=0.0001, ftol=0.00000001,disp=False)
        pix = hp.ang2pix(nside,thetaphi[0],thetaphi[1])
        pix2 = hp.ang2pix(nside,np.pi-thetaphi[0],thetaphi[1]+np.pi)
        direc = hp.query_disc(nside,hp.pix2vec(nside,pix),0.1)
        direc2 = hp.query_disc(nside,hp.pix2vec(nside,pix2),0.1)
        map_filt = hp.alm2map(alm_filt,nside)
        map_filt[direc]=hp.UNSEEN
        map_filt[direc2]=hp.UNSEEN
        if i < 8:
            fig_nums = plt.get_fignums()
            hp.mollview(map_filt,title = "$\ell = $%d"%(i),sub=int("32%d"%(i-1)))
        elif (i>7) and (i<14):
            plt.figure(max(fig_nums)+1)
            hp.mollview(map_filt,title = "$\ell = $%d"%(i),sub=int("32%d"%(i-7)))
        map_zeros[direc] = i
        map_zeros[direc2] = i
        thetaphi_list.append(thetaphi)
    hp.mollview(map_zeros)


def get_average_angle(array_theta_phi):
    '''
    Gives the average position on the sky for a given set of angles 
    '''
    vecs = hp.ang2vec(array_theta_phi[:,0],array_theta_phi[:,1])    
    return hp.vec2ang(vecs.mean(axis=0))


def extend_thetaphi(theta_phi):
    list_temp = np.copy(theta_phi)
    ll = len(theta_phi)
    list_temp[:,0]=np.pi-list_temp[:,0]
    list_temp[:,1]=np.pi+list_temp[:,1]
    return np.hstack([(theta_phi),(list_temp)])

def compute_mean_disp(array_theta_phi, array_mean):
    '''
    For a given set of angles (and mean position), returns the mean angular distance to the mean. 
    '''
    return np.mean([get_angle_gal(array_theta_phi[i,0],array_theta_phi[i,1],array_mean[0],array_mean[1]) for i in xrange(array_theta_phi.shape[0])])

    
def get_angular_mean_disp(alm_in,ell_max,check=0):
    '''
    for a given alm, gives the angular mean dispersions of all the combinaison of directions of multipoles in [2,ell_max]
    if check==1, plots the less dispersed directions
    '''
    thetaphi_list = []
    num_ell = ell_max + 1 - 2 
    for i in range(2,ell_max + 1):
        alm_filt = filtermap(alm_in, i)
        thetaphi = op.fmin_powell(func_to_min, [np.pi/2,np.pi], args=(alm_filt,i),xtol=0.0001, ftol=0.00000001,disp=False)
        thetaphi_list.append(thetaphi)
    tt = extend_thetaphi(np.array(thetaphi_list))
    tt  = tt.reshape(tt.shape[0],2,2)
    angles = []
    angle_mean = []
    a_a_d = []
    for i in range(2**num_ell):
        angles.append(np.array(tt)[np.arange(num_ell),[int(x) for x in list(('{:0%db}'%tt.shape[0]).format(i))]])
        angle_mean.append(get_average_angle(np.array(tt)[np.arange(num_ell),[int(x) for x in list(('{:0%db}'%num_ell).format(i))]]))
        a_a_d.append(compute_mean_disp(np.array(angles)[i,:,:],np.array(angle_mean)[i]))
    if check == 1:
        i_min = np.min(np.where(a_a_d == min(a_a_d))[0])
        print i_min
        map_0 = np.zeros(12*16**2)
        pix = hp.ang2pix(16,np.array(angles)[i_min,:,0],np.array(angles)[i_min,:,1])
        pix_mean = hp.ang2pix(16,np.array(angle_mean)[i_min,0],np.array(angle_mean)[i_min,1])
        map_0[pix]=1
        map_0[pix_mean] = hp.UNSEEN
        hp.mollview(map_0,title = "aad = %f"%(a_a_d[i_min]))        
    return a_a_d



    
def plot_stat_ell_dep(histo,aad_data):
    '''
    This function plots the histograms for the a_a_d, varying with l_max, 
    as well as the l_max dependent plot
    input : histo.shape = (nsim,ell_max-2)
    '''
    ell_max = histo.shape[1]+2
    nsim = histo.shape[0]
    if np.size(aad_data) != ell_max - 2:
        print 'problem in the lmax dimension of the histograms vs data'
    [plt.plot(np.arange(3,ell_max+1),histo[i,:],"y",alpha = 0.1) for i in range(nsim)]
    plt.plot(np.arange(3,ell_max+1),histo.mean(axis=0),"r-")
    plt.plot(np.arange(3,ell_max+1),aad_data,"ko")
    plt.xlabel("$\ell_{\rm max}$")
    plt.ylabel("Average angle (degree)")
