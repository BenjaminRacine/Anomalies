import healpy as hp
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors
import scipy.stats
import scipy.signal
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
    Given a map and a nside, this module shows the ell_max first multipoles (starting at quadrupole), 
    the "axis of maximal power" for each of them, and a map of all directions, 
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



def plot_scal_prod(ell_max,map, name ="Input map"):
    '''
    2D plot of the scalar product of 2 directions
    '''
    c = mcolors.ColorConverter().to_rgb
    rvb = make_colormap([c('red'),c('pink'),0.1, c('pink'),c('WhiteSmoke'),0.3,c('WhiteSmoke'),0.7,c('WhiteSmoke'),(153.0/255,153.0/255,1.),0.9 ,(153.0/255,153.0/255,1.),c('blue')])
    #rvb = make_colormap([c('black'), c('red'),0.1,c('red'),c('white'),0.15 ,c('white'),0.85,c('white'),c('yellow'),0.9,c('yellow'),c('green')])
    alm = hp.map2alm(map,lmax=300)
    a_ij = np.zeros((ell_max-1,ell_max-1))*np.nan
    for i in range(2,ell_max+1):
        for j in range(i+1,ell_max+1):
            a_ij[i-2,j-2] = get_angle_multipole_ij(alm,i,j)
    plt.figure()
    plt.imshow(np.abs(np.cos(a_ij[:,:])),interpolation = 'nearest',cmap=rvb,vmin=0,vmax=1)#,norm=matplotlib.colors.LogNorm())#"gist_yarg")
    plt.xticks(np.arange(0,ell_max-1),np.arange(2,ell_max+1))
    plt.yticks(np.arange(0,ell_max-1),np.arange(2,ell_max+1))
    plt.grid()
    plt.title("%s, (l,b) scalar product"%name)
    plt.colorbar()



def get_average_angle(array_theta_phi):
    '''
    Gives the average position on the sky for a given set of angles 
    '''
    vecs = hp.ang2vec(array_theta_phi[:,0],array_theta_phi[:,1])    
    return hp.vec2ang(vecs.mean(axis=0))


def extend_thetaphi(theta_phi):
    '''
    Intermediate function used to extend a list of angles with their "oposite directions" 
    '''
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






def determine_FD_binning(array_in):
    '''
    returns the optimal binning, according to Freedman-Diaconis rule: see http://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule
    '''
    sorted_arr = np.sort(array_in)
    Q3 = scipy.stats.scoreatpercentile(sorted_arr,3/4.)    
    Q1 = scipy.stats.scoreatpercentile(sorted_arr,1/4.)    
    IQR = Q3-Q1
    bin_size = 2.*IQR*np.size(array_in)**(-1/3.)
    return np.round((sorted_arr[-1] - sorted_arr[0])/bin_size)


def get_stat_histogram(aad_sim,window_len=0,interp=3,check=0,gaussian_smo = 3,smoothing_method = "gaussian"):
    '''
    returns the most likely value, as well as the inf and sup boundary for 68%, and 97%
    '''
    nbin = determine_FD_binning(aad_sim)
    counts,bin = np.histogram(aad_sim,nbin)
    if smoothing_method == "savgol":
        if window_len == 0:
            window_len = round(nbin/6.)*2+1
        smo_counts = scipy.signal.savgol_filter(counts,window_len,interp)
    elif smoothing_method == "gaussian":
        smo_counts = scipy.ndimage.filters.gaussian_filter(counts,gaussian_smo)
    top = min(np.where(smo_counts==max(smo_counts))[0])
    area = np.float(smo_counts[top])/np.size(aad_sim)
    inf_s = top
    sup_s = top
    area_inf=0
    area_sup=0
    endwhile = 0
    while (area < 0.6827) and (endwhile == 0):
        if (sup_s < nbin-1) and (area_sup<=area_inf):
            sup_s = sup_s+1
            area_sup = np.sum(smo_counts[top:sup_s])
        elif (inf_s > 1):
            inf_s = inf_s-1
            area_inf = np.sum(smo_counts[inf_s:top])
        else: 
            print "1 sigma: nbin = %d, inf = %f, sup = %f: not enough stats?"%(nbin,inf_s,sup_s)
            endwhile = 1
        area = np.float(area_sup+area_inf)/np.size(aad_sim)
    inf_2s = inf_s
    sup_2s = sup_s
    endwhile = 0
    while (area < 0.9545) and (endwhile == 0):
        if (sup_2s < nbin-1) and (area_sup<=area_inf):
            sup_2s = sup_2s+1
            area_sup = np.sum(smo_counts[top:sup_2s])
        elif (inf_2s > 0):
            inf_2s = inf_2s-1
            area_inf = np.sum(smo_counts[inf_2s:top])
        else: 
            print "2 sigma: nbin = %d, inf = %f, sup = %f: not enough stats?"%(nbin,inf_2s,sup_2s)
            endwhile = 1
        area = np.float(np.sum(smo_counts[inf_2s:sup_2s]))/np.size(aad_sim)
    bin_mean = (bin[1:]+bin[:-1])/2
    if check == 1:
        plt.plot(bin_mean,counts, label="original histo")
        plt.plot(bin_mean,smo_counts, label="smoothed histo")
        plt.legend(loc="best")
        plt.axvline(bin_mean[top],color = "r")
        plt.axvline(bin_mean[inf_s],color = "b")
        plt.axvline(bin_mean[sup_s],color = "b")
        plt.axvline(bin_mean[inf_2s],color = "g")
        plt.axvline(bin_mean[sup_2s],color = "g")    
    return bin_mean[top],bin_mean[inf_s],bin_mean[sup_s],bin_mean[inf_2s],bin_mean[sup_2s]
    #return top,inf_s,sup_s,inf_2s,sup_2s


def plot_stat_ell_dep_curve(aad_sims,aad_data,unit=180/np.pi):
    '''
    This function plots the l_max dependent average angular distance 
    compared to simulations, with best value, 1sigma and 2sigma error using get_stat_histogram() 
    input : aad_sims = aad of simulations, in radian [aad_sims.shape = (nsim,ell_max-2)]
            unit = factor multiplying the array (default = 180/np.pi)
    '''
    aad_sims = np.array(aad_sims)*unit
    aad_data = np.array(aad_data)*unit
    ell_max = aad_sims.shape[1]+2
    nsim = aad_sims.shape[0]
    if np.size(aad_data) != ell_max - 2:
        print 'problem in the lmax dimension of the sims vs data'
    plt.figure()
    [plt.plot(np.arange(3,ell_max+1),aad_sims[i,:],"y-.",alpha = 0.02) for i in range(nsim)]    
#    plt.plot(np.arange(3,ell_max+1),top,"r-")
    stats = []
    for j in range(0,aad_sims.shape[1]):
        stats.append(get_stat_histogram(aad_sims[:,j]))
    stats = np.array(stats)
    plt.plot(np.arange(3,ell_max+1),stats[:,0],"b--",label="Simulations")
    plt.fill_between(np.arange(3,ell_max+1),stats[:,1],stats[:,2],alpha = 0.2, color="b")
    plt.fill_between(np.arange(3,ell_max+1),stats[:,3],stats[:,4],alpha = 0.2, color="b")
    #plt.plot(np.arange(3,ell_max+1),aad_sims.mean(axis=0),"r-")
    plt.plot(np.arange(3,ell_max+1),aad_data,"ko",label="Data")
    plt.xlabel("$\ell_{\mathrm{max}}$")
    plt.ylabel("Average angle")
    plt.legend(loc="best")


def plot_stat_ell_dep_hist(aad_sims,aad_data,unit=180/np.pi):
    '''
    This function plots the histograms for the a_a_d, varying with l_max.
    input : aad_sims = aad of simulations, in radian [aad_sims.shape = (nsim,ell_max-2)]
            unit = factor multiplying the array (default = 180/np.pi)
    '''
    aad_sims = np.array(aad_sims)*unit
    aad_data = np.array(aad_data)*unit
    ell_max = aad_sims.shape[1]+2
    nsim = aad_sims.shape[0]
    if unit == 180/np.pi:
        angle_un = "Degree"
    elif unit == 1.:
        angle_un = "Radians"
    if np.size(aad_data) != ell_max - 2:
        print 'problem in the lmax dimension of the sims vs data'
    for i,v in enumerate(xrange(aad_sims.shape[1])):
        v+=1
        plt.subplots_adjust(hspace=0.000,wspace = 0.000)
        ww = np.where(aad_sims[:,i]<aad_data[i])[0]
        nbin = determine_FD_binning(aad_sims[:,i])
        ax = plt.subplot(aad_sims.shape[1]/2+aad_sims.shape[1]%2,2,v)
        ax.hist(aad_sims[:,i],nbin,histtype = "step")
        ax.set_xlim(0,1*unit)
        ax.set_xticks([10,20,30,40,50])
        ax.set_xlabel("Average dispersion [%s]"%angle_un)
        ax.axvline(aad_data[i],color="red",label="pval = %.4f"%(np.float(np.size(ww))/nsim))
        ax.legend(frameon=False,loc ="best",fontsize = 'small')
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        #plt.setp(ax.get_xticklabels(), visible=False) 




import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def make_colormap(seq):
    """
    stolen from http://stackoverflow.com/questions/16834861/create-own-colormap-using-matplotlib-and-plot-color-scale
    Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return matplotlib.colors.LinearSegmentedColormap('CustomMap', cdict)


