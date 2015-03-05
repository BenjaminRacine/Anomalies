import Alignement_2_3 as Al
import numpy as np
import healpy as hp
from matplotlib import pyplot as plt
import scipy.optimize as op
import sys




plt.ion()
ell_max = 8

for name in ["commander"]:#,"nilc","smica"]:
    print "name"
    map = hp.read_map("dx11_v2_%s_int_cmb_005a_2048.fits"%name)
    alm = hp.map2alm(map,lmax=300)
    a_ij = np.zeros((ell_max+1,ell_max+1))*np.nan
    for i in range(2,ell_max+1):
        for j in range(i+1,ell_max+1):
            a_ij[i,j] = Al.get_angle_multipole_ij(alm,i,j)
    #print a_ij
    plt.figure()
    plt.xticks(np.arange(1,ell_max))
    plt.yticks(np.arange(1,ell_max))
    plt.imshow(np.abs(np.cos(a_ij)),interpolation = 'nearest',cmap="gist_yarg")
    plt.grid()
    plt.title("%s, (l,b) scalar product"%name)
    plt.colorbar()
