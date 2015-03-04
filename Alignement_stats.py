import Alignement_2_3 as Al
import numpy as np
import healpy as hp
from matplotlib import pyplot as plt
import scipy.optimize as op
import sys



#### compute for multipole i and j, n_sim simulations and compare to Smica values
#n_sim = 10
#i = 2
#j = 3



ell_max = 8


map = hp.read_map("dx11_v2_commander_int_cmb_005a_2048.fits")
alm = hp.map2alm(map,lmax=300)

a_ij = np.zeros((ell_max+1,ell_max+1))

for i in range(2,ell_max+1):
    for j in range(i,ell_max+1):
        a_ij[i,j] = Al.get_angle_multipole_ij(alm,i,j,16)
        




