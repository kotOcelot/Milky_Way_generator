!pip install astroquery
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
import astropy.units as u
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# j = Gaia.launch_job(query='SELECT "I/355/gaiadr3".Plx,  "I/355/gaiadr3".pmRA,  "I/355/gaiadr3".pmDE, "I/355/gaiadr3".RV,  "I/355/gaiadr3".RAJ2000,  "I/355/gaiadr3".DEJ2000, "I/355/gaiadr3".GLAT,  "I/355/gaiadr3".GLON,  "I/355/gaiadr3".RPlx FROM "I/355/gaiadr3" WHERE "I/355/gaiadr3".RPlx>=5 AND "I/355/gaiadr3".RUWE<1.4  AND "I/355/gaiadr3".RV!=0')

# r = j.get_results()

df = pd.read_csv('result(2).csv')

ra = np.array(df.RAJ2000) #deg
dec = np.array(df.DEJ2000) #deg
l = np.array(df.GLON) #deg
b = np.array(df.GLAT) #deg
pr = np.array(df.Plx) #mas/yr
vls = np.array(df.RV) #VR km/s
pmra  = np.array(df.pmRA) #mas/yr *cosdec
pmdec = np.array(df.pmDE) #mas/yr
R0 = 7.5 #kpc
omg0 = 30.0 #km s−1 kpc−1, velocity of rotation at r0
VT0 = omg0*R0 + 12.0 #km s−1
VR0 = 10.0 #km s−1тзю
VZ0 = 7.0 #km s−1,
Z0 = 0.027 #kpc

A = 4.74064 #mas*myr to km s-1
zpcorr = 0.00 #for Gmag > 10
r = 1/(pr - zpcorr)

sc = SkyCoord(ra*u.deg, dec*u.deg, distance = r*u.kpc, pm_ra_cosdec=pmra*u.mas/u.yr, pm_dec=pmdec*u.mas/u.yr)

pmlcosb = sc.galactic.pm_l_cosb.value #mas/yr
pmb = sc.galactic.pm_b.value #mas/yr



def gal_to_cyl(r, l, b, R0,Z0,A, vr, pmlcosb, mb):
  l = np.radians(l)
  b = np.radians(b)
  x = R0 - r*np.cos(l)*np.cos(b)
  y = r*np.sin(l)*np.cos(b)
  z = Z0 + r*np.sin(b)
  R1 = np.sqrt(x**2 + y**2)
  theta = np.arctan2(y, x)
  Vl = A*pmlcosb*r
  Vb = A*mb*r
  alpha = l + theta + math.pi/2
  V_rad = np.sin(alpha)*(vr*np.cos(b) - Vb*np.sin(b)) + Vl*np.cos(alpha) + VT0*np.sin(theta) - VR0*np.cos(theta)
  VT = np.cos(alpha)*(vr*np.cos(b) - Vb*np.sin(b)) - Vl*np.sin(alpha) + VT0*np.cos(theta) + VR0*np.sin(theta)
  VZ = vr*np.sin(b) + Vb*np.cos(b) + VZ0
  return(R1, theta,x, y, z, V_rad, VT, VZ)



R1, theta, x,y,z, V_rad, VT, VZ = gal_to_cyl(r, l, b, R0, Z0, A, vls, pmlcosb, pmb)


size = 150

sum_v = np.zeros((size, size))
vs = np.empty((size, size))
n = np.zeros((size, size))
len_x = 2*max(max(x), abs(min(x)))
len_y = 2*max(max(y), abs(min(y)))
step_x = len_x/size
step_y = len_y/size
xx = np.arange(-len_x/2, len_x/2 , step_x)
yy = np.arange(-len_y/2, len_y/2 , step_y)
for k in range(len(x)):
    if k%100000 == 0:
        print(k)
    ind_x = int(x[k]//step_x + size/2)
    ind_y = int(y[k]//step_y + size/2)
    sum_v[ind_y, ind_x] += VZ[k]
    n[ind_y, ind_x] +=1
    
dispmap = np.zeros((size, size))
for k in range(len(x)):
    if k%100000 == 0:
        print(k)
    ind_x = int(x[k]//step_x + size/2)
    ind_y = int(y[k]//step_y + size/2)
    if n[ind_x, ind_y]!= 0:
        v_mean = sum_v[ind_x, ind_y]/n[ind_x, ind_y]
        dispmap[ind_x, ind_y] += (VZ[k] - v_mean)**2
 
mask = (n==0)
sum_v[mask] = np.nan
n[mask] = np.nan
dispmap[mask] = np.nan
velmap = sum_v/n
disp = np.sqrt(dispmap/n)
np.save('vz_mw', velmap)
np.save('dispz_mw', disp)
