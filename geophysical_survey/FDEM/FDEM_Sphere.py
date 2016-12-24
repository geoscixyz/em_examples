import time
import numpy as np
import scipy as sp
from scipy.constants import mu_0
import matplotlib.pyplot as plt
import re
from matplotlib import animation
from JSAnimation import HTMLWriter

from SimPEG import Mesh, Utils, mkvc


"""
    FDEM Sphere
    ===========

    Setup for a forward and inverse for a sphere in a halfspace.
    Calculations are done with e3d_tiled code
    Created by @fourndo

"""
# Conductivity model [halfspace,sphere]
sig = [1e-2, 1e-0]
air = 1e-8
mfile = 'Sphere_Tensor.con'

vmin = np.log10(sig[0])
vmax = np.log10(3e-2)
# Location and radisu of the sphere
loc = [0, 0, -50.]
radi = 30.

# Survey parameters
rx_height = 20.
rx_offset = 8.
freqs = [1000, 2100, 5000, 12000, 26000, 60000, 100000]
nfreqs = len(freqs)

# Number of stations along x-y
nstn = 11
xlim = 150

# First we need to create a mesh and a model.
# This is our mesh
dx = 2.
nC = 30
npad = 12

# Floor uncertainties for e3D inversion
floor = 100

hxind = [(dx, npad, -1.3), (dx, 2*nC), (dx, npad, 1.3)]
hyind = [(dx, 2*nC)]
hzind = [(dx, 3*nC)]

mesh = Mesh.TensorMesh([hxind, hyind, hzind], 'CC0')
mesh._x0[2] = -np.sum(mesh.hz[:(2*nC)])

# Set background conductivity
model = np.ones(mesh.nC) * sig[0]

# Sphere anomaly
ind = Utils.ModelBuilder.getIndicesSphere(loc, radi, mesh.gridCC)
model[ind] = sig[1]

# Create quick topo
xtopo, ytopo = np.meshgrid(mesh.vectorNx, mesh.vectorNy)
ztopo = np.zeros_like(xtopo)

topo = np.c_[mkvc(xtopo), mkvc(ytopo), mkvc(ztopo)]

# Write out topo file
fid = open('Topo.topo', 'w')
fid.write(str(topo.shape[0])+'\n')
np.savetxt(fid, topo, fmt='%f', delimiter=' ', newline='\n')
fid.close()

ind = Utils.surface2ind_topo(mesh, topo, gridLoc='N')

# Change aircells
model[ind == 0] = air

# Write out model
Mesh.TensorMesh.writeUBC(mesh,'Mesh.msh')
Mesh.TensorMesh.writeModelUBC(mesh,mfile,model)

# Create survey locs centered about origin and write to file
locx, locy = np.meshgrid(np.linspace(-xlim,xlim,nstn), np.linspace(-xlim,xlim,nstn))
#locx += dx/2.
#locy += dx/2.
locz = np.ones_like(locx) * rx_height + dx/2.

rxLoc = np.c_[mkvc(locx),mkvc(locy),mkvc(locz)]

# Create a plane of observation through the grid for display
#pLocx, pLocz = np.meshgrid(np.linspace(-100,100,100), np.linspace(-50,50,50))
#pLocy = np.ones_like(pLocx) * mesh.vectorCCy[nC/2]
#
#pLocs = np.c_[mkvc(pLocx),mkvc(pLocy),mkvc(pLocz)]

# Write out topo file
fid = open('XYZ.loc','w')
np.savetxt(fid, rxLoc, fmt='%f',delimiter=' ',newline='\n')
fid.close()


fid = open('E3D_Obs.loc', 'w')
fid.write('! Export from FDEM_Sphere.py\n')
fid.write('IGNORE NaN\n')

fid.write('N_TRX ' + str(rxLoc.shape[0]*len(freqs)) + '\n\n')
for ii in range(rxLoc.shape[0]):

    txLoc = rxLoc[ii,:].copy()
    txLoc[0] -= rx_offset/2.

    for freq in freqs:

        fid.write('TRX_LOOP\n')
        np.savetxt(fid, np.r_[txLoc, 1., 0, 0].reshape((1,6)), fmt='%f',delimiter=' ',newline='\n')
        fid.write('FREQUENCY ' + str(freq) + '\n')
        fid.write('N_RECV 1\n')

        xloc =  rxLoc[ii,0] + rx_offset/2.

        np.savetxt(fid, np.r_[xloc,rxLoc[ii,1:], np.ones(24)*np.nan].reshape((1,27)), fmt='%e',delimiter=' ',newline='\n\n')

fid.close()

#%% WRITE AEM CODE FILES
#fid = open('AEM_data.obs','w')
#count_tx = 0
#
#
#
#for ii in range(rxLoc.shape[0]):
#
#    count_tx += 1
#    count_fq = 0
#    for freq in freqs:
#
#        count_fq += 1
#
#        fid.write('%i %i %i ' % (count_tx, count_fq, 1))
#        np.savetxt(fid, np.ones((1,5))*-99, fmt='%f',delimiter=' ',newline='\n')
#
#fid.close()
#
#def loop(cnter, r, nseg):
#
#    theta = np.linspace(0,2*np.pi,nseg)
#    xx = cnter[0] + r*np.cos(theta)
#    yy = cnter[1] + r*np.sin(theta)
#    zz = cnter[2] * np.ones_like(xx)
#
#    loc = np.c_[xx, yy, zz]
#    return loc
#
## Write tx file
#fid = open('AEM_tx.dat','w')
#count_tx = 0
#nseg = 9
#
#for ii in range(rxLoc.shape[0]):
#
#    count_tx += 1
#    txLoc = rxLoc[ii,:].copy()
#    txLoc[0] -= rx_offset/2.
#
#    loc = loop(txLoc,1.,nseg)
#
#    np.savetxt(fid, np.c_[count_tx, nseg, 1], fmt='%i',delimiter=' ',newline='\n')
#
#    for jj in range(nseg)    :
#
#         np.savetxt(fid, loc[jj,:].reshape((1,3)), fmt='%f',delimiter=' ',newline='\n')
#
#fid.close()
#
## Write rx file
#fid = open('AEM_rx.dat','w')
#count_tx = 0
#
#
#for ii in range(rxLoc.shape[0]):
#
#    count_tx += 1
#    txLoc = rxLoc[ii,:].copy()
#    txLoc[0] += rx_offset/2.
#
#    loc = loop(txLoc,1.,9)
#
#    np.savetxt(fid, np.c_[count_tx, nseg, 1], fmt='%i',delimiter=' ',newline='\n')
#
#    for jj in range(nseg):
#
#         np.savetxt(fid, loc[jj,:].reshape((1,3)), fmt='%f',delimiter=' ',newline='\n')
#
#fid.close()
#
## Write rx file
#fid = open('AEM_freq.dat','w')
#count_fq = 0
#
#for freq in freqs:
#
#    count_fq +=1
#
#    np.savetxt(fid, np.c_[count_fq, freq], fmt='%f',delimiter=' ',newline='\n')
#
#fid.close()
#%% Read in e3d pred file
def read_e3d_pred(predFile):

    sfile = open(predFile,'r')
    lines = sfile.readlines()

    obs = np.zeros((rxLoc.shape[0]*len(freqs),19))
    count = -1
    ii = -1
    for line in lines:
        count += 1

        if re.match('FREQUENCY',line):
            freq = float(re.split('\s+',line)[1])


        if re.match('N_RECV',line):
            ii += 1
            obs[ii,:] = np.r_[freq,[float(x) for x in re.findall("-?\d+.?\d*(?:[Ee]-\d+)?",lines[count+1])]]

    fid.close()
    return np.asarray(obs)

#%% Load 1D true and inverted model
m1D = Mesh.TensorMesh.readModelUBC(mesh,'EM1D_iter2.dat')
fig3 = plt.figure(figsize=(8,8))

X, Z = mesh.gridCC[:,0].reshape(mesh.vnC, order="F"), mesh.gridCC[:,2].reshape(mesh.vnC, order="F")
Y = mesh.gridCC[:,1].reshape(mesh.vnC, order="F")

axs = plt.subplot(2,1,1)
temp = np.log10(model)
temp[temp==-8] = np.nan
temp = temp.reshape(mesh.vnC, order='F')

ptemp = temp[:,nC,:].T
ps = plt.contourf(X[:,nC,:].T,Z[:,nC,:].T,ptemp,20,vmin=vmin, vmax=vmax, clim=[vmin,vmax], cmap='RdBu_r')
plt.contour(X[:,nC,:].T,Z[:,nC,:].T,ptemp,1, colors='k', linestyles='solid')

plt.scatter(mkvc(locx),mkvc(locz),c='r')
axs.set_title('')
axs.set_aspect('equal')
axs.set_xlim([-xlim-10,xlim+10])
axs.set_ylim([-100,30])
axs.set_xticklabels([])

#Plot recovered model
axs = plt.subplot(2,1,2)
temp = np.log10(m1D)
temp[temp==-8] = np.nan
temp = temp.reshape(mesh.vnC, order='F')

ptemp = temp[:,nC,:].T
plt.contourf(X[:,nC,:].T,Z[:,nC,:].T,ptemp,20,vmin=vmin, vmax=vmax, clim=[vmin,vmax], cmap='RdBu_r')
plt.contour(X[:,nC,:].T,Z[:,nC,:].T,ptemp,5, colors='k',linestyles='solid')

plt.scatter(mkvc(locx),mkvc(locz),c='r')
axs.set_title('Recovered 1D model')
axs.set_aspect('equal')
axs.set_xlim([-xlim-10,xlim+10])
axs.set_ylim([-100,30])
axs.set_xlabel('Easting (m)')
axs.set_ylabel('Elevation (m)')
pos =  axs.get_position()

axs.set_position([pos.x0, pos.y0+.05,  pos.width, pos.height])
cbs = fig3.add_axes([pos.x0 + .3, pos.y0-0.04,  pos.width*0.25, pos.height*0.05])
plt.colorbar(ps,orientation="horizontal",ticks=np.linspace(vmin,vmax, 3), format="$10^{%.1f}$", cmap ='RdBu_r', cax = cbs, ax=axs, label=' S/m ')
plt.show()

plt.savefig('FEM_1D_Model.png')
#%%
dpred = read_e3d_pred('E3D\\Sphere_dpred0.txt')
dprim = read_e3d_pred('E3D\\WholeSpace_dpred0.txt')

# Adjust primary field and convert to ppm
H0true = 0.00017543534291434681*np.pi
dH0 =  dprim[:,-2] - H0true

Hs_R =  (dpred[:,-2] - dprim[:,-2])/ (H0true) * 1e+6
Hs_I = (dpred[:,-1])/(H0true)* 1e+6

# LOAD DATA FROM CYL CODE (THANKS Lindsey)
cylData  = np.loadtxt('cyl_sphere.txt')

# Plot profile and sounding
fig = plt.figure(figsize=(8,8))
ax1 = plt.subplot(2,1,1)
ax2 = plt.subplot(2,1,2)
pos =  ax2.get_position()

#def animate(jj):
#
#    removeFrame2()
#
#    global ax1, ax2, fig

ax1 = plt.subplot(2,1,1)
ax2 = plt.subplot(2,1,2)
pos =  ax1.get_position()
ax1.set_position([pos.x0, pos.y0+0.025,  pos.width, pos.height])

R_stn = np.zeros(len(freqs))
I_stn = np.zeros(len(freqs))
for ii in range(len(freqs)):

    indx = dpred[:,0] == freqs[ii]
    sub_R = np.abs(Hs_R[indx])
    sub_I = np.abs(Hs_I[indx])

    #Create a line of data for profile
    yy = np.linspace(locx[0,:].min(),locx[0,:].max(),nstn*2)
    xx = np.ones_like(yy) * np.mean(locy[:,0])

    R_grid = sp.interpolate.griddata(np.c_[mkvc(locx),mkvc(locy)],
                                     np.log10(sub_R+1.), (xx, yy), method='cubic')

    I_grid = sp.interpolate.griddata(np.c_[mkvc(locx),mkvc(locy)],
                                     np.log10(sub_I+1.), (xx, yy), method='cubic')

    ax1.plot(yy,10**R_grid-1., c='r', lw=np.sqrt(ii+1))    
    ax1.plot(yy,10**I_grid-1.,c='b',ls='--', lw=np.sqrt(ii+1))
    
    ax1.text(yy[nstn],(10**R_grid[nstn])-1.,str(freqs[ii]) + ' Hz',
             bbox={'facecolor':'white',  'pad':1},
             horizontalalignment='left', verticalalignment='center')

    R_stn[ii] = 10.**R_grid[nstn] - 1.
    I_stn[ii] = 10.**I_grid[nstn] - 1.

#    ax1.legend(['I{$H_z$}(-)','R{$H_z$}(+)'], loc=8)



ax1.plot([yy[nstn-1],yy[nstn-1]],
         [np.min(R_stn),np.max(R_stn)],
         c='k', lw=2)

ax1.text(yy[nstn-1],np.mean(R_stn),'Sounding',
         bbox={'facecolor':'white',  'pad':1},
         horizontalalignment='center', verticalalignment='center', rotation = 90.)

ax1.set_xlim([-xlim,xlim])
ax1.set_ylim([1e-1,5e+2])
ax1.grid(True)
ax1.set_xlabel('x (m)')
ax1.set_ylabel('|$H_z$| (ppm)')

ax2.semilogx(cylData[:,0],-cylData[:,1]/mu_0/H0true*1.4e+6, c='r', lw=3)
ax2.semilogx(cylData[:,0],-cylData[:,2]/mu_0/H0true*1.4e+6,c='b',ls='--', lw=3)

ax1.set_title('Profile')
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('|$H_z$| (ppm)')
ax2.legend(['R{$H_z$}(+)','I{$H_z$}(-)'], loc =2)
ax2.set_title('Sounding')
ax2.grid(True)

plt.savefig('FEM_Profile_Sounding.png')

#def removeFrame2():
#    global ax1, ax2, fig
#    fig.delaxes(ax1)
#    fig.delaxes(ax2)
#    plt.draw()
#
#anim = animation.FuncAnimation(fig, animate,
#                               frames=1 , interval=1000, repeat = False)
##/home/dominiquef/3796_AGIC_Research/DCIP3D/MtISa
#anim.save('Freq_slice.html', writer=HTMLWriter(embed_frames=True,fps=1))


#%% Plot animation
fig2 = plt.figure(figsize=(8,9))
ax1 = plt.subplot(2,2,1)
ax2 = plt.subplot(2,2,2)
ax3 = plt.subplot(2,1,2)
pos =  ax3.get_position()
ax4 = fig2.add_axes([pos.x0, pos.y0+0.1,  pos.width, pos.height*0.5])
#cb3 = fig.add_axes([pos.x0, pos.y0+0.1,  pos.width, pos.height])

temp = np.log10(model)
temp[temp==-8] = np.nan
temp = temp.reshape(mesh.vnC, order='F')

ptemp = temp[:,nC,:].T

def animate(ii):

    global ax1, ax2, ax3, ax4, fig2
    removeFrame()

    indx = dpred[:,0] == freqs[ii]
    sub_R = np.abs(Hs_R[indx])
    sub_I = np.abs(Hs_I[indx])

    #Re-grid the data for visual
    xx = locx[0,:]
    yy = locy[:,0]

    x, y = np.meshgrid(np.linspace(xx.min(),xx.max(),50),np.linspace(yy.min(),yy.max(),50))

    R_grid = sp.interpolate.griddata(np.c_[mkvc(locx),mkvc(locy)],
                                     (sub_R), (x, y), method='cubic')


    I_grid = sp.interpolate.griddata(np.c_[mkvc(locx),mkvc(locy)],
                                     ((sub_I)), (x, y), method='cubic')

    ax1 = plt.subplot(2,2,1)

    vminD, vmaxD = np.floor(R_grid.min()-10), np.ceil(R_grid.max()+10)
    im1 = plt.contourf(x[0,:],y[:,0],R_grid,25, clim=(vminD,vmaxD), vminD=vminD, vmaxD=vmaxD, cmap='jet')
    plt.clim(vminD,vmaxD)
    plt.scatter(mkvc(locx),mkvc(locy),c='k')
    ax1.set_xticks([-xlim,0,xlim])
    ax1.set_yticks([-xlim,0,xlim])
    ax1.set_aspect('equal')
    ax1.set_ylabel('Northing (m)')
    plt.title('Real{$H_z$}: ' + str(freqs[ii]/1000) + ' kHz')
    pos =  ax1.get_position()
    ax1.set_position([pos.x0+0.05, pos.y0+0.175,  pos.width*0.75, pos.height*0.75])
    cb1 = fig2.add_axes([pos.x0+0.1, pos.y0+0.1,  pos.width*0.5, pos.height*0.05])
    plt.colorbar(im1,orientation="horizontal",
                 ticks=np.round(np.linspace(R_grid.min(),R_grid.max(), 3)),
                 cax = cb1, label='ppm')
    plt.show()

    ax2 = plt.subplot(2,2,2)
    vminD, vmaxD = np.floor(I_grid.min()-10), np.ceil(I_grid.max()+10)
    im2 = plt.contourf(x[0,:],y[:,0],I_grid,25, clim=(vminD,vmaxD), vmin=vminD, vmax=vmaxD, cmap='jet')
    plt.clim(vminD,vmaxD)
    plt.scatter(mkvc(locx),mkvc(locy),c='k')
    ax2.set_aspect('equal')
    ax2.set_xticks([-xlim,0,xlim])
    ax2.set_yticks([-xlim,0,xlim])
    ax2.set_yticklabels([])
    plt.title('Imag{$H_z$}: ' + str(freqs[ii]/1000) + ' kHz')
    pos =  ax2.get_position()
    ax2.set_position([pos.x0, pos.y0+0.175,  pos.width*0.75, pos.height*0.75])
    cb2 = fig2.add_axes([pos.x0+0.05, pos.y0+0.1,  pos.width*0.5, pos.height*0.05])
    plt.colorbar(im2,orientation="horizontal",
                 ticks=np.round(np.linspace(I_grid.min(),I_grid.max(), 3)),
                 cax = cb2, label='ppm')
    plt.show()

    ax3 = plt.subplot(2,1,2)
#    ps = mesh.plotSlice(np.log10(model),normal = 'Y', ind=nC, ax=ax3, pcolorOpts={'cmap':'RdBu_r'})

    ax3.contourf(X[:,nC,:].T,Z[:,nC,:].T,ptemp,20,vmin=vmin, vmax=vmax, clim=(vmin,vmax), cmap='jet')
    plt.contour(X[:,nC,:].T,Z[:,nC,:].T,ptemp,1, clim=(vmin,vmax), colors='k', linestyles='solid')
    plt.scatter(mkvc(locx),mkvc(locz),c='r')
    ax3.set_title('')
    ax3.set_aspect('equal')
    ax3.set_xlim([-xlim-10,xlim+10])
    ax3.set_ylim([-100,30])
    plt.show()
#    pos =  ax3.get_position()
#    cb3 = fig.add_axes([pos.x0+0.5, pos.y0+0.05,  pos.width*0.25, pos.height*0.05])
#    plt.colorbar(ps[0],orientation="horizontal",ticks=np.linspace(np.log10(air),np.log10(sig[1]), 3), format="$10^{%.1f}$", cax = cb3, label=' S/m ')
#
    ax3.text(loc[0],loc[2],"$10^{0}$" + ' S/m ',
                 horizontalalignment='center', verticalalignment='center', color='w')

    ax3.text(loc[0]-50,loc[1]-50,"$10^{-2}$" + ' S/m ',
                 horizontalalignment='center', verticalalignment='top', color='w')

    #Create a line of data for profile
    yy = np.linspace(locx[0,:].min(),locx[0,:].max(),nstn*10+1)
    xx = np.ones_like(yy) * np.mean(locy[:,0])

    R_grid = sp.interpolate.griddata(np.c_[mkvc(locx),mkvc(locy)],
                                     np.log10(sub_R+1.), (xx, yy), method='cubic')

    I_grid = sp.interpolate.griddata(np.c_[mkvc(locx),mkvc(locy)],
                                     np.log10(sub_I+1.), (xx, yy), method='cubic')



    pos =  ax3.get_position()
    ax3.set_position([pos.x0, pos.y0-0.05,  pos.width, pos.height])
    ax3.set_xlabel('Easting (m)')
    ax4 = fig2.add_axes([pos.x0, pos.y0+0.3,  pos.width, pos.height*0.75])

    ax4.plot(yy,10**R_grid-1., c='r')
    ax4.plot(yy,10**I_grid-1.,c='b',ls='--')
    

#    ax4.semilogy(yy,R_grid,c='r', lw=2)
#    ax4.semilogy(yy,np.abs(I_grid),c='b',ls='--', lw=2)
    ax4.set_xlim([-xlim-10,xlim+10])
    ax4.set_ylim([25,5e+3])
    plt.yscale('log')
    ax4.grid(True)
    ax4.set_ylabel('|$H_z$|')
#    ax4.set_ylim([np.min(np.c_[R_grid,np.abs(I_grid)])*0.5,np.max(np.c_[R_grid,np.abs(I_grid)])*1.5])
    ax4.legend(['R{$H_z$}','I{$H_z$}',])
#    ax4.set_title('Profile')
    ax4.set_xticklabels([])

def removeFrame():
    global ax1, ax2, ax3, ax4, fig2
    fig2.delaxes(ax1)
    fig2.delaxes(ax2)
    fig2.delaxes(ax3)
    fig2.delaxes(ax4)
#    fig.delaxes(cb3)
#    fig.delaxes(cb2)
    plt.draw()

anim = animation.FuncAnimation(fig2, animate,
                               frames=nfreqs , interval=1000, repeat = False)
#/home/dominiquef/3796_AGIC_Research/DCIP3D/MtISa
anim.save('Data_slice.html', writer=HTMLWriter(embed_frames=True,fps=1))



#%% Load 1D data and plot obs vs pred

#%% Plot animation

dobs = read_e3d_pred('E3D\\Sphere_dpred0.txt')#'E3D\\Sphere_ResisBack_dpred0.txt')
dpred = np.loadtxt('Inv_PRED_iter2.pre')
dprim = read_e3d_pred('E3D\\WholeSpace_dpred0.txt')

# Adjust primary field and convert to ppm
H0true = 0.00017543534291434681*np.pi
dH0 =  dprim[:,-2] - H0true

Hs_R =  (dobs[:,-2] - dprim[:,-2])/ (H0true) * 1e+6
Hs_I = (dobs[:,-1])/(H0true)* 1e+6

Hs_R_pred =  dpred[:,-4]
Hs_I_pred = dpred[:,-3]


fig2 = plt.figure(figsize=(8,12))
ax1 = plt.subplot(3,2,1)
ax2 = plt.subplot(3,2,2)
ax3 = plt.subplot(3,2,3)
ax4 = plt.subplot(3,2,4)
ax5 = plt.subplot(3,1,3)
#cb3 = fig.add_axes([pos.x0, pos.y0+0.1,  pos.width, pos.height])

#Plot recovered model
temp = np.log10(m1D)
temp[temp==-8] = np.nan
temp = temp.reshape(mesh.vnC, order='F')

ptemp = temp[:,nC,:].T

def animate(ii):

    global ax1, ax2, ax3, ax4, ax5, fig2
    removeFrame()

    # Grab observed at right frequency
    indx = dobs[:,0] == freqs[ii]
    sub_R = np.abs(Hs_R[indx])
    sub_I = np.abs(Hs_I[indx])
    
    # Grab predicted at right frequency
    indx = dpred[:,3] == freqs[ii]
    sub_R_pred = np.abs(Hs_R_pred[indx])
    sub_I_pred = np.abs(Hs_I_pred[indx])
    
    #Re-grid the data for visual
    xx = locx[0,:]
    yy = locy[:,0]

    x, y = np.meshgrid(np.linspace(xx.min(),xx.max(),50),np.linspace(yy.min(),yy.max(),50))

    R_grid = sp.interpolate.griddata(np.c_[mkvc(locx),mkvc(locy)],
                                     (sub_R), (x, y), method='cubic')


    I_grid = sp.interpolate.griddata(np.c_[mkvc(locx),mkvc(locy)],
                                     ((sub_I)), (x, y), method='cubic')

    ax1 = plt.subplot(3,2,1)

    vminR, vmaxR = np.floor(R_grid.min()-10), np.ceil(R_grid.max()+10)
    im1 = plt.contourf(x[0,:],y[:,0],R_grid,25, clim=(vminR,vmaxR), vmin=vminR, vmax=vmaxR, cmap='jet')
    plt.clim(vminR,vmaxR)
    plt.scatter(mkvc(locx),mkvc(locy),c='k')
    ax1.set_xticks([])
    ax1.set_yticks([-xlim,0,xlim])
    ax1.set_aspect('equal')
    ax1.set_ylabel('Northing (m)')
    plt.title('OBS Real{$H_z$}: ' + str(freqs[ii]/1000) + ' kHz')
    pos =  ax1.get_position()
#    ax1.set_position([pos.x0+0.05, pos.y0+0.15,  pos.width*0.75, pos.height*0.75])
#    cb1 = fig2.add_axes([pos.x0+0.1, pos.y0+0.1,  pos.width*0.5, pos.height*0.05])
#    plt.colorbar(im1,orientation="horizontal",ticks=np.round(np.linspace(R_grid.min(),R_grid.max(), 3)), cax = cb1)
    plt.show()

    ax2 = plt.subplot(3,2,2)
    vminI, vmaxI = np.floor(I_grid.min()-10), np.ceil(I_grid.max()+10)
    im2 = plt.contourf(x[0,:],y[:,0],I_grid,25, clim=(vminI,vmaxI), vmin=vminI, vmax=vmaxI, cmap='jet')
    plt.clim(vminI,vmaxI)
    plt.scatter(mkvc(locx),mkvc(locy),c='k')
    ax2.set_aspect('equal')
    ax2.set_xticks([])
    ax2.set_yticks([-xlim,0,xlim])
    ax2.set_yticklabels([])
    plt.title('OBS Imag{$H_z$}: ' + str(freqs[ii]/1000) + ' kHz')
    pos =  ax2.get_position()
#    ax2.set_position([pos.x0, pos.y0+0.15,  pos.width*0.75, pos.height*0.75])
#    cb2 = fig2.add_axes([pos.x0+0.05, pos.y0+0.1,  pos.width*0.5, pos.height*0.05])
#    plt.colorbar(im2,orientation="horizontal",ticks=np.round(np.linspace(I_grid.min(),I_grid.max(), 3)), cax = cb2)
    plt.show()


    # Grid predicted and plot
    R_grid = sp.interpolate.griddata(np.c_[mkvc(locx),mkvc(locy)],
                                     (sub_R_pred), (x, y), method='cubic')


    I_grid = sp.interpolate.griddata(np.c_[mkvc(locx),mkvc(locy)],
                                     ((sub_I_pred)), (x, y), method='cubic')

    ax3 = plt.subplot(3,2,3)

    im1 = plt.contourf(x[0,:],y[:,0],R_grid,25, clim=(vminR,vmaxR), vmin=vminR, vmax=vmaxR, cmap='jet')
    plt.clim(vminR,vmaxR)
    plt.scatter(mkvc(locx),mkvc(locy),c='k')
    ax3.set_xticks([-xlim,0,xlim])
    ax3.set_yticks([-xlim,0,xlim])
    ax3.set_aspect('equal')
    ax3.set_ylabel('Northing (m)')
    plt.title('PRED Real{$H_z$}')
    pos =  ax3.get_position()
#    ax3.set_position([pos.x0+0.05, pos.y0-0.01,  pos.width*0.75, pos.height*0.75])
    cb1 = fig2.add_axes([pos.x0+0.09, pos.y0-0.03,  pos.width*0.5, pos.height*0.05])
    plt.colorbar(im1,orientation="horizontal",
                 ticks=np.round(np.linspace(R_grid.min(),R_grid.max(), 3)),
                 cax = cb1, label='ppm')
    plt.show()

    ax4 = plt.subplot(3,2,4)
    im2 = plt.contourf(x[0,:],y[:,0],I_grid,25, clim=(vminI,vmaxI), vmin=vminI, vmax=vmaxI, cmap='jet')
    plt.clim(vminI,vmaxI)
    plt.scatter(mkvc(locx),mkvc(locy),c='k')
    ax4.set_aspect('equal')
    ax4.set_xticks([-xlim,0,xlim])
    ax4.set_yticks([-xlim,0,xlim])
    ax4.set_yticklabels([])
    plt.title('PRED Imag{$H_z$}')
    pos =  ax4.get_position()
#    ax4.set_position([pos.x0, pos.y0+0.15,  pos.width*0.75, pos.height*0.75])
    cb2 = fig2.add_axes([pos.x0+0.09, pos.y0-0.03,  pos.width*0.5, pos.height*0.05])
    plt.colorbar(im2,orientation="horizontal",
                 ticks=np.round(np.linspace(I_grid.min(),I_grid.max(), 3)),
                 cax = cb2, label='ppm')
    plt.show()

    ax5 = plt.subplot(3,1,3)
    ps = ax5.contourf(X[:,nC,:].T,Z[:,nC,:].T,ptemp,20,vmin=vmin, vmax=vmax, clim=(vmin,vmax), cmap='jet')
#    plt.contour(X[:,nC,:].T,Z[:,nC,:].T,ptemp,1, clim=(vmin,vmax), cmap='RdBu_r', linestyles='solid')
    plt.scatter(mkvc(locx),mkvc(locz),c='r')
    ax5.set_title('')
    ax5.set_aspect('equal')
    ax5.set_xlim([-xlim-10,xlim+10])
    ax5.set_ylim([-100,30])
    plt.show()
    pos =  ax5.get_position()
    ax5.set_position([pos.x0, pos.y0-0.03,  pos.width, pos.height])
    cb3 = fig2.add_axes([pos.x0, pos.y0-0.05,  pos.width*0.25, pos.height*0.05])
    plt.colorbar(ps,orientation="horizontal",ticks=np.linspace(vmin,vmax, 3), format="$10^{%.1f}$", cax = cb3, label=' S/m ')
    plt.show()
#    ax5.text(loc[0],loc[2],"$10^{0}$" + ' S/m ',
#                 horizontalalignment='center', verticalalignment='center', color='w')
#
#    ax5.text(loc[0]-50,loc[1]-50,"$10^{-6}$" + ' S/m ',
#                 horizontalalignment='center', verticalalignment='top', color='w')


#    pos =  ax5.get_position()
#    ax5.set_position([pos.x0, pos.y0-0.05,  pos.width, pos.height])
    ax5.set_xlabel('Easting (m)')
    ax5.set_title('1D Model')

def removeFrame():
    global ax1, ax2, ax3, ax4, ax5, fig2
    fig2.delaxes(ax1)
    fig2.delaxes(ax2)
    fig2.delaxes(ax3)
    fig2.delaxes(ax4)
    fig2.delaxes(ax5)
#    fig.delaxes(cb3)
#    fig.delaxes(cb2)
    plt.draw()

anim = animation.FuncAnimation(fig2, animate,
                               frames=nfreqs , interval=1000, repeat = False)
#/home/dominiquef/3796_AGIC_Research/DCIP3D/MtISa
anim.save('Inv1D_slice.html', writer=HTMLWriter(embed_frames=True,fps=1))

#%% Export data for inversion
fid = open('E3D_Sphere.obs', 'w')
fid.write('! Export from FDEM_Sphere.py\n')
fid.write('IGNORE nan\n')

fid.write('N_TRX ' + str(rxLoc.shape[0]*len(freqs)) + '\n\n')
count = 0
for ii in range(rxLoc.shape[0]):

    txLoc = rxLoc[ii,:].copy()
    txLoc[0] -= rx_offset

    for freq in freqs:

        uncert = floor*H0true/1e+6

        fid.write('TRX_LOOP\n')
        np.savetxt(fid, np.r_[txLoc, 1., 0, 0].reshape((1,6)), fmt='%f',delimiter=' ',newline='\n')
        fid.write('FREQUENCY ' + str(freq) + '\n')
        fid.write('N_RECV 1\n')

        np.savetxt(fid, np.r_[rxLoc[ii,:], np.ones(20)*np.nan, dobs[count,-2],uncert,dpred[count,-1],uncert ].reshape((1,27)), fmt='%e',delimiter=' ',newline='\n\n')

        count+=1

fid.close()
