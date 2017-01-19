# -*- coding: utf-8 -*-
"""
Created on Fri Dec 09 10:43:37 2016

@author: Devin
"""

    # IMPORT PACKAGES
import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter



##############################################
#   PLOTTING FUNCTIONS FOR WIDGETS
##############################################

def fcn_Cosine_Widget(I,a1,a2,xRx,zRx,azm,R,L,f):


    xmin, xmax, dx, zmin, zmax, dz = -20., 20., 0.5, -20., 20., 0.5
    X,Z = np.mgrid[xmin:xmax+dx:dx, zmin:zmax+dz:dz]
    X = np.transpose(X)
    Z = np.transpose(Z)

    Obj = IndEx(I,a1,a2,xRx,zRx,azm,R,L)
    t_range = (4/f)*np.linspace(0,1,num=100)


    Obj.calc_PrimaryLoop()
    Bpx,Bpz,Babs = Obj.calc_PrimaryRegion(X,Z)
    #calc_IndCurrent_Cos_i(self,f,t)
    Ire,Iim,Is,phi = Obj.calc_IndCurrent_cos_range(f,t_range)

    fig1 = plt.figure(figsize=(13,18))
    ax11 = fig1.add_axes([0,0.7,0.52,0.29])
    ax12 = fig1.add_axes([0.57,0.7,0.43,0.29])
    ax21 = fig1.add_axes([0.1,0.35,0.8,0.3])
    ax22 = fig1.add_axes([0.1,0,0.8,0.3])

    ax11,Cplot = Obj.plot_PrimaryRegion(X,Z,Bpx,Bpz,Babs,ax11);
    ax12 = Obj.plot_PrimaryLoop(ax12)
    ax21,ax21b,ax22 = Obj.plot_InducedCurrent_cos(ax21,ax22,Ire,Iim,Is,phi,f,t_range)

    # plt.tight_layout()

    plt.show(fig1)


def fcn_FDEM_Widget(I,a1,a2,xRx,zRx,azm,R,L,f):

    FS = 20

    xmin, xmax, dx, zmin, zmax, dz = -20., 20., 0.5, -20., 20., 0.5
    X,Z = np.mgrid[xmin:xmax+dx:dx, zmin:zmax+dz:dz]
    X = np.transpose(X)
    Z = np.transpose(Z)

    Obj = IndEx(I,a1,a2,xRx,zRx,azm,R,L)

    Obj.calc_PrimaryLoop()
    Bpx,Bpz,Babs = Obj.calc_PrimaryRegion(X,Z)
    EMF,Is = Obj.calc_IndCurrent_FD_spectrum()
    EMFi,Isi = Obj.calc_IndCurrent_FD_i(f)

    fig1 = plt.figure(figsize=(13,12))
    ax11 = fig1.add_axes([0,0.55,0.52,0.43])
    ax12 = fig1.add_axes([0.57,0.54,0.43,0.43])
    ax2  = fig1.add_axes([0,0,1,0.45])

    ax11,Cplot = Obj.plot_PrimaryRegion(X,Z,Bpx,Bpz,Babs,ax11);
    ax12 = Obj.plot_PrimaryLoop(ax12)
    ax2 = Obj.plot_InducedCurrent_FD(ax2,Is,f,EMFi,Isi)

    f_str    = '{:.1e}'.format(f)
    EMF_str  = '{:.1e}j'.format(EMFi.imag)
    ax12.text(-2.9,-1.0,'f = '+f_str+' Hz',fontsize=FS)
    ax12.text(-2.9,-1.4,'EMF = '+EMF_str+' V',fontsize=FS)

    plt.show(fig1)


def fcn_TDEM_Widget(I,a1,a2,xRx,zRx,azm,R,L,t):

    FS = 20

    xmin, xmax, dx, zmin, zmax, dz = -20., 20., 0.5, -20., 20., 0.5
    X,Z = np.mgrid[xmin:xmax+dx:dx, zmin:zmax+dz:dz]
    X = np.transpose(X)
    Z = np.transpose(Z)

    Obj = IndEx(I,a1,a2,xRx,zRx,azm,R,L)

    Obj.calc_PrimaryLoop()
    Bpx,Bpz,Babs = Obj.calc_PrimaryRegion(X,Z)
    V,Is = Obj.calc_IndCurrent_TD_offtime()
    EMFi,Isi = Obj.calc_IndCurrent_TD_i(t)

    fig1 = plt.figure(figsize=(13,12))
    ax11 = fig1.add_axes([0,0.55,0.52,0.43])
    ax12 = fig1.add_axes([0.57,0.54,0.43,0.43])
    ax2  = fig1.add_axes([0,0,1,0.45])

    ax11,Cplot = Obj.plot_PrimaryRegion(X,Z,Bpx,Bpz,Babs,ax11);
    ax12 = Obj.plot_PrimaryLoop(ax12)
    ax2 = Obj.plot_InducedCurrent_TD(ax2,Is,t,EMFi,Isi)

    EMF_str = '{:.1e}'.format(EMFi)
    ax12.text(-2.9,-1.4,'EMF = '+EMF_str+' *$\delta$(t) V',fontsize=FS)

    plt.show(fig1)


############################################
#   DEFINE CLASS
############################################

class IndEx():
    """Fucntionwhcihdf
    Input variables:

        Output variables:
    """

    def __init__(self,I,a1,a2,x,z,azm,R,L):
        """Defines Initial Attributes"""

        # INITIALIZES OBJECT

        # I: Transmitter loop Current
        # f: Transmitter frequency
        # a1: Transmitter Loop Radius
        # a2: Receiver loop Radius
        # x: Horizontal Receiver Loop Location
        # z: Vertical Receiver Loop Location
        # azm: Azimuthal angle for normal vector of receiver loop relative to up (-90,+90)
        # R: Resistance of receiver loop
        # L: Inductance of receiver loop

        self.I   = I
        self.a1  = a1
        self.a2  = a2
        self.x   = x
        self.z   = z
        self.azm = azm
        self.R   = R
        self.L   = L

    def calc_PrimaryRegion(self,X,Z):
        """Predicts magnitude and direction of primary field in region"""

        # CALCULATES INDUCING FIELD WITHIN REGION AND RETURNS AT LOCATIONS

        # Initiate Variables from object
        I   = self.I
        a1  = self.a1
        eps = 1e-6
        mu0 = 4*np.pi*1e-7   # 1e9*mu0

        s = np.abs(X)   # Define Radial Distance

        k = 4*a1*s/(Z**2 + (a1+s)**2)

        Bpx  = mu0*np.sign(X)*(Z*I/(2*np.pi*s + eps))*(1/np.sqrt(Z**2 + (a1+s)**2))*(-sp.ellipk(k) + ((a1**2 + Z**2 + s**2)/(Z**2 + (s-a1)**2))*sp.ellipe(k))
        Bpz  = mu0*           (  I/(2*np.pi           ))*(1/np.sqrt(Z**2 + (a1+s)**2))*( sp.ellipk(k) + ((a1**2 - Z**2 - s**2)/(Z**2 + (s-a1)**2))*sp.ellipe(k))
        Bpx[(X>-1.025*a1) & (X<-0.975*a1) & (Z>-0.025*a1) & (Z<0.025*a1)] = 0.
        Bpx[(X<1.025*a1) & (X>0.975*a1) & (Z>-0.025*a1) & (Z<0.025*a1)] = 0.
        Bpz[(X>-1.025*a1) & (X<-0.975*a1) & (Z>-0.025*a1) & (Z<0.025*a1)] = 0.
        Bpz[(X<1.025*a1) & (X>0.975*a1) & (Z>-0.025*a1) & (Z<0.025*a1)] = 0.
        Babs = np.sqrt(Bpx**2 + Bpz**2)

        return Bpx,Bpz,Babs

    def calc_PrimaryLoop(self):
        """Predicts magnitude and direction of primary field in loop center"""

        # CALCULATES INDUCING FIELD AT RX LOOP CENTER

        # Initiate Variables

        I   = self.I
        a1  = self.a1
        x   = self.x
        z   = self.z
        eps = 1e-7
        mu0 = 4*np.pi*1e-7   # 1e9*mu0

        s = np.abs(x)   # Define Radial Distance

        k = 4*a1*s/(z**2 + (a1+s)**2)

        Bpx = mu0*np.sign(x)*(z*I/(2*np.pi*s + eps))*(1/np.sqrt(z**2 + (a1+s)**2))*(-sp.ellipk(k) + ((a1**2 + z**2 + s**2)/(z**2 + (s-a1)**2))*sp.ellipe(k))
        Bpz = mu0*           (  I/(2*np.pi           ))*(1/np.sqrt(z**2 + (a1+s)**2))*( sp.ellipk(k) + ((a1**2 - z**2 - s**2)/(z**2 + (s-a1)**2))*sp.ellipe(k))

        self.Bpx = Bpx
        self.Bpz = Bpz

    def calc_IndCurrent_Cos_i(self,f,t):
        """Induced current at particular time and frequency"""

        Bpx = self.Bpx
        Bpz = self.Bpz
        a2  = self.a2
        azm = np.pi*self.azm/180.
        R   = self.R
        L   = self.L

        w = 2*np.pi*f

        ax = np.pi*a2**2*np.sin(azm)
        Az = np.pi*a2**2*np.cos(azm)

        Phi = (ax*Bpx + Az*Bpz)
        EMF = w*Phi*np.sin(w*t)
        Is = (Phi/(R**2 + (w*L)**2))*(-w**2*L*np.cos(w*t) + w*R*np.sin(w*t))

        return EMF,Is

    def calc_IndCurrent_cos_range(self,f,t):
        """Induced current over a range of times"""

        Bpx = self.Bpx
        Bpz = self.Bpz
        a2  = self.a2
        azm = np.pi*self.azm/180.
        R   = self.R
        L   = self.L

        w = 2*np.pi*f

        ax = np.pi*a2**2*np.sin(azm)
        Az = np.pi*a2**2*np.cos(azm)

        Phi = (ax*Bpx + Az*Bpz)
        phi = np.arctan(R/(w*L))-np.pi  # This is the phase and not phase lag
        Is  = -(w*Phi/(R*np.sin(phi) + w*L*np.cos(phi)))*np.cos(w*t + phi)
        Ire = -(w*Phi/(R*np.sin(phi) + w*L*np.cos(phi)))*np.cos(w*t)*np.cos(phi)
        Iim =  (w*Phi/(R*np.sin(phi) + w*L*np.cos(phi)))*np.sin(w*t)*np.sin(phi)

        return Ire,Iim,Is,phi

    def calc_IndCurrent_FD_i(self,f):
        """Give FD EMF and current for single frequency"""

        #INITIALIZE ATTRIBUTES
        Bpx = self.Bpx
        Bpz = self.Bpz
        a2  = self.a2
        azm = np.pi*self.azm/180.
        R   = self.R
        L   = self.L

        w = 2*np.pi*f

        ax = np.pi*a2**2*np.sin(azm)
        Az = np.pi*a2**2*np.cos(azm)

        Phi = (ax*Bpx + Az*Bpz)
        EMF = -1j*w*Phi
        Is = EMF/(R + 1j*w*L)

        return EMF,Is

    def calc_IndCurrent_FD_spectrum(self):
        """Gives FD induced current spectrum"""

        #INITIALIZE ATTRIBUTES
        Bpx = self.Bpx
        Bpz = self.Bpz
        a2  = self.a2
        azm = np.pi*self.azm/180.
        R   = self.R
        L   = self.L

        w = 2*np.pi*np.logspace(0,8,101)

        ax = np.pi*a2**2*np.sin(azm)
        Az = np.pi*a2**2*np.cos(azm)

        Phi = (ax*Bpx + Az*Bpz)
        EMF = -1j*w*Phi
        Is = EMF/(R + 1j*w*L)

        return EMF,Is

    def calc_IndCurrent_TD_i(self,t):
        """Give FD EMF and current for single frequency"""

        #INITIALIZE ATTRIBUTES
        Bpx = self.Bpx
        Bpz = self.Bpz
        a2  = self.a2
        azm = np.pi*self.azm/180.
        R   = self.R
        L   = self.L



        ax = np.pi*a2**2*np.sin(azm)
        Az = np.pi*a2**2*np.cos(azm)

        Phi = (ax*Bpx + Az*Bpz)
        Is = (Phi/L)*np.exp(-(R/L)*t)
#        V = (Phi*R/L)*np.exp(-(R/L)*t) - (Phi*R/L**2)*np.exp(-(R/L)*t)
        EMF = Phi

        return EMF,Is

    def calc_IndCurrent_TD_offtime(self):
        """Gives FD induced current spectrum"""

        #INITIALIZE ATTRIBUTES
        Bpx = self.Bpx
        Bpz = self.Bpz
        a2  = self.a2
        azm = np.pi*self.azm/180.
        R   = self.R
        L   = self.L

        t = np.logspace(-6,0,101)

        ax = np.pi*a2**2*np.sin(azm)
        Az = np.pi*a2**2*np.cos(azm)

        Phi = (ax*Bpx + Az*Bpz)
        Is = (Phi/L)*np.exp(-(R/L)*t)
        V = (Phi*R/L)*np.exp(-(R/L)*t) - (Phi*R/L**2)*np.exp(-(R/L)*t)

        return V,Is




       ###########################################
       #    PLOTTING FUNCTIONS
       ###########################################


    def plot_PrimaryRegion(self,X,Z,Bpx,Bpz,Babs,ax):

        # INITIALIZE ATTRIBUTES
        a1  = self.a1
        a2  = self.a2
        xR  = self.x
        zR  = self.z
        azm = self.azm*np.pi/180

        FS = 20

        # LOOP ORIENTATIONS
        Phi = np.linspace(0,2*np.pi,101)
        xTx = a1*np.cos(Phi)
        zTx = 0.07*a1*np.sin(Phi)
        xRx = xR + a2*np.cos(Phi)*np.cos(azm) + 0.1*a2*np.sin(Phi)*np.sin(azm)
        zRx = zR - a2*np.cos(Phi)*np.sin(azm) + 0.1*a2*np.sin(Phi)*np.cos(azm)


        ax.plot(xTx,zTx,color='black',linewidth=6)
        ax.plot(xTx,zTx,color=((0.6,0.6,0.6)),linewidth=4)
        ax.plot(xRx,zRx,color='black',linewidth=6)
        ax.plot(xRx,zRx,color=((0.4,0.4,0.4)),linewidth=4)
        #Cplot = ax.contourf(X,Z,np.log10(Babs),40,cmap='ocean_r')
        Cplot = ax.contourf(X,Z,np.log10(1e9*Babs),40,cmap='viridis')
        cbar = plt.colorbar(Cplot, ax=ax)
        cbar.set_label('log$_{10}\mathbf{|B_p|}$ [nT]', rotation=270, labelpad = 25, size=FS)
        cbar.ax.tick_params(labelsize=FS-2)
        #ax.streamplot(X,Z,Bpx,Bpz,color=(0.2,0.2,0.2),linewidth=2)
        ax.streamplot(X,Z,Bpx,Bpz,color=(1,1,1),linewidth=2)

        ax.set_xbound(np.min(X),np.max(X))
        ax.set_ybound(np.min(Z),np.max(Z))
        ax.set_xlabel('X [m]',fontsize=FS+2)
        ax.set_ylabel('Z [m]',fontsize=FS+2)
        ax.tick_params(labelsize=FS-2)


        return ax,Cplot

    def plot_PrimaryLoop(self,ax):

        FS = 20

        # INITIALIZE ATTRIBUTES
        azm = self.azm*np.pi/180
        a2  = self.a2
        Bpx = self.Bpx
        Bpz = self.Bpz

        Phi = np.linspace(0,2*np.pi,101)
        xRx =   np.cos(Phi)*np.cos(azm) + 0.1*np.sin(Phi)*np.sin(azm)
        zRx = - np.cos(Phi)*np.sin(azm) + 0.1*np.sin(Phi)*np.cos(azm)
        dxB = 1.75*Bpx/np.sqrt(Bpx**2 + Bpz**2)
        dzB = 1.75*Bpz/np.sqrt(Bpx**2 + Bpz**2)
        dxn = np.sin(azm)
        dzn = np.cos(azm)

        Babs = np.sqrt(Bpx**2 + Bpz**2)
        Bnor = Bpx*np.sin(azm) + Bpz*np.cos(azm)
        Area = np.pi*a2**2
        #EMF  = - 2*np.pi*f*Area*Bnor


        ax.plot(xRx,zRx,color='black',linewidth=6)
        ax.plot(xRx,zRx,color=((0.4,0.4,0.4)),linewidth=4)
        ax.arrow(0., 0., dxB, dzB, fc="b", ec="k",head_width=0.3, head_length=0.3,width=0.08 )
        ax.arrow(0., 0., dxn, dzn, fc="r", ec="k",head_width=0.3, head_length=0.3,width=0.08 )

        ax.set_xbound(-3,3)
        ax.set_ybound(-1.5,4.5)
        ax.set_xticks([])
        ax.set_yticks([])

        ax.text(1.2*dxn,1.3*dzn,'$\mathbf{n}$',fontsize=FS+4,color='r')
        ax.text(1.2*dxB,1.2*dzB,'$\mathbf{B_p}$',fontsize=FS+4,color='b')

        Babs_str = '{:.1e}'.format(1e9*Babs)
        Bn_str   = '{:.1e}'.format(1e9*Bnor)
        A_str    = '{:.2f}'.format(Area)
        #f_str    = '{:.1e}'.format(f)
        #EMF_str  = '{:.1e}j'.format(EMF)

        ax.text(-2.9,4.1,'$\mathbf{|B_p|}$ = '+Babs_str+' nT',fontsize=20)
        ax.text(-2.9,3.7,'$\mathbf{|B_{n}|}$ = '+Bn_str+' nT',fontsize=20)
        ax.text(-2.9,3.3,'Area = '+A_str+' m$^2$',fontsize=FS)
        #3ax.text(-2.9,-2.1,'f = '+f_str+' Hz',fontsize=FS)
        #ax.text(-2.9,-1.7,'EMF = '+EMF_str+' V',fontsize=FS)

        return ax


    def plot_InducedCurrent_cos(self,ax1,ax2,Ire,Iim,Is,phi,f,t):

        FS = 20

        # Numerical Values
        w  = 2*np.pi*f
        I0 = self.I*np.cos(w*t)
        Ipmax = self.I
        Ismax = np.max(Is)
        Iremax= np.max(Ire)
        Iimmax= np.max(Iim)
        T = 1/f

        tL_phase = np.array([2*T,2*T])
        IL_phase = np.array([Ipmax,1.25*Ipmax])
        tR_phase = np.array([2*T-phi/w,2*T-phi/w])
        IR_phase = np.array([Ismax,4.1*Ismax])


        xTicks  = (np.max(t)/8)*np.linspace(0,8,9)
        xLabels = ['0','T/2','T','3T/2','2T','5T/2','3T','7T/2','4T']

        ax1.plot(t,I0,color='k',linewidth=4)
        ax1.plot(tL_phase,IL_phase,color='k',ls=':',linewidth=8)
        ax1.set_xbound(0,np.max(t))
        ax1.set_ybound(1.51*np.min(I0),1.51*np.max(I0))
        ax1.set_xlabel('Time',fontsize=FS+2)
        ax1.set_ylabel('Primary Current [A]',fontsize=FS+2)
        ax1.tick_params(labelsize=FS-2)



        ax1b = ax1.twinx()
        ax1b.plot(t,Is,color='g',linewidth=4)
        ax1b.plot(tR_phase,IR_phase,color='k',ls=':',linewidth=8)
        ax1b.set_xbound(0,np.max(t))
        ax1b.set_ybound(5.01*np.min(Is),5.01*np.max(Is))
        ax1b.set_ylabel('Secondary Current [A]',fontsize=FS+2,color='g')
        ax1b.tick_params(labelsize=FS-2)
        ax1b.tick_params(axis='y',colors='g')
        ax1b.xaxis.set_ticks(xTicks)
        ax1b.xaxis.set_ticklabels(xLabels)
        ax1b.yaxis.set_major_formatter(FormatStrFormatter('%0.0e'))

        T_str  = '{:.1e}'.format(T)
        Ip_str = '{:.1e}'.format(self.I)
        Is_str = '{:.1e}'.format(np.max(Is))
        phi_str= '{:.1f}'.format(-180*phi/np.pi)
        ax1.text(0.05*T,1.35*Ipmax,'Period = '+T_str+' s',fontsize=FS)
        ax1.text(0.05*T,-1.2*Ipmax,'$\max|I_p|$ = '+Ip_str+' A',fontsize=FS)
        ax1.text(0.05*T,-1.4*Ipmax,'$\max|I_s|$ = '+Is_str+' A',fontsize=FS,color='g')
        ax1.text(1.7*T,1.35*Ipmax,'Phase Lag ($\phi$) = '+phi_str+'$^o$',fontsize=FS,color='k')




        ax2.plot(t,Ire,color='b',linewidth=4)
        ax2.plot(t,Iim,color='r',linewidth=4)
        ax2.set_xbound(0,np.max(t))
        ax2.set_ybound(1.5*np.min(Is),1.5*np.max(Is))
        ax2.set_xlabel('Time',fontsize=FS+2)
        ax2.set_ylabel('Secondary Current [A]',fontsize=FS+2)
        ax2.tick_params(labelsize=FS-2)
        ax2.xaxis.set_ticks(xTicks)
        ax2.xaxis.set_ticklabels(xLabels)
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%0.0e'))

        Ire_str = '{:.1e}'.format(Iremax)
        Iim_str = '{:.1e}'.format(Iimmax)
        ax2.text(0.05*T,-1.2*Ismax,'$\max|I_{\\rm{in phase}}|$ = '+Ire_str+' A',fontsize=FS,color='b')
        ax2.text(0.05*T,-1.4*Ismax,'$\max|I_{\\rm{quad}}|$ = '+Iim_str+' A',fontsize=FS,color='r')



        return ax1, ax1b, ax2




    def plot_InducedCurrent_FD(self,ax,Is,fi,EMFi,Isi):

        FS = 20

        R = self.R
        L = self.L

        Imax = np.max(-np.real(Is))

        f = np.logspace(0,8,101)



        ax.semilogx(f,-np.real(Is),color='k',linewidth=4)
        ax.semilogx(f,-np.imag(Is),color='k',ls='--',linewidth=4)
        ax.semilogx(fi*np.array([1.,1.]),np.array([0,1.1*Imax]),color='r',ls='-',linewidth=3)

        ax.set_xlabel('Frequency [Hz]',fontsize=FS+2)
        ax.set_ylabel('-Current [A]',fontsize=FS+2)
        ax.set_ybound(0,1.1*Imax)
        ax.tick_params(labelsize=FS-2)

        R_str    = '{:.1e}'.format(R)
        L_str    = '{:.1e}'.format(L)
        f_str    = '{:.1e}'.format(fi)
        EMF_str  = '{:.1e}j'.format(EMFi.imag)
        I_str    = '{:.1e} - {:.1e}j'.format(float(np.real(Isi)),np.abs(float(np.imag(Isi))))

        ax.text(1.4,1.01*Imax,'$R$ = '+R_str+' $\Omega$',fontsize=FS)
        ax.text(1.4,0.94*Imax,'$L$ = '+L_str+' H',fontsize=FS)
        ax.text(1.4,0.87*Imax,'$f$ = '+f_str+' Hz',fontsize=FS,color='r')
        ax.text(1.4,0.8*Imax,'$V$ = '+EMF_str+' V',fontsize=FS,color='r')
        ax.text(1.4,0.73*Imax,'$I_s$ = '+I_str+' A',fontsize=FS,color='r')

        return ax

    def plot_InducedCurrent_TD(self,ax,Is,ti,Vi,Isi):

        FS = 20

        R = self.R
        L = self.L

        Imax = np.max(Is)

        t = np.logspace(-6,0,101)

        ax.semilogx(t,Is,color='k',linewidth=4)
        ax.semilogx(ti*np.array([1.,1.]),np.array([0,1.3*Imax]),color='r',ls='-',linewidth=3)

        ax.set_xlabel('Time [s]',fontsize=FS+2)
        ax.set_ylabel('Induced Current [A]',fontsize=FS+2)
        ax.set_ybound(0,1.2*Imax)
        ax.tick_params(labelsize=FS-2)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%0.0e'))

        R_str    = '{:.1e}'.format(R)
        L_str    = '{:.1e}'.format(L)
        t_str    = '{:.1e}'.format(ti)
        V_str    = '{:.1e}'.format(Vi)
        I_str    = '{:.1e}'.format(Isi)
#
        ax.text(1.4e-6,1.12*Imax,'$R$ = '+R_str+' $\Omega$',fontsize=FS)
        ax.text(1.4e-6,1.04*Imax,'$L$ = '+L_str+' H',fontsize=FS)
        ax.text(4e-2,1.12*Imax,'$t$ = '+t_str+' s',fontsize=FS,color='r')
        ax.text(4e-2,1.04*Imax,'$V$ = '+V_str+' V',fontsize=FS,color='r')
        ax.text(4e-2,0.96*Imax,'$I_s$ = '+I_str+' A',fontsize=FS,color='r')

        return ax














