# copan:DISCOUNT model integration and analysis script as used in:
#
# J.F. Donges, W. Lucht, J. Heitzig, W. Barfuss, S.E. Cornell, S.J. Lade,
# and M. Schlüter, Taxonomies for structuring models for World-Earth system
# analysis of the Anthropocene: subsystems, their interactions and
# social-ecological feedback loops, Earth System Dynamics, in press (2021),
# Discussion paper: DOI: 10.5194/esd-2018-27.

from pylab import *
import numpy
from scipy.optimize import fsolve

# possible damage functions:

def f_lin(S): return gamma*S
def f_quad(S):
    if S>2*sigma: 0/0
    else: return 2.*gamma*(S-S**2/2/sigma)/sigma
def f_exp(S): return gamma*S*exp(1-S/sigma)
def f_norm(S): return zeta + gamma*exp(-(S-mu)**2/(2*sigma**2))  # USED IN PAPER

# other derived quantities, see paper for definitions:

def z(A):  # this is called phi in the paper
    if coop: return A**2*N*al/(1-al)+(1-A)*be/(1-be)
    else: return A*al/(1-al)+(1-A)*be/(1-be)

def D(A,S):
    b=unit(S)
    if coop: return ((al-be)*(sn-b*(E-b*eff*z(A))-1)-((A*N)**2*al**2/(1-al)-be**2/(1-be))*b**2*eff/2/N)
    else: return ((al-be)*(sn-b*(E-b*eff*z(A))-1)-(al**2/(1-al)-be**2/(1-be))*b**2*eff/2/N)

def P(D): return 1./(1.+(1./p0-1.)*exp(-D*q/p0/(1.-p0)))

def A(z): return (z - be/(1-be))/(al/(1-al)-be/(1-be))

# derivatives:

def dA(A,S): theD=D(A,S); return l*dt*A*(1-A)*(P(theD)-P(-theD))

def dS(A,S): b=unit(S); return dt*(s*(max(0,E-b*eff*z(A))) - r*S)

def z_dA0(S): b=unit(S); return max(be/(1-be),min(al/(1-al),((al-be)*(1+b*E-sn)+(al**2/(1-al)-be**2/(1-be))*b**2*eff/2/N)/((al-be)*b**2*eff)))
def z_dS0(S): b=unit(S); return max(be/(1-be),min(al/(1-al),(s*E-r*S)/(b*eff*s)))

# for cooperative version (not used in paper):

def coopks_dA0(S):
    b=unit(S)
    C1 = (al-be)*b**2*eff*al/(1-al)/N - al**2/(1-al)*b**2*eff/2/N
    C2 = -(al-be)*b**2*eff*be/(1-be)/N
    C3 = (al-be)*(sn-b*E+b**2*eff*be/(1-be)-1) + be**2/(1-be)*b**2*eff/2/N
    print(C1,C2,C3)
    return maximum(-inf,minimum(inf,(-C2+array([1.,-1.])*sqrt(C2**2-4*C1*C3))/C1/2))

def coopk_dS0(S):
    b=unit(S)
    def target(klocal): return z(1.*klocal/N) - (s*E-r*S)/(b*eff*s)
    return max(0,min(N,fsolve(target,N*1.)[0]))

# META-PARAMETERS:

case = 3
stoch = True
plot_nullclines = True#False

# DETAILED PARAMETER SETS:

if case == 1:
    tit = 'exponential marginal damages'
    unit = f_exp
    gamma = 1.
    sigma = 1.
    f_prefix = 'exp_'+str(gamma)+'_'+str(sigma)
    al = 0.5
    be = 0.1
    eff = 1.
    E = 1.5
    sn = 2.
    N = 20
    p0 = 0.5
    q = 3.
    r = 1.25
    s = 1.
    S1 = 1.85
    k1 = 5
    Smax = 5. # 2sigma if squared

if case == 2:
    tit = 'linear marginal damages'
    unit = f_lin
    gamma = 2.
    f_prefix = 'lin_'+str(gamma)
    al,be = 0.5,0.1
    eff = 1.
    E = 1.5
    sn = 2.
    N = 20
    p0,q = 0.5,3.
    r = 0.4 # .3 -- .5
    s = 1.
    S1,k1 = 0.,6 # 3.2,10 # 2.6,6
    Amin,Amax = 0.,1. #.2,.6 # 0.,1.
    Smin,Smax = 0.,6. # 2.,3. # 0.,6.

if case == 3:  # USED IN PAPER!
    # interesting cases:
    # norm_0.0_2.0_1.0_2.0_0.5_0.1_1.0_1.5_2.0_20_0.5_3.0_0.3_1.0_3.2,10: complex
    # norm_0.0_1.0_1.0_2.0_0.5_0.1_1.0_1.5_2.0_20_0.5_3.0_0.45_1.0_2.6_6: stable focus
    # norm_0.0_2.0_1.0_2.0_0.5_0.1_1.0_1.5_2.0_20_0.5_3.0_0.45_1.0_0.0_6
    # norm_0.0_1.0_1.0_2.0_0.5_0.1_1.0_1.5_2.0_20_0.5_3.0_0.35_1.0_0.0_6
    # todo: norm_0.0_2.0_1.0_2.0_0.5_0.1_1.0_1.5_2.0_20_0.5_3.0_0.5_1.0_1.85_5
    # norm_0.0_1.5_1.0_2.0_0.5_0.3_1.0_1.5_2.0_20_0.5_3.0_0.3_1.0_3.0_10.0 coop: unstable focus, stable cycle for fast learning, globally attractive desirable state for slow learning!
    tit = 'normal marginal damages'
    unit = f_norm
    eff = 1.0
    coop, zeta, gamma,sigma,mu, al,   be,   eff,    E,    sn,    N,    p0,   q,    r,    s,    S1,   k1 =\
    0,    0.,   1.1,   1.,   2., 0.5,  0.1,  1.*eff**2, \
                                               1.6/eff, 2.,   50,   0.5,  3.,   0.45, 1.,   1.,   25

# choose from:
#    eff = 1.0 or 1.1
#    0,    0.,   2.,   1.,   2., 0.5,  0.1,  1.*eff**2, \
#                                               1.5/eff, 2.,   20,   0.5,  3.,   0.3,  1.,   3.2,  10

# stable/unstable focus
#    eff = 1.0 or 0.9
#    0,    0.,   1.,   1.,   2., 0.5,  0.1,  1.*eff**2, \
#                                               1.5/eff, 2.,   20,   0.5,  3.,   0.45, 1.,   0.,   6

#    eff = 1.0 or 0.8
#    0,    0.,   2.,   1.,   2., 0.5,  0.1,  1.*eff**2, \
#                                               1.5/eff, 2.,   20,   0.5,  3.,   0.45, 1.,   0.,   6

#    0,    0.,   1.,   1.,   2., 0.5,  0.1,  1.,   1.5,  2.,   20,   0.5,  3.,   0.35, 1.,   0.,   6
#    1,    0.,   1.5,  1.,   2., 0.5,  0.3,  1.,   1.5,  2.,   20,   0.5,  3.,   0.3,  1.,   3.,  .5*20

    f_prefix = 'norm_'+str(zeta)+'_'+str(gamma)+'_'+str(sigma)+'_'+str(mu)
    Amin,Amax = 0.,1.
    Smin,Smax = 0.5,3.5

    # find intersection of z_dA0 and z_dS0 around S=2.6:
    def target(S): return z_dA0(S)-z_dS0(S)
    Sfocus = fsolve(target,2.6)[0]
    Afocus = A((z_dA0(Sfocus)+z_dS0(Sfocus))/2)
    print(Sfocus,Afocus)

fileprefix = f_prefix+'_'+str(al)+'_'+str(be)+'_'+str(eff)+'_'+str(E)+'_'+str(sn)+'_'+str(N)+'_'+str(p0)+'_'+str(q)+'_'+str(r)+'_'+str(s)+'_'+str(S1)+'_'+str(k1)
print(fileprefix)

# timestep:
dt = 0.001

# grid:
nSs = 301
nAs = 301
As = linspace(Amin,Amax,nAs)
Ss = linspace(Smin,Smax,nSs)

# quantities on grid:
A_dA0s = array([A(z_dA0(Ss[x])) for x in range(nSs)])
A_dS0s = array([A(z_dS0(Ss[x])) for x in range(nSs)])
coopA_dA0s = array([coopks_dA0(Ss[x])/N for x in range(nSs)])
coopA_dS0s = array([coopk_dS0(Ss[x])/N for x in range(nSs)])

# plot of nullclines:
figure()
title(tit)
plot(Ss,A_dA0s,'r--',lw=2)
plot(Ss,A_dS0s,'r:',lw=2)
plot(Ss,coopA_dA0s,'y--',lw=2)
plot(Ss,coopA_dS0s,'y:',lw=2)
gca().invert_xaxis()
savefig(fileprefix+'_regions.pdf')

# plot of dynamics (figure in paper):
iterations = 50000
repetitions = 5
fracs = []
ls = [.2,1.3]
spl = 0
figure(figsize=(4,10))
ax = subplot2grid((5,1),(2, 0))
ax.plot(Ss,cumsum([unit(S) for S in Ss]),"r")
ax.set_xticklabels([])
ax.set_yticks([])
ax.set_ylabel('Climate damages')

# generate one panel per l parameter value:
for l in ls:

    ax = subplot2grid((5,1),(spl, 0),rowspan=2)
    #title(tit)
    dAs = array([[dA(As[y],Ss[x]) for x in range(nSs)] for y in range(nAs)])
    dSs = array([[dS(As[y],Ss[x]) for x in range(nSs)] for y in range(nAs)])
    dAs2 = dAs**2
    dSs2 = dSs**2
    speed = sqrt(dAs2/dAs2.mean()+dSs2/dSs2.mean())

    # average dynamics from ODEs:

    print('streamplot...')
    ax.streamplot(Ss,As,dSs,dAs,linewidth=3.*speed/speed.max(),density=1.)
    if plot_nullclines:
        if coop:
            ax.plot(Ss,coopA_dA0s,'y--',lw=2)
            ax.plot(Ss,coopA_dS0s,'y:',lw=2)
        else:
            ax.plot(Ss,1.01*A_dA0s-.005,'r',lw=1)
            ax.plot(Ss,1.01*A_dS0s-.005,'r',lw=2)
    ax.set_xlim(Smin,Smax)
    ax.set_ylim(Amin,Amax)
    if spl == 3:
        ax.set_xlabel('Excess atmospheric carbon stock $C$ [fictitious units]')
    else:
        ax.set_xticklabels([])
    ax.set_ylabel('Fraction $F$ of patient countries')
    pmax=0.
    pmin=1.
    ngood = 0

    # individual stochastic trajectories:

    if repetitions>0: plot(S1,k1*1./N,'g.',ms=10)
    for rep in range(repetitions):
        Ss2 = [S1]
        ks2 = [k1]
        usal = []
        usbe = []
        for it in range(1,iterations):
            theA = 1.*ks2[it-1]/N
            theD = D(theA,Ss2[it-1])
            pal = P(theD)
            pbe = P(-theD)
            if pal>pmax: pmax=pal
            if pbe>pmax: pmax=pbe
            if pal<pmin: pmin=pal
            if pbe<pmin: pmin=pbe
            Ss2.append(max(0,Ss2[it-1] + dS(theA,Ss2[it-1])))
            if Ss2[it]>100.:
                ks2.append(0)
            else:
                if stoch:
                    ks2.append(ks2[it-1] + numpy.random.binomial(N-ks2[it-1],l*dt*pal*theA) - numpy.random.binomial(ks2[it-1],l*dt*pbe*(1.-theA)))
                else:
                    ks2.append(ks2[it-1] + ((N-ks2[it-1])*l*dt*pal*theA - ks2[it-1]*l*dt*pbe*(1.-theA)))
            b = unit(Ss2[it])
            thez = z(ks2[it]*1./N)
            usal.append(1+al*(sn-b*E+b**2*eff*thez-1)-al**2*b**2*eff/2/N/(1-al))
            usbe.append(1+be*(sn-b*E+b**2*eff*thez-1)-be**2*b**2*eff/2/N/(1-be))
            if ks2[it] in [0,N]:
                if ks2[it] == N: ngood += 1
                break
            if it%1000 == 0: print(rep,it,Ss2[it],ks2[it])
        if rep<20: ax.plot(Ss2,array(ks2)*1./N,'g-',alpha=0.5,color=(.5*rand(),.75,.5*rand()))
    if repetitions>0:
        fracs.append(ngood*1./repetitions)
        print(l,pmin,pmax,ngood*1./repetitions)
    spl = spl+3
savefig(fileprefix+'.pdf')
print("saved ",fileprefix+'.pdf')
show()
