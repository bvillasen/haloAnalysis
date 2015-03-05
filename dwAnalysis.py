import numpy as np
from scipy.optimize import curve_fit
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import h5py as h5
import os, sys, tempfile, time

currentDirectory = os.getcwd()
#Add Modules from other directories
parentDirectory = currentDirectory[:currentDirectory.rfind("/")]
toolsDirectory = currentDirectory + "/tools"
sys.path.append( toolsDirectory )
import plots
import tools
from geometry import *
from findCenters import *
import multi_haloAnalysis as mha
#from interpolation import *
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
from calculus import *

h = 0.7

dataDir = "/home/bruno/Desktop/data/galaxy/Dw9/CDM/"
starFile = 'Stars_Dw9_CDM_hres.DAT'
dmFile   = 'Dark_Dw9_CDM_hres.DAT'
haloCatalogFile = 'hpro_1.000.dat'
starsKinFile = 'kinematics_star_Dw9hres.DAT'


#r = np.arange(0, 10, 1. )
#data = np.arange(0, 10, 1.)
#nAverg = 3
#step = 2


def smooth( r, data, nAverg, step ):
  N = r.shape[0]
  r_out, data_out = [], [] 
  left, right = 0, nAverg
  while right <= N:
    avr = data[ left : right ].mean()
    r_out.append( ( r[left] + r[right-1] )/2 )
    data_out.append( avr )
    left += step
    right = left + nAverg
  return np.array( r_out ), np.array( data_out ) 

def fineProfile_dens( r, mass, npLeft, npRight ):
  nPart = mass.shape[0]
  nPts  = nPart - ( npLeft + npRight )
  dProf = np.zeros( nPts )
  rProf = np.zeros( nPts )
  for i in range( nPts ):
    rL = r[ i ]
    rR = r[ i + npLeft + npRight  ]
    shellVol = 4.*np.pi/3 * ( rR**3 - rL**3 )
    mInShell = mass[ i : i + npLeft + npRight + 1 ].sum()
    #print mInShell, shellVol
    rProf[i] = r[ npLeft + i ]
    dProf[i] = mInShell / shellVol
  return rProf, dProf

def fineProfile_sigma( r, data, npLeft, npRight ):
  nPart = data.shape[0]
  nPts  = nPart - ( npLeft + npRight )
  rProf = np.zeros( nPts )
  meanProf = np.zeros( nPts )
  sigmaProf = np.zeros( nPts )
  for i in range( nPts ):
    rL = r[ i ]
    rR = r[ i + npLeft + npRight  ]
    inShell_vals = data[ i : i + npLeft + npRight + 1 ]
    nInShell = len( inShell_vals )
    inShell_avg = inShell_vals.mean()
    inShell_sig = np.sqrt( ( ( inShell_vals - inShell_avg )**2 ).mean() )
    rProf[i] = r[ npLeft + i ]
    meanProf[i] = inShell_avg
    sigmaProf[i] = inShell_sig
  return rProf, meanProf, sigmaProf


#Load profiles data from halo catalog
profilesData = []
catalogFile = open( dataDir + haloCatalogFile, 'r' )

rhoc = 1.879*1.4775e-7*h**2.0
Om=0.3
lineCounter = 0
for line in catalogFile:
  lineCounter += 1
  if lineCounter == 21:	haloData = np.fromstring( line, dtype=float, sep='      ' ) 
  if lineCounter > 21 and lineCounter < 161: profilesData.append( np.fromstring( line, dtype=float, sep='      ' ) ) 
  if lineCounter == 161: break
profilesData = np.array( profilesData ) 
centersDW = profilesData[:,0] / h
densityDW = profilesData[:,3] * rhoc * Om * 1e9 
starsData = np.loadtxt( dataDir + starsKinFile )
centersST = starsData[:,3] / h
densityST = starsData[:,5]
starsVelR_avr = starsData[:,11]
starsVelT_avr = starsData[:,12]
starsVelR_sig = starsData[:,14]
starsVelT_sig = starsData[:,15]

profiles = {}
particles = {}
centers = {}
betaVal = {}
for pType in ['star', 'dm']:
  #pType = 'star'
  profiles[pType] = {}
  particles[pType] = {}


  print pType
  print "\nLoading simulation data..."
  start = time.time()
  dataFile = starFile if pType=='star' else dmFile
  data = np.loadtxt( dataDir + dataFile )
  print " Time: ", time.time() - start

  #Particles 
  particlesPos  = data[:,1:4]/h
  particlesVel  = data[:,4:7]
  particlesMass = data[:,7]

  #Halo properties
  haloRvmax = 15.
  haloRvir = 73.34
  haloMass = 10**10.46
  starMass = 10**8.55

  #Exclude far partiticles
  #Use origin as center
  dist = np.sqrt( ( particlesPos**2 ).sum(axis=1) )
  distMask = np.where( dist < haloRvir )[0]
  inHaloPartPos  = particlesPos[distMask]
  inHaloPartVel  = particlesVel[distMask]
  inHaloPartMass = particlesMass[distMask]
  nParticles = inHaloPartMass.shape[0]

  #Center
  #if pType == 'dm':
    #cm = ( inHaloPartPos * inHaloPartMass[:,None] ).sum(axis=0) / particlesMass.sum()
    #tree = KDTree( inHaloPartPos )
    #print " Initial sphere:"  
    #ball_pos = cm
    #massRatio = 0.3 if pType == 'star' else 0.3
    #ball_radius, massRatio_i = getBallFromMassRatio( ball_pos, haloRvmax, massRatio, inHaloPartMass, tree )
    #print '  massRatio_i: {0}'.format(massRatio_i)
    #print '  ball_radius: {0}, rvmax: {1}, rvir: {2}'.format( ball_radius, haloRvmax, haloRvir )
    #rLast = 0.80
    #nPartLast = 40
    #center, radius_f, nInBall_f = findCenter_CM( ball_pos, ball_radius, rLast, nPartLast, inHaloPartMass, inHaloPartPos, tree )
    #print '   cm:{1} n:{0} r:{2}'.format(nInBall_f, center, radius_f)            
    #haloCenter = moveBall( center, radius_f, inHaloPartMass, inHaloPartPos, tree )
    #center['pType']  = haloCenter
  ##center_star = np.array([ -0.00164466,  0.02685468,  0.00439187] )
  #center_dm   = np.array([ -0.02682095,  0.03045762, -0.00483639] )
  #center_dm   = np.array([-0.03169116,  0.04313293, -0.00467443]
  center_dm   = np.array([-0.01057742,  0.011094,   0.0036434 ])
  center_star = np.array([-0.01064021,  0.0368292,   0.0058786 ])
  haloCenter = center_star if pType == 'star' else center_dm
  haloVel = 0
  #haloVel = inHaloPartVel.sum(axis=0)/nParticles
  #offset = np.sqrt( ( ( center_dm - center_star )**2 ).sum() )

  #Particles properties
  posRel       = inHaloPartPos - haloCenter
  velRel       = inHaloPartVel - haloVel
  distToCenter = np.sqrt( np.sum(posRel**2, axis=1) )
  sortedIndx   = distToCenter.argsort()
  distToCenter   = distToCenter[sortedIndx]
  inHaloPartMass = inHaloPartMass[sortedIndx]
  inHaloPartPos = inHaloPartPos[sortedIndx]
  inHaloPartVel = inHaloPartVel[sortedIndx]
  posRel         = posRel[sortedIndx]
  velRel         = velRel[sortedIndx]
  #Get velocity components
  radial_vec, theta_vec, phi_vec = spherical_vectors( posRel )  #Tangential is phi vector NOT THETA
  vel_radial_val = np.array([ radial_vec[i].dot(velRel[i]) for i in range(nParticles) ]) 
  vel_tangen_val = np.array([    phi_vec[i].dot(velRel[i]) for i in range(nParticles) ]) #TANGENTIAL
  vel_theta_val  = np.array([  theta_vec[i].dot(velRel[i]) for i in range(nParticles) ])
  sigma_rad2 = ( ( vel_radial_val - vel_radial_val.mean() )**2 ).mean()
  sigma_tan2 = ( ( vel_tangen_val - vel_tangen_val.mean() )**2 ).mean()
  betaVal[pType] = 1 - ( sigma_tan2 / sigma_rad2 )

    
  particles[pType]['mass'] = inHaloPartMass
  particles[pType]['dist'] = distToCenter

  #Profiles
  binning = 'pow'
  nPerBin = 5000
  rMin, rMax = .300, distToCenter.max()
  nBins = nParticles / nPerBin
  if binning == 'pow' or binning == 'exp': nBins = 100
  binCenters, binWidths, binVols, densProf, velProf_rad, velProf_theta, velProf_phi, betaDist, nPartDist = mha.get_profiles( rMin, rMax,
	      distToCenter, inHaloPartMass, vel_radial_val, vel_theta_val, vel_tangen_val,
	      nBins=nBins, binning=binning )

  profiles[pType]['density'] = densProf
  profiles[pType]['vel_rad'] = velProf_rad
  profiles[pType]['vel_theta'] = velProf_theta
  profiles[pType]['vel_phi'] = velProf_phi
  profiles[pType]['beta'] = betaDist
  profiles[pType]['nPartDist'] = nPartDist
  profiles[pType]['binCenters'] = binCenters
  profiles[pType]['binWidths'] = binWidths
  profiles[pType]['binVols'] = binVols
  if pType == 'dm':
    profiles[pType]['binCenters_dw'] = centersDW	
    profiles[pType]['density_dw'] = densityDW
  if pType == 'star':
    profiles[pType]['binCenters_dw'] = centersST	
    profiles[pType]['density_dw'] = densityST
  
  if pType == 'star':
    #Fine profiles
    npLeft = 1000
    npRight = 800
    rProf_f, dProf_f = fineProfile_dens( distToCenter, inHaloPartMass, npLeft, npRight )
    rProf_f, avgVelR_f, sigmaR_f = fineProfile_sigma( distToCenter, vel_radial_val, npLeft, npRight )
    rProf_f, avgVelT_f, sigmaT_f = fineProfile_sigma( distToCenter, vel_tangen_val, npLeft, npRight )
    profiles[pType]['r_fine'] = rProf_f	
    profiles[pType]['dens_fine'] = dProf_f
    profiles[pType]['avgVelR_fine'] = avgVelR_f
    profiles[pType]['sigmaR_fine'] = sigmaR_f
    profiles[pType]['avgVelT_fine'] = avgVelT_f
    profiles[pType]['sigmaT_fine'] = sigmaT_f
    
    #Smooth fine profiles
    nAverg = 5000
    step = 5000
    r_smooth , dens_smooth = smooth( rProf_f, dProf_f, nAverg, step )
    r_smooth , sigmaR_smooth = smooth( rProf_f, sigmaR_f, nAverg, step )
    r_smooth , sigmaT_smooth = smooth( rProf_f, sigmaT_f, nAverg, step )
    profiles[pType]['r_smooth'] = r_smooth	
    profiles[pType]['dens_smooth'] = dens_smooth
    profiles[pType]['sigmaR_smooth'] = sigmaR_smooth
    profiles[pType]['sigmaT_smooth'] = sigmaT_smooth
#####################################################  
output =  currentDirectory + '/DW9/{0}/'.format(binning)  
if not os.path.exists( output ): os.makedirs( output )
#Plot profiles
if binning == 'equiPart': 
  title = 'NperBin: {0}'.format( nPerBin )
  name = '_{0}'.format( nPerBin )
if binning == 'exp' or binning=='pow': 
  title = 'nBins: {0}'.format( nBins )
  name = '_{0}'.format( nBins )
plots.plotDensity_p( profiles, haloRvir, haloRvmax, output=output, title=title, name=name, addDW=True, addFine=False, show=False )
plots.plotKinemat_p( profiles, haloRvir, haloRvmax, output=output, title=title, name=name )
  
pType = 'star'
r        = profiles[pType]['binCenters']
r_fine   = profiles[pType]['r_fine']
r_smooth = profiles[pType]['r_smooth']

density = profiles[pType]['density']
dens_fine = profiles[pType]['dens_fine']
dens_smooth = profiles[pType]['dens_smooth']
density_der = derivative( r, density )
dens_sm_der = derivative( r_smooth, dens_smooth )
 
sigmaR   = profiles[pType]['vel_rad']
sigmaR2  = profiles[pType]['vel_rad']**2
sigmaR_fine = profiles[pType]['sigmaR_fine']
sigmaR2_fine = sigmaR_fine**2
sigmaR_smooth = profiles[pType]['sigmaR_smooth']
sigmaR2_smooth = sigmaR_smooth**2
sigmaR2_der = derivative( r, sigmaR2 )
sigmaR2_sm_der = derivative( r_smooth, sigmaR2_smooth )

sigmaT  = profiles[pType]['vel_phi']
sigmaT2 = profiles[pType]['vel_phi']**2
sigmaT_fine = profiles[pType]['sigmaT_fine']
sigmaT2_fine = sigmaT_fine**2
sigmaT_smooth = profiles[pType]['sigmaT_smooth']
sigmaT2_smooth = sigmaT_smooth**2

nPart   = profiles[pType]['nPartDist']
#######################################################
##from scipy.interpolate import UnivariateSpline
##def spline( x, y, x_i ):
  ##spline = UnivariateSpline( x, y, k=5 )
  ###spline = InterpolatedUnivariateSpline( x, y, k=5 )
  ###spline = Rbf( x, y, function='quintic' ) 
  ##y_i = spline( x_i )
  ##return y_i
#######################################################




###Smoothing
##nSmooth= 100
##r_s = np.linspace( r[0], r[-1], nSmooth, endpoint=True )
##r_s = r.copy()
##density_s = spline( r, density, r_s )
##density_der_s = spline( r, density_der, r_s )
##sigmaR_s  = spline( r, sigmaR,  r_s )
##sigmaT_s  = spline( r, sigmaT,  r_s )
##sigmaR2 = sigmaR_s**2
##sigmaT2 = sigmaT_s**2
##density_s_der = derivative( r_s, density_s )
##sigmaR_s_der  = derivative( r_s, sigmaR_s )
##sigmaR2_der = derivative( r_s, sigmaR2 )



######################################################
#Jeans Calc
G    = 6.67384e-11 #m**3/(kg*s**2)
mSun = 1.98e30     #kg
pc   = 3.08e16     #m
G = G / (1e3*pc) / 1e3**2 * mSun


#######################################################
#Jeans Diff	
jR       = r
jDensity = density
jDensDer = density_der
jSigmaR2 = sigmaR2
jSigR2Der = sigmaR2_der
jSigmaT2 = sigmaT2

Diff = jDensity * jSigR2Der + jSigmaR2 * jDensDer
#Diff =                        jSigmaR2.mean() * jDensDer
jM = 1./G * ( jR**2 / jDensity * Diff + 2 * jR * ( jSigmaR2 - jSigmaT2 ) )
#jM = jR**2 / ( G * jDensity ) * Diff 
jeansMassDiff_0 = -jM
jR_0 = jR.copy()


jR       = r_smooth
jDensity = dens_smooth
jDensDer = dens_sm_der
jSigmaR2 = sigmaR2_smooth
jSigR2Der = sigmaR2_sm_der
jSigmaT2 = sigmaT2_smooth


Diff = jDensity * jSigR2Der + jSigmaR2 * jDensDer
#Diff =                        jSigmaR2 * jDensDer
jM = 1./G * ( jR**2 / jDensity * Diff + 2 * jR * ( jSigmaR2 - jSigmaT2 ) )
#jM = jR**2 / ( G * jDensity ) * Diff 
jeansMassDiff_1 = -jM
jR_dif = jR.copy()
########################################################
#Jeans Int	

#jR       = r
#jDeltaR  = profiles[pType]['binWidths'] 
#jDensity = density
#jSigmaR2 = sigmaR2

jR       = r_smooth
jDensity = dens_smooth
jSigmaR2 = sigmaR2_smooth

beta = betaVal[pType]
nPoints = len(jR)
jeansMassInt = np.zeros( nPoints )
for n in range(nPoints-1,-1,-1):
  r_n   = jR[n]
  #dr_n  = jDeltaR[n]
  dr_n  = jR[n] if n==0 else jR[n]-jR[n-1]
  rho_n = jDensity[n]
  r_right    = jR[n+1:]
  rho_right  = jDensity[n+1:]
  mass_right = jeansMassInt[n+1:]
  #beta_right = betaVal
  integrand  = rho_right * mass_right * r_right**(2*beta-2)
  integral = 0 if len(integrand)==0 else simps( integrand, r_right )
  sigmaR2_n = jSigmaR2[n]
  #beta_n    = betaVal[pType]
  jeansMassInt[n] = sigmaR2_n * r_n**2 / ( G * dr_n ) - integral / ( rho_n * r_n**(2*beta-2) * dr_n )
jR_int = jR.copy()


#####################################################
#Accum Mass
dist_dm = particles['dm']['dist']
acumMass_dm = particles['dm']['mass'].cumsum()

dist_all = np.concatenate( [ dist_dm, particles['star']['dist'] ] )
mass_all = np.concatenate( [ particles['dm']['mass'], particles['star']['mass'] ] )
sorted_all = dist_all.argsort()
dist_all = dist_all[sorted_all]
mass_all = mass_all[sorted_all]
acumMass_all = mass_all.cumsum()

###############################################################
#Plot
fig, ax = plt.subplots( 2, 2 )
fig.set_size_inches(16,10)
ax[0,0].loglog( r, density, 'o' )
ax[0,0].loglog( r_smooth, dens_smooth, 'o-'   )
ax[0,0].set_xlim(r[0], r[-1])

ax[0,1].set_xscale('log')
#ax[0,1].set_yscale('log')
ax[0,1].plot( r, density_der, '-o' )
ax[0,1].plot( r_smooth, dens_sm_der, '-o' )
ax[0,1].set_xlim(r[0], r[-1])

ax[1,0].set_xscale('log')
ax[1,0].errorbar( r, sigmaR,  yerr=sigmaR/np.sqrt(nPart) )
ax[1,0].plot( r_fine, sigmaR_fine, '-'   )
ax[1,0].plot( r_smooth, sigmaR_smooth, '-o' )
ax[1,0].errorbar( r, sigmaT,  yerr=sigmaT/np.sqrt(nPart) )
ax[1,0].plot( r_fine, sigmaT_fine, '-'   )
ax[1,0].plot( r_smooth, sigmaT_smooth, '-o' )
ax[1,0].set_xlim(r[0], r[-1])

ax[1,1].set_xscale('log')
ax[1,1].plot( r, sigmaR2_der, '-o' )
ax[1,1].plot( r_smooth, sigmaR2_sm_der, '-' )
##ax[1,1].plot( r_s, 2*sigmaR_s*sigmaR_s_der, '-' )
ax[1,1].set_xlim(r[0], r[-1])

#fig.subplots_adjust(hspace=0)
if binning == 'equiPart':  
  title = 'NperBin: {0}'.format( nPerBin )
  name = 'jeans_{0}'.format(nPerBin) 
if binning == 'exp' or binning == 'pow':  
  title = 'nBins: {0}'.format( nBins )
  name = 'jeans_{0}'.format(nBins) 
ax[0,0].set_title(title)
#fig.savefig(output+ name +'.png')
fig.show()

fig1 = plt.figure(1)
fig1.clf()
ax = fig1.add_subplot(111)
ax.set_xscale('log')
ax.set_yscale('log')
ax.plot( dist_dm, acumMass_dm )
ax.plot( dist_all, acumMass_all )
ax.plot( jR_0, jeansMassDiff_0, 'o--' )
ax.plot( jR_dif, jeansMassDiff_1, 'o--' )
ax.plot( jR_int, jeansMassInt, 'o--' )
fig1.show()

######################################################
##if jeansCalc == 'jeansInt':
  ##for n in range(nInterp-1,-1,-1):
    ##rho_n = density_i[n]
    ##r_n   = r_i[n]
    ##dr_n  = r_i[n] if n==0 else r_i[n]-r_i[n-1]
    ##r_right    = r_i[n+1:]
    ##rho_right  = density_i[n+1:]
    ##mass_right = jeansMass[n+1:]
    ##integrand  = rho_right * mass_right * r_right**(2*beta-2)
    ##integral = 0 if len(integrand)==0 else simps( integrand, r_right )
    ##sigma2_n = sigmaR_i[n]**2
    ##jeansMass[n] = sigma2_n * r_n**2 / ( G * dr_n ) - integral / ( rho_n * r_n**(2*beta-2) * dr_n )

  
#######################################################
##if jeansCalc == 'jeansDiff':	   
  ##for n in range( nPoints ):
    ##r_n   = r_i[n]
    ##rho_n = density_i[n]
    ##sigma_r2_n  = sigmaR_i[n]**2
    ##sigma_t2_n  = sigmaT_i[n]**2
    ##if n == 0:
      ##r_nr        = r_i[n+1]
      ##rho_nr      = density_i[n+1]
      ##sigma_r2_nr = sigmaR_i[n+1]**2
      ##Diff        = ( rho_nr*sigma_r2_nr - rho_n*sigma_r2_n ) / (r_nr - r_n )
    ##elif n == nInterp-1:  
      ##r_nl        = r_i[n-1]
      ##rho_nl      = density_i[n-1]
      ##sigma_r2_nl = sigmaR_i[n-1]**2
      ##Diff        = ( rho_n*sigma_r2_n - rho_nl*sigma_r2_nl ) / (r_n - r_nl )
    ##else:
      ##r_nr        = r_i[n+1]
      ##r_nl        = r_i[n-1]
      ##rho_nr      = density_i[n+1]
      ##rho_nl      = density_i[n-1]
      ##sigma_r2_nr = sigmaR_i[n+1]**2
      ##sigma_r2_nl = sigmaR_i[n-1]**2
      ##Diff        = ( rho_nr*sigma_r2_nr - rho_nl*sigma_r2_nl ) / (r_nr - r_nl )
    ####Diff = sigma_r2_n*( rho_nr - rho_nl ) / (r_nr - r_nl )
    ##jM = r_n**2 / ( G * rho_n ) * Diff 
    ##jM = r_n**2 / ( G * rho_n ) * Diff + 2*r_n/G*( sigma_r2_n - sigma_t2_n )
    ##jeansMass[n] = -jM



###jeansDensity = np.zeros(nInterp)
###for i in range(nInterp-1):
  ###jeansDeltaMass = jeansMass[i] if i==0 else jeansMass[i] - jeansMass[i-1]
  ###vol = 4*np.pi/3 * r_i[i]**3 if i==0 else 4*np.pi/3* ( r_i[i+1]**3 - r_i[i-1]**3 ) 
  ###if jeansDeltaMass > 0: jeansDensity[i] = jeansDeltaMass / vol



##########################################################################
###jeansMass = np.zeros( nBins )
###for n in range(nBins-1,-1,-1):
  ###rho_n = densProf[n]
  ###r_n   = binCenters[n]
  ###dr_n  = binWidths[n]
  ###r_right    = binCenters[n+1:]
  ###rho_right  = densProf[n+1:]
  ###mass_right = jeansMass[n+1:]
  ###integrand  = rho_right * mass_right * r_right**(2*beta-2)
  ###integral = 0 if len(integrand)==0 else simps( integrand, r_right )
  ###sigma2_n = velProf_rad[n]**2
  ###jeansMass[n] = sigma2_n * r_n**2 / ( G * dr_n ) - integral / ( rho_n * r_n**(2*beta-2) * dr_n )
  

###plt.figure(0)
###plt.clf()
###plt.plot(jeansMass)
###plt.show()

###jeansMass = np.zeros( nBins )
###for n in range( 0, nBins ):
  ###r_n   = binCenters[n]
  ###rho_n = densProf[n]
  ###sigma_r2_n  = velProf_rad[n]**2
  ###sigma_t2_n  = velProf_phi[n]**2
  ###if n == 0:
    ###sigma_r2_nr = velProf_rad[n+1]**2
    ###rho_nr = densProf[n+1]
    ###r_nr  = binCenters[n+1]
    ###Diff = ( rho_nr*sigma_r2_nr - rho_n*sigma_r2_n ) / (r_nr - r_n )
  ###elif n == nBins-1:  
    ###sigma_r2_nl = velProf_rad[n-1]**2
    ###rho_nl = densProf[n-1]
    ###r_nl  = binCenters[n-1]
    ###Diff = ( rho_n*sigma_r2_n - rho_nl*sigma_r2_nl ) / (r_n - r_nl )
  ###else:
    ###sigma_r2_nr = velProf_rad[n+1]**2
    ###rho_nr = densProf[n+1]
    ###r_nr  = binCenters[n+1]
    ###sigma_r2_nl = velProf_rad[n-1]**2
    ###rho_nl = densProf[n-1]
    ###r_nl  = binCenters[n-1]
    ###Diff = ( rho_nr*sigma_r2_nr - rho_nl*sigma_r2_nl ) / (r_nr - r_nl )
  ####Diff = sigma_r2_n*( rho_nr - rho_nl ) / (r_nr - r_nl )
  ###jM = r_n**2 / ( G * rho_n ) * Diff 
  ####jM = r_n**2 / ( G * rho_n ) * Diff + 2*r_n/G*( sigma_r2_n - sigma_t2_n )
  ###jeansMass[n] = -jM

###jeansProf = np.zeros( nBins )
###for i in range(nBins):
  ###binMass = jeansMass[i] if i==0 else jeansMass[i] - jeansMass[i-1] 
  ###if binMass < 0: continue
  ###jeansProf[i] = binMass/binVols[i]


###plt.figure(1)
###plt.clf()
####plt.plot(jeansMass[:-1])
###plt.loglog( binCenters, jeansProf )
###plt.show()


