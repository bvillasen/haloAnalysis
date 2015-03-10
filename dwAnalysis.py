import numpy as np
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
from interpolation import *
from calculus import *
from sphericalProfile import *
from jeansMass import *

h = 0.7


dataDir = "/home/bruno/Desktop/data/galaxy/Dw9/CDM/"
starFile = 'Stars_Dw9_CDM_hres.DAT'
dmFile   = 'Dark_Dw9_CDM_hres.DAT'
haloCatalogFile = 'hpro_1.000.dat'
starsKinFile = 'kinematics_star_Dw9hres.DAT'

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

#Halo properties
haloPos = np.array([ 0., 0., 0. ])
haloVel = np.array([ 0., 0., 0. ])
haloRvmax = 15.        #Not real
haloRvir = 73.34
haloMass = 10**10.46
starMass = 10**8.55

profiles = {}
particles = {}
centers = {}
betaVal = {}
for pType in ['star', 'dm']:
  profiles[pType] = {}
  particles[pType] = {}

  print pType
  print " Loading simulation data..."
  start = time.time()
  dataFile = starFile if pType=='star' else dmFile
  data = np.loadtxt( dataDir + dataFile )
  print " Time: ", time.time() - start

  #Particles 
  particlesPos  = data[:,1:4]/h
  particlesVel  = data[:,4:7]
  particlesMass = data[:,7]

  #Exclude far partiticles
  #Use origin as center
  dist = np.sqrt( ( particlesPos**2 ).sum(axis=1) )
  distMask = np.where( dist < haloRvir )[0]
  inHaloPartPos  = particlesPos[distMask]
  inHaloPartVel  = particlesVel[distMask]
  inHaloPartMass = particlesMass[distMask]
  nParticles = inHaloPartMass.shape[0]

  #Center
  findCenter = False
  if findCenter:
    centers[pType] = getCenter( haloPos, inHaloPartPos, inHaloPartMass, rLast_cm=0.2, nPartLast_cm=40 )
  centers['star'] = np.array([ 0.00134214,  0.03050503, -0.00056914 ])
  centers['dm']   = np.array([ -0.02455578,  0.02416358,  0.00088997 ])
  haloCenter = centers[pType]
  
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

  #Binning profile
  binning = 'exp'
  nPerBin = 1000
  rMin = .300
  rMax = 20 if pType=='star' else haloRvir
  rMax = distToCenter.max()
  nBins = nParticles / nPerBin
  if binning == 'pow' or binning == 'exp': nBins = 50
  binCenters, binWidths, binVols, densProf, velR_avr, velTH_avr, velT_avr, velProf_rad, velProf_theta, velProf_phi, betaDist, nPartDist = mha.get_profiles( rMin, rMax,
	      distToCenter, inHaloPartMass, vel_radial_val, vel_theta_val, vel_tangen_val,
	      nBins=nBins, binning=binning )
  profiles[pType]['r_bin'] = binCenters
  profiles[pType]['dens_bin'] = densProf  
  profiles[pType]['widths_bin'] = binWidths
  profiles[pType]['nPart_bin'] = nPartDist
  #Velocity Avrgs
  profiles[pType]['vRad_avr_bin'] = velR_avr
  profiles[pType]['vTheta_avr_bin'] = velTH_avr
  profiles[pType]['vPhi_avr_bin'] = velT_avr
  #Velociy sigma
  profiles[pType]['vRad_sig_bin'] = velProf_rad
  profiles[pType]['vTheta_sig_bin'] = velProf_theta
  profiles[pType]['vPhi_sig_bin'] = velProf_phi
  profiles[pType]['beta_bin'] = betaDist

  #Sampling profile
  if pType == 'star':
    findSamplingPforfile = False
    nRadialSamples = 100
    nGlobalSamples = 30
    samplingProfileFile = 'DW9/profiles/dens_samples_200_all.dat'
    if findSamplingPforfile:
      sampleSphRadius = np.linspace(0.2, 2, nRadialSamples)
      nLocalSamples = np.linspace(500, 2000, nRadialSamples)
      r_samples = np.linspace(0.3, 19, nRadialSamples)
      dens_samples = getSamplingProfile( r_samples, sampleSphRadius, nLocalSamples, nGlobalSamples, nRadialSamples, posRel, velRel, inHaloPartMass )
      np.savetxt( samplingProfileFile, np.concatenate( (np.array([r_samples]).T, dens_samples), axis=1) )
    samplesData = np.loadtxt( samplingProfileFile )
    samplesData = samplesData[::2]
    r_samples = samplesData[:,0]
    dens_samples = samplesData[:,1:]
    profiles[pType]['r_samples'] = r_samples
    profiles[pType]['dens_samples'] = dens_samples
    profiles[pType]['widths_samples'] = (r_samples[1]*r_samples[0]) * np.ones( r_samples.shape[0] )

  #Profiles from Alejandro
  if pType == 'dm':
    profiles[pType]['r_dw'] = centersDW	
    profiles[pType]['dens_dw'] = densityDW
  if pType == 'star':
    profiles[pType]['r_dw'] = centersST	
    profiles[pType]['dens_dw'] = densityST
    profiles[pType]['vRad_avr_dw'] = starsVelR_avr
    profiles[pType]['vPhi_avr_dw'] = starsVelT_avr
    profiles[pType]['vRad_sig_dw'] = starsVelR_sig
    profiles[pType]['vPhi_sig_dw'] = starsVelT_sig  
  
#####################################################  
pType = 'star'
offset = np.sqrt( ( ( centers['dm'] - centers['star'] )**2 ).sum() )


#####################################################
#Jeans Analysis
#####################################################
output =  currentDirectory + '/DW9/jeans/'  
if not os.path.exists( output ): os.makedirs( output )

#density
r_samples      = profiles[pType]['r_samples']
dens_samples   = profiles[pType]['dens_samples'][:,0]

#sigma vel radial
r_bin         = profiles[pType]['r_bin']
sigmaR_bin    = profiles[pType]['vRad_sig_bin']
sigmaR_intrp  = spline( r_bin, sigmaR_bin,  r_samples )

#sigma vel tangential
sigmaT_bin    = profiles[pType]['vPhi_sig_bin']
sigmaT_intrp  = spline( r_bin, sigmaT_bin,  r_samples )

#Jeans Mass integral
jR       = r_samples
jDeltaR  = profiles[pType]['widths_samples']
jDens    = dens_samples
jSigmaR2 = sigmaR_intrp**2
beta     = betaVal[pType]
jM_int   = jeansMass_int( jR, jDeltaR, jDens, jSigmaR2, beta )

#Jeans Mass differential
jR        = r_samples
jDens     = dens_samples
jDensDer  = derivative( r_samples, dens_samples )
#jDensDer  = derivative4( dens_samples, jDeltaR[0] )
jSigmaR2  = sigmaR_intrp**2
jSigR2Der = derivative( r_samples, sigmaR_intrp**2 )
#jSigR2Der = derivative4( sigmaR_intrp**2, jDeltaR[0] )
jSigmaT2  = sigmaT_intrp**2
jM_dif    = jeansMass_dif( jR, jDens, jDensDer, jSigmaR2, jSigR2Der, jSigmaT2 )


#####################################################
#Accum Mass
#####################################################
dist_dm = particles['dm']['dist']
acumMass_dm = particles['dm']['mass'].cumsum()
dist_all = np.concatenate( [ dist_dm, particles['star']['dist'] ] )
mass_all = np.concatenate( [ particles['dm']['mass'], particles['star']['mass'] ] )
sorted_all = dist_all.argsort()
dist_all = dist_all[sorted_all]
mass_all = mass_all[sorted_all]
acumMass_all = mass_all.cumsum()
#####################################################

fig1 = plt.figure()
fig1.clf()
ax = fig1.add_subplot(111)
ax.set_xscale('log')
ax.set_yscale('log')
ax.plot( dist_dm, acumMass_dm )
ax.plot( dist_all, acumMass_all )
ax.plot( jR, jM_int, 'o--' )
ax.plot( jR, jM_dif, 'o--' )
ax.set_xlim(jR[0], jR[-1])
fig1.show()












##Plot
#fig, ax = plt.subplots( 2, 2 )
#fig.set_size_inches(16,10)
#ax[0,0].set_xscale('log')
#ax[0,0].set_yscale('log')
#ax[0,0].plot( r_bin, profiles[pType]['dens_bin'], '-o' )
#ax[0,0].errorbar( r_samples, dens_samples, yerr=profiles[pType]['dens_samples'][:,1]  )
#ax[0,0].set_xlim(r_bin[0], r_bin[-1])

#ax[1,0].set_xscale('log')
#ax[1,0].errorbar( r_bin, sigmaR_bin, yerr=sigmaR_bin/np.sqrt(profiles[pType]['nPart_bin'])  )
#ax[1,0].plot( r_samples, sigmaR_intrp, 'o-' )
#ax[1,0].set_xlim(r_bin[0], r_bin[-1])
#fig.show()
######################################################
######################################################













#Plot Profiles
output =  currentDirectory + '/DW9/{0}/'.format(binning)  
if not os.path.exists( output ): os.makedirs( output )
#Plot profiles
if binning == 'equiPart': 
  title = 'NperBin: {0}'.format( nPerBin )
  name = '_{0}'.format( nPerBin )
if binning == 'exp' or binning=='pow': 
  title = 'nBins: {0}'.format( nBins )
  name = '_{0}'.format( nBins )
#plots.plotDensity_p( profiles, haloRvir, haloRvmax, output=output, title=title, name=name, addDW=True, addFine=False, show=False )
#plots.plotKinemat_p( profiles, haloRvir, haloRvmax, output=output, title=title, name=name, addDW=True)
  




