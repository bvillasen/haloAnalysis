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
haloRvmax = 6.0        #measured by me
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
  rMin = .06
  rMax = haloRvir if pType=='star' else haloRvir
  rMax = distToCenter.max()
  nBins = nParticles / nPerBin
  if binning == 'pow' or binning == 'exp': nBins = 25
  binCenters, binWidths, binVols, densProf, velR_avr, velTH_avr, velT_avr, velProf_rad, velProf_theta, velProf_phi, betaDist, nPartDist = mha.get_profiles( rMin, rMax,
	      distToCenter, inHaloPartMass, vel_radial_val, vel_theta_val, vel_tangen_val,
	      nBins=nBins, binning=binning )
  profiles[pType]['r_bin'] = binCenters
  profiles[pType]['dens_bin'] = densProf  
  profiles[pType]['widths_bin'] = binWidths
  profiles[pType]['nPart_bin'] = nPartDist
  profiles[pType]['binVols'] = binVols
  #Velocity Avrgs
  profiles[pType]['vRad_avr_bin'] = velR_avr
  profiles[pType]['vTheta_avr_bin'] = velTH_avr
  profiles[pType]['vPhi_avr_bin'] = velT_avr
  #Velociy sigma
  profiles[pType]['vRad_sig_bin'] = velProf_rad
  profiles[pType]['vTheta_sig_bin'] = velProf_theta
  profiles[pType]['vPhi_sig_bin'] = velProf_phi
  profiles[pType]['beta_bin'] = betaDist

  #kernel profie
  #if pType == 'star':
  rMin_k = 0.06
  rMax_k = haloRvir if pType=='dm' else haloRvir*0.75
  nRadialSamples_k = 20
  r_kernel = np.logspace( np.log10(rMin_k), np.log10(rMax_k), base=10, num=nRadialSamples_k)
  #r_kernel = np.linspace( rMin_k, rMax_k, nRadialSamples_k )
  #########################################
  h_pilot = 0.08
  outputKernel = currentDirectory + '/DW9/kernel/'
  kernelProfileFile = outputKernel + 'densKernelPilot_{1}_h{0:.2f}.dat'.format(h_pilot, pType)
  getPilot = False
  if getPilot:
    print '\n Getting kernel density pilot...'
    start = time.time()
    densPilot = np.array([ densityKernel( r_i, h_pilot, distToCenter, inHaloPartMass ) for r_i in distToCenter ]) 
    np.savetxt( kernelProfileFile, densPilot )
    print " Time: ", time.time() - start
  densPilot = np.loadtxt( kernelProfileFile )
  g = geoMean( densPilot )
  alpha = 0.3
  h_kernel = h_pilot * (densPilot / g )**(-alpha)
  densKernel = np.array([ densityKernel( r_i, h_kernel, distToCenter, inHaloPartMass ) for r_i in r_kernel ])
  #########################################
  profiles[pType]['r_kernel'] = r_kernel
  profiles[pType]['dens_kernel'] = densKernel
  #profiles[pType]['centers_kernel_geo'] = np.sqrt(
  widths_kernel = np.zeros(r_kernel.shape[0])
  widths_kernel[0] = r_kernel[1] - r_kernel[0]
  widths_kernel[1:-1] = ( r_kernel[2:] - r_kernel[:-2] )/2
  widths_kernel[-1] = ( r_kernel[-1] - r_kernel[-2] )
  profiles[pType]['widths_kernel'] = widths_kernel
    
    
  #Sampling profile
  if pType == 'star':
    findSamplingPforfile = False
    nRadialSamples = 20
    nGlobalSamples = 10
    sampleSphRadius = np.linspace(0.2, 1, nRadialSamples)
    nLocalSamples = np.linspace(500, 1000, nRadialSamples)
    rMin, rMax =  0.3,  8
    #r_samples = np.linspace( rMin, rMax, nRadialSamples)
    r_samples = np.logspace( np.log10(rMin), np.log10(rMax), base=10, num=nRadialSamples)
    output = currentDirectory + '/DW9/profiles'
    #samplingProfileFile = output + '/dens_samples_[0.3:8:20]_kLin_sLog.dat'
    samplingProfileFile = output + '/dens_samples_[0:9:50]_kLin.dat'
    if findSamplingPforfile:
      print 'saving profile in: ', samplingProfileFile
      dens_samples = getSamplingProfile( r_samples, sampleSphRadius, nLocalSamples, nGlobalSamples, nRadialSamples, posRel, velRel, inHaloPartMass )
      np.savetxt( samplingProfileFile, np.concatenate( (np.array([r_samples]).T, dens_samples), axis=1) )
    samplesData = np.loadtxt( samplingProfileFile )
    samplesData = samplesData[::1]
    r_samples = samplesData[:,0]
    dens_samples = samplesData[:,1:]
    profiles[pType]['r_samples'] = r_samples
    profiles[pType]['dens_samples'] = dens_samples
    widths_samples = np.zeros(r_samples.shape[0])
    widths_samples[0] = r_samples[1] - r_samples[0]
    widths_samples[1:-1] = ( r_samples[2:] - r_samples[:-2] )/2
    widths_samples[-1] = (r_samples[-1] - r_samples[-2])
    profiles[pType]['widths_samples'] = widths_samples
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

#####################################################
#Jeans Analysis
#####################################################
output =  currentDirectory + '/DW9/jeans/'  
if not os.path.exists( output ): os.makedirs( output )


#####################################################
#Jeans Analysis
#####################################################
output =  currentDirectory + '/halos/jeans/'  
if not os.path.exists( output ): os.makedirs( output )

#radius
r_bin     = profiles[pType]['r_bin']
#r_samples = profiles[pType]['r_samples']
r_kernel = profiles[pType]['r_kernel']

#density
dens_bin     = profiles[pType]['dens_bin']
#dens_samples = profiles[pType]['dens_samples'][:,0]
dens_kernel = profiles[pType]['dens_kernel']
#dens_der_samples = derivative( r_samples, dens_samples )
dens_der_bin = derivative( r_bin, dens_bin )

#sigma vel radial
sigmaR_bin    = profiles[pType]['vRad_sig_bin']
sigmaR_intrp_bin  = spline( r_bin, sigmaR_bin,  r_bin )
#sigmaR_intrp_samples  = spline( r_bin, sigmaR_bin,  r_samples )
sigmaR_intrp_kernel  = spline( r_bin, sigmaR_bin,  r_kernel )
#sigmaR_intrp_kernel  = rbf( r_bin, sigmaR_bin,  r_kernel )
sigmaR2_der_bin = derivative( r_bin, sigmaR_intrp_bin**2 )
#sigmaR2_der_samples = derivative( r_samples, sigmaR_intrp_samples**2 )

#sigma vel tangential
sigmaT_bin    = profiles[pType]['vPhi_sig_bin']
sigmaT_intrp_bin  = spline( r_bin, sigmaT_bin,  r_bin )
#sigmaT_intrp_samples  = spline( r_bin, sigmaT_bin,  r_samples )

##Jeans mass samples
##########################################################
##Jeans Mass integral
#jR_samples = r_samples
#jDeltaR  = profiles[pType]['widths_samples']
#jDens    = dens_samples
#jSigmaR2 = sigmaR_intrp_samples**2
#beta     = betaVal[pType]
#jM_int_samples   = jeansMass_int( jR_samples, jDeltaR, jDens, jSigmaR2, beta )
#jM_int_samples_cor = jeansMass_int_corr( jR_samples, jDens, jSigmaR2, beta )
##Jeans Mass differential
#jDens     = dens_samples
#jDensDer  = dens_der_samples
#jSigmaR2  = sigmaR_intrp_samples**2
#jSigR2Der = sigmaR2_der_samples
#jSigmaT2  = sigmaT_intrp_samples**2
#jM_dif_samples    = jeansMass_dif( jR_samples, jDens, jDensDer, jSigmaR2, jSigR2Der, jSigmaT2 )
#########################################################

#Jeans mass bin
#########################################################
#Jeans Mass integral
jR_bin       = r_bin
jDeltaR  = profiles[pType]['widths_bin']
jDens    = dens_bin
jSigmaR2 = sigmaR_intrp_bin**2
beta     = betaVal[pType]
jM_int_bin   = jeansMass_int_corr( jR_bin, jDens, jSigmaR2, beta )
#Jeans Mass differential
jDens     = dens_bin
jDensDer  = dens_der_bin
jSigmaR2  = sigmaR_intrp_bin**2
jSigR2Der = sigmaR2_der_bin
jSigmaT2  = sigmaT_intrp_bin**2
jM_dif_bin    = jeansMass_dif( jR_bin, jDens, jDensDer, jSigmaR2, jSigR2Der, jSigmaT2 )
#########################################################

#Jeans mass kernel
#########################################################
#Jeans Mass integral
jR_kernel= r_kernel
jDeltaR  = profiles[pType]['widths_kernel']
jDens    = dens_kernel
jSigmaR2 = sigmaR_intrp_kernel**2
beta     = betaVal[pType]
jM_int_kernel = jeansMass_int( jR_kernel, jDeltaR, jDens, jSigmaR2, beta )
jM_int_kernel_corr = jeansMass_int_corr( jR_kernel, jDens, jSigmaR2, beta )



#fig1 = plt.figure()
#fig1.clf()
#ax = fig1.add_subplot(111)
#ax.set_xscale('log')
#ax.set_yscale('log')
#ax.plot( dist_dm, acumMass_dm )
#ax.plot( dist_all, acumMass_all )
#ax.plot( jR_bin, jM_int_bin, 'o--' )
#ax.plot( jR_kernel, jM_int_kernel_corr, 'o--' )
#ax.axvline( x=haloRvmax, color='m')
#ax.axvline( x=haloRvir, color='y')
#ax.set_xlim(0.05, haloRvir*1.1 )
#ax.legend(['samp', 'bin'], loc=0)
#ax.set_title('Jeans Int')
#fig1.show()

#Plot
fig, ax = plt.subplots( 2, 2 )
fig.set_size_inches(16,10)
ax[0,0].set_xscale('log')
ax[0,0].set_yscale('log')
ax[0,0].plot( r_bin, profiles[pType]['dens_bin'], '-o' )
ax[0,0].plot( r_kernel, profiles[pType]['dens_kernel'], '-o' )
#ax[0,0].errorbar( r_samples, dens_samples, yerr=profiles[pType]['dens_samples'][:,1]  )
ax[0,0].axvline( x=haloRvmax, color='m')
ax[0,0].axvline( x=haloRvir, color='y')
ax[0,0].legend(['bin', 'kernel'], loc=0)
ax[0,0].set_xlim(0.05, haloRvir*1.1 )

ax[1,0].set_xscale('log')
#ax[1,0].plot( r_bin, sigmaR_intrp_bin, 'o-' )
#ax[1,0].plot( r_samples, sigmaR_intrp_samples, 'o-' )
ax[1,0].errorbar( r_bin, sigmaR_bin, yerr=sigmaR_bin/np.sqrt(profiles[pType]['nPart_bin'])  )
ax[1,0].plot( r_kernel, sigmaR_intrp_kernel, 'o-' )
ax[1,0].axvline( x=haloRvmax, color='m')
ax[1,0].axvline( x=haloRvir, color='y')
ax[1,0].legend(['bin', 'kernel'], loc=0)
ax[1,0].set_xlim(0.05, haloRvir*1.1 )

ax[0,1].set_xscale('log')
ax[0,1].set_yscale('log')
ax[0,1].plot( dist_dm, acumMass_dm )
ax[0,1].plot( dist_all, acumMass_all )
ax[0,1].plot( jR_bin, jM_int_bin, 'o--' )
ax[0,1].plot( jR_kernel, jM_int_kernel_corr, 'o--' )
ax[0,1].legend(['M_dm', 'M_all', 'jM_bin', 'jM_ker'], loc=0)
ax[0,1].axvline( x=0.3 )
ax[0,1].axvline( x=haloRvmax, color='m')
ax[0,1].axvline( x=haloRvir, color='y')
ax[0,1].set_xlim(0.05, haloRvir*1.1 )
ax[0,1].set_ylim(1e6, 1e11)

ax[1,1].set_xscale('log')
ax[1,1].set_yscale('log')
ax[1,1].plot( r_bin, profiles[pType]['nPart_bin'], 'o' )
ax[1,1].axvline( x=haloRvmax, color='m')
ax[1,1].axvline( x=haloRvir, color='y')
ax[1,1].set_xlim(0.05, haloRvir*1.1 )
fig.subplots_adjust(hspace=0)
title = ''
ax[0,0].set_title( title )
fig.show()
























#fig0 = plt.figure()
#fig0.clf()
#ax = fig0.add_subplot(111)
#fig0.set_size_inches(8,6)
#ax.set_xscale('log')
#ax.set_yscale('log')
#ax.loglog(   profiles['dm']['r_dw'], profiles['dm']['dens_dw'], '-o' )
#ax.loglog(   profiles['dm']['r_kernel'], profiles['dm']['dens_kernel'], '-' )
#ax.loglog(   profiles['star']['r_dw'], profiles['star']['dens_dw'], '-o' )
#ax.loglog(   profiles['star']['r_kernel'], profiles['star']['dens_kernel'], '-' )
#ax.axvline( x=haloRvmax, color='m')
#ax.axvline( x=haloRvir, color='y')
#ax.set_xlim(0.1, haloRvir*1.05)
#ax.legend(['dm_Alejan', 'dm_kernel', 'st_kernel'], loc=0)
#ax.set_xlabel(r'r [kpc]')
#ax.set_ylabel(r'Density $[{\rm M_{\odot} kpc^{-3}  }]$')
#ax.set_title('DW_9 density profile')
#fig0.savefig( outputKernel + 'densProf_h{0:.2f}.png'.format(h_pilot) )
##fig0.show()


#fig2 = plt.figure()
#fig2.clf()
#ax = fig2.add_subplot(111)
#ax.set_xscale('log')
#ax.set_yscale('log')
#ax.plot( dist_dm, acumMass_dm )
#ax.plot( dist_all, acumMass_all )
#l1 = ax.plot( jR_samples, jM_dif_samples, 'o--', label='samp' )
#l2 = ax.plot( jR_bin, jM_dif_bin, 'o--', label='bin' )
#ax.axvline( x=haloRvmax, color='m')
#ax.set_xlim(jR_samples[0], jR_samples[-1])
#ax.legend(['samp', 'bin'], loc=0)
#ax.set_title('Jeans Diff')
#fig2.show()


##Plot
#fig, ax = plt.subplots( 2, 2 )
#fig.set_size_inches(16,10)
#ax[0,0].set_xscale('log')
#ax[0,0].set_yscale('log')
#ax[0,0].plot( r_bin, profiles[pType]['dens_bin'], '-o' )
#ax[0,0].errorbar( r_samples, dens_samples, yerr=profiles[pType]['dens_samples'][:,1]  )
#ax[0,0].axvline( x=haloRvmax, color='m')
#ax[0,0].set_xlim(r_bin[0], r_bin[-1])

#ax[1,0].set_xscale('log')
#ax[1,0].plot( r_bin, sigmaR_intrp_bin, 'o-' )
#ax[1,0].plot( r_samples, sigmaR_intrp_samples, 'o-' )
#ax[1,0].errorbar( r_bin, sigmaR_bin, yerr=sigmaR_bin/np.sqrt(profiles[pType]['nPart_bin'])  )
#ax[1,0].axvline( x=haloRvmax, color='m')
#ax[1,0].set_xlim(r_bin[0], r_bin[-1])

#ax[0,1].set_xscale('log')
#ax[0,1].set_yscale('log')
#ax[0,1].plot( r_bin, -dens_der_bin, 'o-' )
#ax[0,1].plot( r_samples, -dens_der_samples, 'o-' )
#ax[0,1].axvline( x=haloRvmax, color='m')
#ax[0,1].set_xlim(r_bin[0], r_bin[-1])

#ax[1,1].set_xscale('log')
#ax[1,1].plot( r_bin, sigmaR2_der_bin, 'o-' )
#ax[1,1].plot( r_samples, sigmaR2_der_samples, 'o-' )
#ax[1,1].axvline( x=haloRvmax, color='m')
#ax[1,1].set_xlim(r_bin[0], r_bin[-1])
#title = ''
#ax[0,0].set_title( title )
##fig.show()









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
plots.plotDensity_p( profiles, haloRvir, haloRvmax, output=output, title=title, name=name, addBin=False, addDW=True, addKernel=True, show=False )
plots.plotKinemat_p( profiles, haloRvir, haloRvmax, output=output, title=title, name=name, addDW=False)
  




