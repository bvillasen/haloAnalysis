from snapAnalysis import *
from calculus import *
from interpolation import *
from sphericalProfile import *
from jeansMass import *

snapshot = 3
  
#Host halo
hostId  = ms_hostId[snapshot][0]
hostPos = ms_halosData[snapshot][ hostId ].attrs['pos']*1e3

#Halo properties
#hId   = hostId
#hId    = '1317'
#hId    = '1320'
#hId   = '163'
#hId   = '1459'
#hId   = '1838'
#hId   = '936'
#hId   = '725'
#hId   = '882'
#hId   = '1490'
#hId   = '1825'
hId   = '1851'

findSamplingProfile = False
#Read in-line parameters
for option in sys.argv:
  if option.find("h=")>=0 : hId = option[option.find("=")+1:]
  if option == 'samprof' : findSamplingProfile = True

print 'Halo: ', hId


haloMass   = ms_halosData[snapshot][ hId ].attrs['mbound_vir']
haloPos    = ms_halosData[snapshot][ hId ].attrs['pos']*1e3
haloVel    = ms_halosData[snapshot][ hId ].attrs['vel']
haloRvir   = ms_halosData[snapshot][ hId ].attrs['rvir']*1e3
haloRvmax  = ms_halosData[snapshot][ hId ].attrs['rvmax']*1e3
haloVmax   = ms_halosData[snapshot][ hId ].attrs['vmax']
distToHost = np.sqrt( np.sum( (haloPos - hostPos)**2, axis=-1) )
sMass      = ms_halosData[snapshot][ hId ].attrs['SM']

profiles = {}
particles = {}
centers = {}
betaVal = {}
nParticles = {}
for pType in ['star', 'dm']:
  profiles[pType] = {}
  particles[pType] = {}
  print '\n'+pType
  
  #Particles
  particlesPos  = ms_partsData[snapshot][pType]['pos'][...]*1e3
  particlesVel  = ms_partsData[snapshot][pType]['vel'][...]
  particlesMass = ms_partsData[snapshot][pType]['mass'][...]

  #In halo particles
  inHaloPartIds  = ms_halosPartsIds[snapshot][ hId ][pType+'_bound'][...] 
  inHaloPartPos  = particlesPos[inHaloPartIds]
  inHaloPartVel  = particlesVel[inHaloPartIds]
  inHaloPartMass = particlesMass[inHaloPartIds]
  nParticles[pType] = inHaloPartMass.shape[0]
  center = ms_centers[snapshot][pType][hId]
  
  #Particles properties relative to halo center
  posRel_h       = inHaloPartPos - haloPos
  distToCenter_h = np.sqrt( np.sum(posRel_h**2, axis=1) )
  sortedIndx_h   = distToCenter_h.argsort()
  distToCenter_h   = distToCenter_h[sortedIndx_h]
  inHaloPartMass_h = inHaloPartMass[sortedIndx_h]
  particles[pType]['mass_halo'] = inHaloPartMass_h.copy()
  particles[pType]['dist_halo'] = distToCenter_h.copy()
  
  #Particles properties relative to dm-star center
  posRel       = inHaloPartPos - center
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
  vel_radial_val = np.array([ radial_vec[i].dot(velRel[i]) for i in range(len(distToCenter)) ]) 
  vel_tangen_val = np.array([    phi_vec[i].dot(velRel[i]) for i in range(len(distToCenter)) ]) #TANGENTIAL
  vel_theta_val  = np.array([  theta_vec[i].dot(velRel[i]) for i in range(len(distToCenter)) ])
  sigma_rad2 = ( ( vel_radial_val - vel_radial_val.mean() )**2 ).mean()
  sigma_tan2 = ( ( vel_tangen_val - vel_tangen_val.mean() )**2 ).mean()
  betaVal[pType] = 1 - ( sigma_tan2 / sigma_rad2 )

  particles[pType]['mass_center'] = inHaloPartMass.copy()
  particles[pType]['dist_center'] = distToCenter.copy()
  

  #Binning profile
  rMin = .11
  rMax = haloRvir if pType=='dm' else haloRvmax*6
  binning = 'exp'
  nBins = 17
  nPerBin = nParticles[pType] / nBins
  #if binning == 'pow' or binning == 'exp': nBins = 22
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
  rMin_k = 0.11
  rMax_k = haloRvir if pType=='dm' else haloRvmax*4
  nRadialSamples_k = 12
  r_kernel = np.logspace( np.log10(rMin_k), np.log10(rMax_k), base=10, num=nRadialSamples_k)
  #r_kernel = np.linspace( rMin_k, rMax_k, nRadialSamples_k )
  #########################################
  h_pilot = 0.2
  outputKernel = currentDirectory + '/profiles/h_{0}/kernel/'.format(hId)
  if not os.path.exists( outputKernel ): os.makedirs( outputKernel )
  kernelProfileFile = outputKernel + 'densKernelPilot_{1}_h{0:.2f}.dat'.format(h_pilot, pType)
  getPilot = True
  if getPilot:
    print '\n Getting kernel pilot density ...'
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
  
  
  #Sampling profile
  #if pType == 'star':
    #nRadialSamples = 20
    #nGlobalSamples = 10
    #nLocalSamples = np.linspace(1000, 1000, nRadialSamples)
    #rMin, rMax =  0.1,  haloRvmax*4
    ##r_samples = np.linspace( rMin, rMax, nRadialSamples)
    #r_samples = np.logspace( np.log10(rMin), np.log10(rMax), base=10, num=nRadialSamples)
    ##sampleSphRadius = ( np.linspace(0.1**(1./3), 5**(1./3), nRadialSamples) )**3
    ##sampleSphRadius = np.linspace( 0.05, rMax/20., nRadialSamples) 
    #sampleSphRadius = np.exp( np.linspace( np.log(0.08), np.log(rMax/20.), nRadialSamples) )    #sampleSphRadius = r_samples/10.    
    #kernel = 'gaus'
    #output = currentDirectory + '/profiles/h_{0}/samples/'.format( hId )
    #fileName = 'samProf_r[{0:.1f}:{1:.1f}:{2}]_sphR[{4:.2f}:{5:.1f}]_\
##k{3}_rsampLog_sphExp.dat'.format( rMin, rMax, nRadialSamples,kernel, sampleSphRadius[0], sampleSphRadius[-1]  )
    #samplingProfileFile = output + fileName
    #if not os.path.exists( output ): os.makedirs( output )
    ##if findSamplingProfile:
      #print 'saving profile in: ', samplingProfileFile
      #dens_samples = getSamplingProfile( r_samples, sampleSphRadius, nLocalSamples, nGlobalSamples, nRadialSamples, posRel, vel_radial_val, inHaloPartMass, kernel )
      #h = '#nGlobalSamples: {0}\n#nRadialSamples: {1}\n#sampleSphereRadius: [ {2:.1f} , {3:.1f} ]_cuadratic\n#nLocalSamples: [ {4:.0f} , {5:.0f} ]_linear'.format(nGlobalSamples, nRadialSamples, sampleSphRadius[0], sampleSphRadius[-1], nLocalSamples[0], nLocalSamples[-1])
      #np.savetxt( samplingProfileFile, np.concatenate( (np.array([r_samples]).T, dens_samples), axis=1),  header=h )
    #samplesData = np.loadtxt( samplingProfileFile )
    #samplesData = samplesData[0:]
    #r_samples = samplesData[:,0]
    #dens_samples = samplesData[:,1]
    #densSigma_samples = samplesData[:,2]
    #velRSigma_samples = samplesData[:,3]
    #profiles[pType]['r_samples'] = r_samples
    #profiles[pType]['dens_samples'] = dens_samples
    #profiles[pType]['densSigma_samples'] = densSigma_samples
    #profiles[pType]['densHigh_samples'] = dens_samples + densSigma_samples/np.sqrt(nGlobalSamples)
    #profiles[pType]['densLow_samples']  = dens_samples - densSigma_samples/np.sqrt(nGlobalSamples)
    #profiles[pType]['velRSigma_samples'] = velRSigma_samples
    #widths_samples = np.zeros(r_samples.shape[0])
    #widths_samples[0] = r_samples[1] - r_samples[0]
    #widths_samples[1:-1] = ( r_samples[2:] - r_samples[:-2] )/2
    #widths_samples[-1] = (r_samples[-1] - r_samples[-2])
    #profiles[pType]['widths_samples'] = widths_samples

#####################################################  
pType = 'star'
offset = np.sqrt( ( ( ms_centers[snapshot]['dm'][hId] - ms_centers[snapshot]['star'][hId] )**2 ).sum() )
offsets = [ ms_offsets[3-i][descId] for i,descId in enumerate(ms_descendentIds[hId])]
offsetsStr = '[{0:.2f}, {1:.2f}, {2:.2f}, {3:.2f}]'.format(offsets[0], offsets[1], offsets[2], offsets[3])
#####################################################
#Accum Mass
#####################################################
dist_dm_halo = particles['dm']['dist_halo']
dist_st_halo = particles['star']['dist_halo']
acumMass_dm_halo = particles['dm']['mass_halo'].cumsum()
acumMass_star_halo = particles['star']['mass_halo'].cumsum()
dist_all_halo = np.concatenate( [ dist_dm_halo, dist_st_halo ] )
mass_all_halo = np.concatenate( [ particles['dm']['mass_halo'], particles['star']['mass_halo'] ] )
sorted_all_halo = dist_all_halo.argsort()
dist_all_halo = dist_all_halo[sorted_all_halo]
mass_all_halo = mass_all_halo[sorted_all_halo]
acumMass_all_halo = mass_all_halo.cumsum()

dist_dm_center = particles['dm']['dist_center']
dist_st_center = particles['star']['dist_center']
acumMass_dm_center = particles['dm']['mass_center'].cumsum()
acumMass_star_center = particles['star']['mass_center'].cumsum()
dist_all_center = np.concatenate( [ dist_dm_center, dist_st_center ] )
mass_all_center = np.concatenate( [ particles['dm']['mass_center'], particles['star']['mass_center'] ] )
sorted_all_center = dist_all_center.argsort()
dist_all_center = dist_all_center[sorted_all_center]
mass_all_center = mass_all_center[sorted_all_center]
acumMass_all_center = mass_all_center.cumsum()
#####################################################
#Jeans Analysis
#####################################################
#output =  currentDirectory + '/halos/jeans/'  
#if not os.path.exists( output ): os.makedirs( output )

#radius
r_bin     = profiles[pType]['r_bin']
#r_samples = profiles[pType]['r_samples']
r_kernel = profiles[pType]['r_kernel']

#density
dens_bin     = profiles[pType]['dens_bin']
dens_kernel  = profiles[pType]['dens_kernel']
#dens_samples = profiles[pType]['dens_samples']
#dens_der_samples = derivative( r_samples, dens_samples )
dens_der_bin = derivative( r_bin, dens_bin )

#sigma vel radial
sigmaR_bin    = profiles[pType]['vRad_sig_bin']
#sigmaR_samples = profiles[pType]['velRSigma_samples']
#sigmaR_intrp_samples  = spline( r_bin, sigmaR_bin,  r_samples )
sigmaR_intrp_kernel  = rbf( r_bin, sigmaR_bin,  r_kernel )
sigmaR_intrp_bin  = spline( r_bin, sigmaR_bin,  r_bin )
#sigmaR2_der_bin = derivative( r_bin, sigmaR_intrp_bin**2 )
#sigmaR2_der_samples = derivative( r_samples, sigmaR_intrp_samples**2 )

#sigma vel tangential
sigmaT_bin    = profiles[pType]['vPhi_sig_bin']
#sigmaT_intrp_samples  = spline( r_bin, sigmaT_bin,  r_samples )
sigmaT_intrp_bin  = spline( r_bin, sigmaT_bin,  r_bin )
sigmaT_intrp_kernel  = spline( r_bin, sigmaT_bin,  r_kernel )
##Jeans mass samples
##########################################################
##Jeans Mass integral
#jR_samples = r_samples
#jDeltaR  = profiles[pType]['widths_samples']
#jDens    = dens_samples
#jSigmaR2 = sigmaR_intrp_samples**2
#betaConst = betaVal[pType]
#jM_int_samples = jeansMass_int( jR_samples, jDeltaR, jDens, jSigmaR2, betaConst )
#jM_int_samples_lessStar = substract( dist_st_center, acumMass_star_center, jR_samples, jM_int_samples )

##Using sigmaR_samples profile
#jSigmaR2 = sigmaR_samples**2
#jM_int_samples_densAndSigma = jeansMass_int( jR_samples, jDeltaR, jDens, jSigmaR2, betaConst )

##betaVar  = betaConst * np.ones_like( jDens )
#betaVar = 1 - sigmaT_intrp_samples**2 / sigmaR_intrp_samples**2
#jM_intBvar_samples = jeansMass_int_betaVar( jR_samples, jDeltaR, jDens, jSigmaR2, betaVar )
##jMhigh_int_samples = jeansMass_int( jR_samples, jDeltaR, densHigh_samples, jSigmaR2, beta )
##jMlow_int_samples = jeansMass_int( jR_samples, jDeltaR, densLow_samples, jSigmaR2, beta )
#########################################################

#Jeans mass bin
#########################################################
#Jeans Mass integral
jR_bin   = r_bin
jDens    = dens_bin
#jSigmaR2 = sigmaR_intrp_bin**2
jSigmaR2 = sigmaR_bin**2
beta     = betaVal[pType]
jM_int_bin   = jeansMass_int_corr( jR_bin, jDens, jSigmaR2, beta )

#Delta sigma jeans mass
deltaSigmaR = max( sigmaR_bin/np.sqrt(profiles[pType]['nPart_bin']) )
sigmaR_bin_upper = sigmaR_bin + deltaSigmaR 
sigmaR_bin_lower = sigmaR_bin - deltaSigmaR
jSigmaR2 = sigmaR_bin_upper**2
jM_int_bin_up   = jeansMass_int_corr( jR_bin, jDens, jSigmaR2, beta )
jSigmaR2 = sigmaR_bin_lower**2
jM_int_bin_low   = jeansMass_int_corr( jR_bin, jDens, jSigmaR2, beta )

#########################################################

#Jeans mass kernel
#########################################################
#Jeans Mass integral
jR_kernel= r_kernel
jDens    = dens_kernel
jSigmaR2 = sigmaR_intrp_kernel**2
beta     = betaVal[pType]
jM_int_kernel = jeansMass_int_corr( jR_kernel, jDens, jSigmaR2, beta )
#Delta sigma jeans mass
deltaSigmaR = max( sigmaR_bin/np.sqrt(profiles[pType]['nPart_bin']) )
sigmaR_kernel_upper = sigmaR_intrp_kernel + deltaSigmaR 
sigmaR_kernel_lower = sigmaR_intrp_kernel - deltaSigmaR 
jSigmaR2 = sigmaR_kernel_upper**2
jM_int_kernel_up   = jeansMass_int_corr( jR_kernel, jDens, jSigmaR2, beta )
jSigmaR2 = sigmaR_kernel_lower**2
jM_int_kernel_low   = jeansMass_int_corr( jR_kernel, jDens, jSigmaR2, beta )

#########################################################

#Plot Profiles
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
ax[0,1].plot( dist_dm_halo, acumMass_dm_halo )
ax[0,1].plot( dist_all_halo, acumMass_all_halo )
#ax[0,1].plot( jR_bin, jM_int_bin, 'o--' )
ax[0,1].plot( jR_kernel, jM_int_kernel, 'o--' )
ax[0,1].plot( jR_kernel, jM_int_kernel_up, '--', color='c' )
ax[0,1].plot( jR_kernel, jM_int_kernel_low, '--', color='c' )
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
title = 'hId:{2}, mD:{0:1.1e}, mS:{3:1.1e}, d:{1:.0f}, nD:{6:1.1e} nS:{4:1.1e}, off:{5}'.format(haloMass, distToHost, hId, sMass, nParticles['star'], offsetsStr, nParticles['dm'] )
ax[0,0].set_title( title )
fName = outputKernel + 'densKernelPilot_{1}_h{0:.2f}.dat'.format(h_pilot, pType)
#fig.savefig( 
fig.show()
































######################################################  
##Plot Profiles
#output =  currentDirectory + '/profiles/h_{0}/{1}/'.format(hId, binning)  
#if not os.path.exists( output ): os.makedirs( output )
##Plot profiles
#if binning == 'equiPart': 
  #title = 'NperBin: {0}'.format( nPerBin )
  #name = '_{0}_{1}'.format( hId, nPerBin )
#if binning == 'exp' or binning=='pow': 
  #title = 'nBins: {0}'.format( nBins )
  #name = '_{0}_{1}'.format( hId, nBins )
#plots.plotDensity_p( profiles, haloRvir, haloRvmax, output=output, title=title, name=name, addFine=False, show=False)
#plots.plotKinemat_p( profiles, haloRvir, haloRvmax, output=output, title=title, name=name )








#fig, ax = plt.subplots( 2, 2 )
#fig.set_size_inches(16,10)
#ax[0,0].set_xscale('log')
#ax[0,0].set_yscale('log')
#ax[0,0].plot( r_bin, dens_bin, '-o' )
#ax[0,0].plot( r_kernel, dens_kernel, '-o' )
#ax[0,0].axvline( x=haloRvmax, color='m')
#ax[0,0].set_xlim(r_bin[0], r_bin[-1])

#ax[1,0].set_xscale('log')
#ax[1,0].errorbar( r_bin, sigmaR_bin, yerr=sigmaR_bin/np.sqrt(profiles[pType]['nPart_bin'])  )
#ax[1,0].plot( r_kernel, sigmaR_intrp_kernel, '-o' )
#ax[1,0].axvline( x=haloRvmax, color='m')
#ax[1,0].set_xlim(r_bin[0], r_bin[-1])

#ax[0,1].set_xscale('log')
#ax[0,1].set_yscale('log')
#ax[0,1].plot( dist_dm_halo, acumMass_dm_halo )
#ax[0,1].plot( dist_all_halo, acumMass_all_halo )
#ax[0,1].plot( jR_kernel, jM_int_kernel, 'o--' )
#ax[0,1].plot( jR_kernel, jM, 'o--' )
#ax[0,1].plot( jR_bin, jM_int_bin, 'o--' )
#ax[0,1].axvline( x=haloRvmax, color='m')
#ax[0,1].set_xlim(jR_kernel[0]*0.9, jR_kernel[-1]*1.1)
##ax[0,1].set_ylim(1e7, 2e10)

##fig.show()

























#fig, ax = plt.subplots( 1, 2 )
#fig.set_size_inches(22,9)
#ax[0].set_xscale('log')
#ax[0].set_yscale('log')
#ax[0].plot( dist_dm_center, acumMass_dm_center )
#ax[0].plot( dist_st_center, acumMass_star_center )
##ax[0].plot( dist_all_center, acumMass_all_center )
#ax[0].plot( dist_dm_halo, acumMass_dm_halo )
#ax[0].plot( dist_st_halo, acumMass_star_halo )
#ax[0].plot( dist_all_halo, acumMass_all_halo )
##ax[0].plot( jR_samples, jM_int_samples, 'o--' )
##ax[0].plot( jR_samples, jM_int_samples_lessStar, 'o--' )
##ax[0].plot( jR_samples, jM_int_samples_densAndSigma, 'o--' )
##ax[0].plot( jR_samples, jM_intBvar_samples, 'o--' )
#ax[0].plot( jR_bin, jM_int_bin, 'o--' )
#ax[0].axvline( x=haloRvmax, color='m')
##ax[0].set_xlim(jR_samples[0], jR_samples[-1])
##ax[0].set_ylim(1e7, 2e10)
#ax[0].legend(['dm_cen', 'st_c', 'dm_h', 'st_h', 'all_h'], loc=0)
#title = 'hId:{2}, mD:{0:1.1e}, mS:{3:1.1e}, d:{1:.0f}, nS:{4:1.1e}, off:{5}'.format(haloMass, distToHost, hId, sMass, nParticles['star'], offsetsStr )
#ax[0].set_title(title)

#ax[1].set_xscale('log')
#ax[1].set_yscale('log')
#ax[1].plot( dist_dm_halo, acumMass_dm_halo )
#ax[1].plot( dist_all_halo, acumMass_all_halo )
##ax[1].plot( jR_samples, jM_int_samples, 'o--' )
##ax[1].plot( jR_samples, jM_int_samples_lessStar, 'o--' )
##ax[1].plot( jR_samples, jM_intBvar_samples, 'o--' )
#ax[1].plot( jR_bin, jM_int_bin, 'o--' )
#ax[1].axvline( x=haloRvmax, color='m')
##ax[1].set_xlim(jR_samples[0], jR_samples[-1])
#ax[1].set_ylim(1e7, 2e10)
#fig.savefig(samplingProfileFile[:samplingProfileFile.find('.dat')]+'.png')
##fig.show()






















#plotProfiles = True
#if plotProfiles:
  ##Plot
  #fig, ax = plt.subplots( 2, 2 )
  #fig.set_size_inches(16,10)
  #ax[0,0].set_xscale('log')
  #ax[0,0].set_yscale('log')
  #ax[0,0].plot( r_bin, profiles[pType]['dens_bin'], '-o' )
  ##ax[0,0].errorbar( r_samples, dens_samples, yerr=profiles[pType]['densSigma_samples']  )
  #ax[0,0].axvline( x=haloRvmax, color='m')
  #ax[0,0].set_xlim(r_bin[0], r_bin[-1])

  #ax[1,0].set_xscale('log')
  #ax[1,0].errorbar( r_bin, sigmaR_bin, yerr=sigmaR_bin/np.sqrt(profiles[pType]['nPart_bin'])  )
  ##ax[1,0].plot( r_samples, sigmaR_samples, 'o-' )
  ##ax[1,0].plot( r_bin, sigmaR_intrp_bin, 'o-' )
  ##ax[1,0].plot( r_samples, sigmaR_intrp_samples, 'o-' )
  ##ax[1,0].errorbar( r_bin, sigmaT_bin, yerr=sigmaT_bin/np.sqrt(profiles[pType]['nPart_bin'])  )
  ##ax[1,0].plot( r_samples, sigmaT_intrp_samples, 'o-' )
  #ax[1,0].axvline( x=haloRvmax, color='m')
  #ax[1,0].set_xlim(r_bin[0], r_bin[-1])

  #ax[0,1].set_xscale('log')
  #ax[0,1].set_yscale('log')
  #ax[0,1].plot( r_bin, -dens_der_bin, 'o-' )
  ##ax[0,1].plot( r_samples, -dens_der_samples, 'o-' )
  #ax[0,1].axvline( x=haloRvmax, color='m')
  #ax[0,1].set_xlim(r_bin[0], r_bin[-1])

  #ax[1,1].set_xscale('log')
  #ax[1,1].plot( r_bin, sigmaR2_der_bin, 'o-' )
  ##ax[1,1].plot( r_samples, sigmaR2_der_samples, 'o-' )
  #ax[1,1].axvline( x=haloRvmax, color='m')
  #ax[1,1].set_xlim(r_bin[0], r_bin[-1])
  #title = 'mD:{0:1.1e}, mS:{3:1.1e}, d:{1:.0f} kpc, nS:{4:1.1e}'.format(haloMass, distToHost, hId, sMass, nParticles['star'] )
  #ax[0,0].set_title( title )
  #fig.savefig(output + 'prf_' + fileName[:fileName.find('.dat')]+'.png')
  ##fig.show()

#########################################################
#########################################################
##Jeans Mass differential
#jDens     = dens_samples
#jDensDer  = dens_der_samples
#jSigmaR2  = sigmaR_intrp_samples**2
#jSigR2Der = sigmaR2_der_samples
#jSigmaT2  = sigmaT_intrp_samples**2
#jM_dif_samples    = jeansMass_dif( jR_samples, jDens, jDensDer, jSigmaR2, jSigR2Der, jSigmaT2 )
##Jeans Mass differential
#jDens     = dens_bin
#jDensDer  = dens_der_bin
#jSigmaR2  = sigmaR_intrp_bin**2
#jSigR2Der = sigmaR2_der_bin
#jSigmaT2  = sigmaT_intrp_bin**2
#jM_dif_bin    = jeansMass_dif( jR_bin, jDens, jDensDer, jSigmaR2, jSigR2Der, jSigmaT2 )

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
##ax.legend([l1, l2],['samp','bin'], loc=0)
#ax.set_title('Jeans Diff')
#fig2.show()
#########################################################
#########################################################





















#pType = 'star'
#r        = profiles[pType]['binCenters']
#r_fine   = profiles[pType]['r_fine']
#r_smooth = profiles[pType]['r_smooth']

#density = profiles[pType]['density']
#dens_fine = profiles[pType]['dens_fine']
#dens_smooth = profiles[pType]['dens_smooth']
#density_der = derivative( r, density )
#dens_sm_der = derivative( r_smooth, dens_smooth )

#sigmaR   = profiles[pType]['vel_rad']
#sigmaR2  = profiles[pType]['vel_rad']**2
#sigmaR_fine = profiles[pType]['sigmaR_fine']
#sigmaR2_fine = sigmaR_fine**2
#sigmaR_smooth = profiles[pType]['sigmaR_smooth']
#sigmaR2_smooth = sigmaR_smooth**2
#sigmaR2_der = derivative( r, sigmaR2 )
#sigmaR2_sm_der = derivative( r_smooth, sigmaR2_smooth )

#sigmaT  = profiles[pType]['vel_phi']
#sigmaT2 = profiles[pType]['vel_phi']**2
#sigmaT_fine = profiles[pType]['sigmaT_fine']
#sigmaT2_fine = sigmaT_fine**2
#sigmaT_smooth = profiles[pType]['sigmaT_smooth']
#sigmaT2_smooth = sigmaT_smooth**2

#nPart   = profiles[pType]['nPartDist']
######################################################
#def spline( x, y, x_i ):
  #spline = UnivariateSpline( x, y, k=2, s=20 )
  ##spline = InterpolatedUnivariateSpline( x, y, k=5 )
  ##spline = Rbf( x, y, function='thin_plate', ) 
  #y_i = spline( x_i )
  #return y_i
######################################################
#def rbf( x, y, x_i ):
  #spline = Rbf( x, y, function='thin_plate', ) 
  #y_i = spline( x_i )
  #return y_i
######################################################
##Interpolation using splines 
##for sigmas rbf-thin_plate 
#nInterp= 100
#r_intrp = np.linspace( r[0], r[-1], nInterp, endpoint=True )
#r_intrp = r_smooth.copy()
#sigmaR_intrp  = spline( r, sigmaR,  r_intrp )
#sigmaT_intrp  = spline( r, sigmaT,  r_intrp )
#sigmaR2_intrp = sigmaR_intrp**2
#sigmaT2_intrp = sigmaT_intrp**2
#sigmaR2_intrp_der  = derivative( r_intrp, sigmaR2_intrp )

#######################################################
##Jeans Calc
#G    = 6.67384e-11 #m**3/(kg*s**2)
#mSun = 1.98e30     #kg
#pc   = 3.08e16     #m
#G = G / (1e3*pc) / 1e3**2 * mSun

########################################################
##Jeans Diff	
#jR       = r
#jDensity = density
#jDensDer = density_der
#jSigmaR2 = sigmaR2
#jSigR2Der = sigmaR2_der
#jSigmaT2 = sigmaT2

#Diff = jDensity * jSigR2Der + jSigmaR2 * jDensDer
##Diff =                        jSigmaR2.mean() * jDensDer
#jM = 1./G * ( jR**2 / jDensity * Diff + 2 * jR * ( jSigmaR2 - jSigmaT2 ) )
##jM = jR**2 / ( G * jDensity ) * Diff 
#jeansMassDiff_0 = -jM
#jR_0 = jR.copy()

#jR       = r_smooth
#jDensity = dens_smooth
#jDensDer = dens_sm_der
#jSigmaR2 = sigmaR2_intrp
#jSigR2Der = sigmaR2_intrp_der
#jSigmaT2 = sigmaT2_intrp

#Diff = jDensity * jSigR2Der + jSigmaR2 * jDensDer
##Diff =                        jSigmaR2 * jDensDer
#jM = 1./G * ( jR**2 / jDensity * Diff + 2 * jR * ( jSigmaR2 - jSigmaT2 ) )
##jM = jR**2 / ( G * jDensity ) * Diff 
#jeansMassDiff_1 = -jM
#jR_dif = jR.copy()

#########################################################
##Jeans Int	

#jR       = r
#jDeltaR  = profiles[pType]['binWidths'] 
#jDensity = density
#jSigmaR2 = sigmaR2
#beta = betaVal[pType]
#nPoints = len(jR)
#jeansMassInt_0 = np.zeros( nPoints )
#for n in range(nPoints-1,-1,-1):
  #r_n   = jR[n]
  #dr_n  = jDeltaR[n]
  ##if n == nPoints-1: dr_n = jR[n]-jR[n-1]
  ##else: dr_n  = jR[n] if n==0 else (jR[n+1]-jR[n-1])/2.
  #rho_n = jDensity[n]
  #r_right    = jR[n+1:]
  #rho_right  = jDensity[n+1:]
  #mass_right = jeansMassInt_0[n+1:]
  ##beta_right = betaVal
  #integrand  = rho_right * mass_right * r_right**(2*beta-2)
  #integral = 0 if len(integrand)==0 else simps( integrand, r_right )
  #sigmaR2_n = jSigmaR2[n]
  ##beta_n    = betaVal[pType]
  #jeansMassInt_0[n] = sigmaR2_n * r_n**2 / ( G * dr_n ) - integral / ( rho_n * r_n**(2*beta-2) * dr_n )
#jR_int_0 = jR.copy()




#jR       = r_smooth
#jDeltaR  = profiles[pType]['widths_smooth'] 
#jDensity = dens_smooth
#jSigmaR2 = sigmaR2_smooth
#beta = betaVal[pType]
#nPoints = len(jR)
#jeansMassInt_1 = np.zeros( nPoints )
#for n in range(nPoints-1,-1,-1):
  #r_n   = jR[n]
  #dr_n  = jDeltaR[n]
  ##if n == nPoints-1: dr_n = jR[n]-jR[n-1]
  ##else: dr_n  = jR[n] if n==0 else (jR[n+1]-jR[n-1])/2.
  #rho_n = jDensity[n]
  #r_right    = jR[n+1:]
  #rho_right  = jDensity[n+1:]
  #mass_right = jeansMassInt_1[n+1:]
  ##beta_right = betaVal
  #integrand  = rho_right * mass_right * r_right**(2*beta-2)
  #integral = 0 if len(integrand)==0 else simps( integrand, r_right )
  #sigmaR2_n = jSigmaR2[n]
  ##beta_n    = betaVal[pType]
  #jeansMassInt_1[n] = sigmaR2_n * r_n**2 / ( G * dr_n ) - integral / ( rho_n * r_n**(2*beta-2) * dr_n )
#jR_int_1 = jR.copy()


######################################################
##Accum Mass
#dist_dm = particles['dm']['dist']
#acumMass_dm = particles['dm']['mass'].cumsum()

#dist_all = np.concatenate( [ dist_dm, particles['star']['dist'] ] )
#mass_all = np.concatenate( [ particles['dm']['mass'], particles['star']['mass'] ] )
#sorted_all = dist_all.argsort()
#dist_all = dist_all[sorted_all]
#mass_all = mass_all[sorted_all]
#acumMass_all = mass_all.cumsum()

######################################################
#fig1 = plt.figure()
#fig1.clf()
#ax = fig1.add_subplot(111)
#ax.set_xscale('log')
#ax.set_yscale('log')
#ax.plot( dist_dm, acumMass_dm )
#ax.plot( dist_all, acumMass_all )
##ax.plot( jR_0, jeansMassDiff_0, 'o--' )
##ax.plot( jR_dif, jeansMassDiff_1, 'o--' )
#ax.plot( jR_int_0, jeansMassInt_0, 'o--' )
#ax.plot( jR_int_1, jeansMassInt_1, 'o--' )
#fig1.show()
















#################################################################
##Plot
#fig, ax = plt.subplots( 2, 2 )
#fig.set_size_inches(16,10)
#ax[0,0].loglog( r, density, 'o' )
#ax[0,0].loglog( r_s, density_s, '-'   )
#ax[0,0].set_xlim(r[0], r[-1])

#ax[0,1].set_xscale('log')
#ax[0,1].set_yscale('log')
#ax[0,1].plot( r, -density_der, '-o' )
##ax[0,1].plot( r_s, -density_s_der, '-' )
#ax[0,1].plot( r_s, -density_der_s, '-' )
#ax[0,1].set_xlim(r[0], r[-1])

#ax[1,0].set_xscale('log')
#ax[1,0].errorbar( r, sigmaR,  yerr=sigmaR/np.sqrt(nPart) )
#ax[1,0].errorbar( r, sigmaT,  yerr=sigmaT/np.sqrt(nPart) )
#ax[1,0].plot( r_s, sigmaR_s, '-'   )
#ax[1,0].plot( r_s, sigmaT_s, '-'   )
#ax[1,0].set_xlim(r[0], r[-1])

##ax[1,1].set_xscale('log')
##ax[1,1].plot( r, sigmaR_der, '-o' )
##ax[1,1].plot( r_s, sigmaR_s_der, '-' )
##ax[1,1].plot( r_s, sigmaR2_der, '-' )
##ax[1,1].plot( r_s, 2*sigmaR_s*sigmaR_s_der, '-' )
#ax[1,1].set_xscale('log')
#ax[1,1].set_yscale('log')
#ax[1,1].plot( dist_dm, acumMass_dm )
#ax[1,1].plot( dist_all, acumMass_all )
#ax[1,1].plot( jR, jeansMassDiff, 'o' )
#ax[1,1].plot( jR, jeansMassInt, 'o' )
#ax[1,1].set_xlim(r[0], r[-1])
#ax[1,1].set_ylim(jeansMassDiff[0], jeansMassDiff[-1])
#fig.subplots_adjust(hspace=0)

#if binning == 'equiPart':  
  #title = 'NperBin: {0}'.format( nPerBin )
  #name = 'jeans_{0}'.format(nPerBin) 
#if binning == 'exp' or binning == 'pow':  
  #title = 'nBins: {0}'.format( nBins )
  #name = 'jeans_{0}'.format(nBins) 
#ax[0,0].set_title(title)
#fig.savefig(output+ name +'.png')
#fig.show()







