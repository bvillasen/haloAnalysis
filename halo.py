from snapAnalysis import *
from calculus import *
from interpolation import *
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
from scipy.interpolate import UnivariateSpline

snapshot = 3
pType    = 'star'
  
#Host halo
hostId  = ms_hostId[snapshot][0]
hostPos = ms_halosData[snapshot][ hostId ].attrs['pos']*1e3

#Halo properties
#hId   = hostId
#hId   = '1490'
#hId   = '1825'
hId   = '1851'

#haloCenter = ms_centers[snapshot][130][pType][hId]['CM']
haloMass   = ms_halosData[snapshot][ hId ].attrs['mbound_vir']
haloPos    = ms_halosData[snapshot][ hId ].attrs['pos']*1e3
haloVel    = ms_halosData[snapshot][ hId ].attrs['vel']
haloRvir   = ms_halosData[snapshot][ hId ].attrs['rvir']*1e3
haloRvmax  = ms_halosData[snapshot][ hId ].attrs['rvmax']*1e3
haloVmax   = ms_halosData[snapshot][ hId ].attrs['vmax']
distToHost = np.sqrt( np.sum( (haloPos - hostPos)**2, axis=-1) )

profiles = {}
particles = {}
centers = {}
betaVal = {}
for pType in ['star', 'dm']:
  profiles[pType] = {}
  particles[pType] = {}

  #Particles
  particlesPos  = ms_partsData[snapshot][pType]['pos'][...]*1e3
  particlesVel  = ms_partsData[snapshot][pType]['vel'][...]
  particlesMass = ms_partsData[snapshot][pType]['mass'][...]

  #In halo particles
  inHaloPartIds  = ms_halosPartsIds[snapshot][ hId ][pType+'_bound'][...] 
  inHaloPartPos  = particlesPos[inHaloPartIds]
  inHaloPartVel  = particlesVel[inHaloPartIds]
  inHaloPartMass = particlesMass[inHaloPartIds]
  
  ##Center
  #tree = KDTree( inHaloPartPos )
  #print " Initial sphere:"  
  #ball_pos = haloPos
  #massRatio = 0.6 if pType == 'star' else 0.6
  #ball_radius, massRatio_i = getBallFromMassRatio( haloPos, haloRvmax, massRatio, inHaloPartMass, tree )
  #print '  massRatio_i: {0}'.format(massRatio_i)
  #print '  ball_radius: {0}, rvmax: {1}, rvir: {2}'.format( ball_radius, haloRvmax, haloRvir )
  #rLast = 0.150
  #nPartLast = 50
  #center, radius_f, nInBall_f = findCenter_CM( ball_pos, ball_radius, rLast, nPartLast, inHaloPartMass, inHaloPartPos, tree )
  #print '   cm:{1} n:{0} r:{2}'.format(nInBall_f, center, radius_f)            
  #haloCenter = moveBall( center, radius_f, inHaloPartMass, inHaloPartPos, tree )
  center = ms_centers[snapshot][130][pType][hId]['CM']
  
  #Particles properties
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

  particles[pType]['mass'] = inHaloPartMass
  particles[pType]['dist'] = distToCenter

  #maxDist = { 'star': haloRvir, 'dm': haloRvir }
  #distMask = np.where( distToCenter < maxDist[pType] )[0]
  #distToCenter   = distToCenter[distMask]
  #inHaloPartPos  = particlesPos[distMask]
  #inHaloPartVel  = particlesVel[distMask]
  #inHaloPartMass = particlesMass[distMask]
  #posRel         = posRel[distMask]
  #velRel	 = velRel[distMask]
  nParticles = inHaloPartMass.shape[0]
  
  #Get velocity components
  radial_vec, theta_vec, phi_vec = spherical_vectors( posRel )  #Tangential is phi vector NOT THETA
  vel_radial_val = np.array([ radial_vec[i].dot(velRel[i]) for i in range(len(distToCenter)) ]) 
  vel_tangen_val = np.array([    phi_vec[i].dot(velRel[i]) for i in range(len(distToCenter)) ]) #TANGENTIAL
  vel_theta_val  = np.array([  theta_vec[i].dot(velRel[i]) for i in range(len(distToCenter)) ])
  sigma_rad2 = ( ( vel_radial_val - vel_radial_val.mean() )**2 ).mean()
  sigma_tan2 = ( ( vel_tangen_val - vel_tangen_val.mean() )**2 ).mean()
  betaVal[pType] = 1 - ( sigma_tan2 / sigma_rad2 )

  #Profiles
  binning = 'equiPart'
  nPerBin = 200
  rMin, rMax = .110, distToCenter.max()
  nBins = nParticles / nPerBin
  if binning == 'pow' or binning == 'exp': nBins = 20
  binCenters, binWidths, binVols, densProf, velR_avr, velTH_avr, velT_avr, velProf_rad, velProf_theta, velProf_phi, betaDist, nPartDist = mha.get_profiles( rMin, rMax,
	      distToCenter, inHaloPartMass, vel_radial_val, vel_theta_val, vel_tangen_val,
	      nBins=nBins, binning=binning )

  profiles[pType]['density'] = densProf
  #Velocity Avrgs
  profiles[pType]['vel_rad_avr'] = velR_avr
  profiles[pType]['vel_theta_avr'] = velTH_avr
  profiles[pType]['vel_phi_avr'] = velT_avr
  #Velociy sigma
  profiles[pType]['vel_rad'] = velProf_rad
  profiles[pType]['vel_theta'] = velProf_theta
  profiles[pType]['vel_phi'] = velProf_phi
  profiles[pType]['beta'] = betaDist
  profiles[pType]['nPartDist'] = nPartDist
  profiles[pType]['binCenters'] = binCenters
  profiles[pType]['binWidths'] = binWidths
  profiles[pType]['binVols'] = binVols
  
  #Fine profiles
  npLeft = 100
  npRight = 80				
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
  nAverg = 200
  step = 200
  r_smooth , dens_smooth = smooth( rProf_f, dProf_f, nAverg, step )
  r_smooth , sigmaR_smooth = smooth( rProf_f, sigmaR_f, nAverg, step )
  r_smooth , sigmaT_smooth = smooth( rProf_f, sigmaT_f, nAverg, step )
  profiles[pType]['r_smooth'] = r_smooth	
  profiles[pType]['dens_smooth'] = dens_smooth
  profiles[pType]['sigmaR_smooth'] = sigmaR_smooth
  profiles[pType]['sigmaT_smooth'] = sigmaT_smooth

#####################################################  
output =  currentDirectory + '/profiles/h_{0}/{1}/'.format(hId, binning)  
if not os.path.exists( output ): os.makedirs( output )
#Plot profiles
if binning == 'equiPart': 
  title = 'NperBin: {0}'.format( nPerBin )
  name = '_{0}_{1}'.format( hId, nPerBin )
if binning == 'exp' or binning=='pow': 
  title = 'nBins: {0}'.format( nBins )
  name = '_{0}_{1}'.format( hId, nBins )
plots.plotDensity_p( profiles, haloRvir, haloRvmax, output=output, title=title, name=name, addFine=True, show=False)
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
#####################################################
def spline( x, y, x_i ):
  spline = UnivariateSpline( x, y, k=2, s=20 )
  #spline = InterpolatedUnivariateSpline( x, y, k=5 )
  #spline = Rbf( x, y, function='thin_plate', ) 
  y_i = spline( x_i )
  return y_i
#####################################################
def rbf( x, y, x_i ):
  spline = Rbf( x, y, function='thin_plate', ) 
  y_i = spline( x_i )
  return y_i
#####################################################
#Interpolation using splines 
#for sigmas rbf-thin_plate 
nInterp= 100
r_intrp = np.linspace( r[0], r[-1], nInterp, endpoint=True )
r_intrp = r_smooth.copy()
sigmaR_intrp  = spline( r, sigmaR,  r_intrp )
sigmaT_intrp  = spline( r, sigmaT,  r_intrp )
sigmaR2_intrp = sigmaR_intrp**2
sigmaT2_intrp = sigmaT_intrp**2
sigmaR2_intrp_der  = derivative( r_intrp, sigmaR2_intrp )

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
jSigmaR2 = sigmaR2_intrp
jSigR2Der = sigmaR2_intrp_der
jSigmaT2 = sigmaT2_intrp

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

#####################################################
fig1 = plt.figure()
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







