from snapAnalysis import *
from calculus import *
#from interpolation import *
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
from scipy.interpolate import UnivariateSpline

snapshot = 3
pType    = 'star'
  


#Host halo
hostId  = ms_hostId[snapshot][0]
hostPos = ms_halosData[snapshot][ hostId ].attrs['pos']*1e3


#Halo properties
#hId   = hostId
hId   = '1490'
#hId   = '1825'
#hId   = '1851'

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

  maxDist = { 'star': haloRvir, 'dm': haloRvir }
  distMask = np.where( distToCenter < maxDist[pType] )[0]
  distToCenter   = distToCenter[distMask]
  inHaloPartPos  = particlesPos[distMask]
  inHaloPartVel  = particlesVel[distMask]
  inHaloPartMass = particlesMass[distMask]
  posRel         = posRel[distMask]
  velRel	 = velRel[distMask]
  nParticles = inHaloPartMass.shape[0]
  
  radial_vec, theta_vec, phi_vec = spherical_vectors( posRel )  #Tangential is phi vector NOT THETA
  vel_radial_val = np.array([ radial_vec[i].dot(velRel[i]) for i in range(len(distToCenter)) ]) 
  vel_tangen_val = np.array([    phi_vec[i].dot(velRel[i]) for i in range(len(distToCenter)) ]) #TANGENTIAL
  vel_theta_val  = np.array([  theta_vec[i].dot(velRel[i]) for i in range(len(distToCenter)) ])
  sigma_rad2 = ( ( vel_radial_val - vel_radial_val.mean() )**2 ).mean()
  sigma_tan2 = ( ( vel_tangen_val - vel_tangen_val.mean() )**2 ).mean()
  betaVal = 1 - ( sigma_tan2 / sigma_rad2 )

  #Profiles
  binning = 'exp'
  nPerBin = 200
  rMin, rMax = .110, distToCenter.max()
  nBins = nParticles / nPerBin
  if binning == 'pow' or binning == 'exp': nBins = 20
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
plots.plotDensity_p( profiles, haloRvir, haloRvmax, output=output, title=title, name=name, addDW=False)
#plots.plotKinemat_p( profiles, haloRvir, haloRvmax, output=output, title=title, name=name )


pType = 'star'
#interpolate profiles
r       = profiles[pType]['binCenters']
regionMask = np.where( (r > 0.2) * (r < haloRvir*2 ) )[0]
r       = r[regionMask]
dr      = profiles[pType]['binWidths'][regionMask]
density = profiles[pType]['density'][regionMask]
sigmaR  = profiles[pType]['vel_rad'][regionMask]
sigmaT  = profiles[pType]['vel_phi'][regionMask]
beta    = profiles[pType]['beta'][regionMask]
sigmaR2 = sigmaR**2
sigmaT2 = sigmaT**2
nPart   = profiles[pType]['nPartDist'][regionMask]
density_der = derivative( r, density )
sigmaR2_der = derivative( r, sigmaR2 )




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
#for sigmas rbf-thin_plate 

#Smoothing
nSmooth= 20
r_s = np.linspace( r[0], r[-1], nSmooth, endpoint=True )
r_s = r.copy()
density_s = spline( r, density, r_s )
density_der_s = spline( r, density_der, r_s )
density_s_der = derivative( r_s, density_s )

sigmaR_s  = spline( r, sigmaR,  r_s )
sigmaR2_s = sigmaR_s**2
sigmaR2_s_der = derivative( r_s, sigmaR2_s )

sigmaT_s  = spline( r, sigmaT,  r_s )
sigmaT2_s = sigmaT_s**2
######################################################
#Jeans Calc
G    = 6.67384e-11 #m**3/(kg*s**2)
mSun = 1.98e30     #kg
pc   = 3.08e16     #m
G = G / (1e3*pc) / 1e3**2 * mSun

nPoints = r_s.shape[0]
#jeansCalc = 'jeansDiff'

######################################################
#Jeans Diff	    
jR       = r
jDensity = density
jDensDer = density_der
jSigmaR2 = sigmaR2
jSigR2Der = sigmaR2_der
jSigmaT2 = sigmaT2

#jR       = r_s
#jDensity = density_s
#jDensDer = density_der_s
#jSigmaR2 = sigmaR2_s
#jSigR2Der = sigmaR2_s_der
#jSigmaT2 = sigmaT2_s



Diff = jDensity * jSigR2Der + jSigmaR2 * jDensDer
#Diff =                        jSigmaR2.mean() * jDensDer
jM = 1./G * ( jR**2 / jDensity * Diff + 2 * jR * ( jSigmaR2 - jSigmaT2 ) )
#jM = jR**2 / ( G * jDensity ) * Diff 
jeansMassDiff = -jM

#jM = r_n**2 / ( G * rho_n ) * Diff + 2*r_n/G*( sigma_r2_n - sigma_t2_n )

######################################################
#Jeans Int	

jR       = r
jDensity = density
jDensDer = density_der
jSigmaR2 = sigmaR2
jSigR2Der = sigmaR2_der
jSigmaT2 = sigmaT2

#jR       = r_s
#jDensity = density_s
#jDensDer = density_der_s
#jSigmaR2 = sigmaR2_s
#jSigR2Der = sigmaR2_s_der
#jSigmaT2 = sigmaT2_s

nPoints = len(jR)
jeansMassInt = np.zeros( nPoints )
for n in range(nPoints-1,-1,-1):
  r_n   = jR[n]
  dr_n  = dr[n]
  rho_n = jDensity[n]
  #dr_n  = jR[n] if n==0 else jR[n]-jR[n-1]
  r_right    = jR[n+1:]
  rho_right  = jDensity[n+1:]
  mass_right = jeansMassInt[n+1:]
  beta_right = betaVal
  integrand  = rho_right * mass_right * r_right**(2*beta_right-2)
  integral = 0 if len(integrand)==0 else simps( integrand, r_right )
  sigmaR2_n = jSigmaR2[n]
  beta_n    = betaVal
  jeansMassInt[n] = sigmaR2_n * r_n**2 / ( G * dr_n ) - integral / ( rho_n * r_n**(2*beta_n-2) * dr_n )


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

################################################################
#Plot
fig, ax = plt.subplots( 2, 2 )
fig.set_size_inches(16,10)
ax[0,0].loglog( r, density, 'o' )
ax[0,0].loglog( r_s, density_s, '-'   )
ax[0,0].set_xlim(r[0], r[-1])

ax[0,1].set_xscale('log')
ax[0,1].set_yscale('log')
ax[0,1].plot( r, -density_der, '-o' )
#ax[0,1].plot( r_s, -density_s_der, '-' )
ax[0,1].plot( r_s, -density_der_s, '-' )
ax[0,1].set_xlim(r[0], r[-1])

ax[1,0].set_xscale('log')
ax[1,0].errorbar( r, sigmaR,  yerr=sigmaR/np.sqrt(nPart) )
ax[1,0].errorbar( r, sigmaT,  yerr=sigmaT/np.sqrt(nPart) )
ax[1,0].plot( r_s, sigmaR_s, '-'   )
ax[1,0].plot( r_s, sigmaT_s, '-'   )
ax[1,0].set_xlim(r[0], r[-1])

#ax[1,1].set_xscale('log')
#ax[1,1].plot( r, sigmaR_der, '-o' )
#ax[1,1].plot( r_s, sigmaR_s_der, '-' )
#ax[1,1].plot( r_s, sigmaR2_der, '-' )
#ax[1,1].plot( r_s, 2*sigmaR_s*sigmaR_s_der, '-' )
ax[1,1].set_xscale('log')
ax[1,1].set_yscale('log')
ax[1,1].plot( dist_dm, acumMass_dm )
ax[1,1].plot( dist_all, acumMass_all )
ax[1,1].plot( jR, jeansMassDiff, 'o' )
ax[1,1].plot( jR, jeansMassInt, 'o' )
ax[1,1].set_xlim(r[0], r[-1])
ax[1,1].set_ylim(jeansMassDiff[0], jeansMassDiff[-1])
fig.subplots_adjust(hspace=0)

if binning == 'equiPart':  
  title = 'NperBin: {0}'.format( nPerBin )
  name = 'jeans_{0}'.format(nPerBin) 
if binning == 'exp' or binning == 'pow':  
  title = 'nBins: {0}'.format( nBins )
  name = 'jeans_{0}'.format(nBins) 
ax[0,0].set_title(title)
fig.savefig(output+ name +'.png')
fig.show()







