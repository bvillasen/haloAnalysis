import numpy as np
from scipy.optimize import curve_fit
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import os, sys, tempfile, time
from subprocess import call, Popen, PIPE
import h5py as h5

currentDirectory = os.getcwd()
#Add Modules from other directories
parentDirectory = currentDirectory[:currentDirectory.rfind("/")]
toolsDirectory = currentDirectory + "/tools"
sys.path.append( toolsDirectory )
import multi_haloAnalysis as mha
import tools
from ms_haloTable import writeHaloTable

paramNumber = 'p02'

plotDensityProjection = False
plotDensityProfile = False
fileHalos = False
linkHP = False
writeTables = False
plotMF = False
starsProfiles = False
profiles = False
for option in sys.argv:
  if option == "dproj": plotDensityProjection = True
  if option == "dprof": plotDensityProfile = True
  if option == "prof": profiles = True
  if option == "sprof": starsProfiles = True
  if option == "fileH": fileHalos = True
  if option == "linkHP": linkHP = True
  if option == 'tables': writeTables = True
  if option == 'mfunc': plotMF = True
h = 0.7
DM_particleMass = 92496.7 
positionsChanged = True
sphereCenter = np.array([ 10.8, 1.3, 15.7 ])

dataDir = "/home/bruno/Desktop/data/galaxy/"
halosDir = dataDir + 'halos/{0}/'.format(paramNumber)
outputDir = halosDir + 'multi_snap/'
particlesDir = dataDir + 'particles/'
#Use RKS/examples calc_potentials to get baound pairticls ids
getBoundCommand = '/home/bruno/apps/rockstar-galaxies/examples/calc_potentials'

listFiles = [outputDir + f for f in os.listdir( outputDir ) if f.find(".list")>=0 ]
boundFiles = [outputDir + f for f in os.listdir( outputDir ) if f.find("boundParticles")>=0 ]
snapshots = sorted( [ int(f[f.find('.list')-1]) for f in listFiles ], reverse=True)
snapshots = [ max(snapshots) ]
print "\nAnalysing  snapshots: ", snapshots




if fileHalos:
  ms_halosCatalog = {}
  print "\nLoading list files..."
  start = time.time()
  ms_halosCatalog, n_list = mha.load_listFiles( ms_halosCatalog, outputDir, snapshots  )
  print "Time: ", time.time() - start

  print "\nLoading ascii files..."
  start = time.time()
  ms_halosCatalog, n_ascii = mha.load_asciiFiles( ms_halosCatalog, outputDir, snapshots  )
  print "Time: ", time.time() - start

  print "\nWriting complete halo data file..."
  start = time.time()
  divideByH = [ 'mvir',  'mbound_vir',  'rvir', 'rvmax', 'x', 'y', 'z', 'Rs', 'SM' ]
  kpcToMpc = ['rvir', 'rvmax', 'Rs' ]
  outputFile = outputDir + "/halosData.h5"
  #h5File_halos = h5.File( outputFile, 'w' )
  for snapshot in snapshots:
    print " Snapshot: {0}".format(snapshot)
    #snapshot = str( snapshot )
    outputFile = outputDir + "halosData_{0}.h5".format(snapshot)
    h5File_halos = h5.File( outputFile, 'w' )
    catalog = ms_halosCatalog[snapshot]
    for haloId in catalog['id']:
      #hId = str(hId)
      #print haloId
      halo = h5File_halos.create_group( str( int(haloId) ) )
      for attribute in catalog.keys():
	val = catalog[attribute][haloId]
	if positionsChanged:
	  if attribute == 'x': val += sphereCenter[0]
	  if attribute == 'y': val += sphereCenter[1]
	  if attribute == 'z': val += sphereCenter[2]
	val = val/h if attribute in divideByH else val
	val = val*1e-3 if attribute in kpcToMpc else val
	halo.attrs[attribute] = val
      haloPos = np.array( [ catalog['x'][haloId], catalog['y'][haloId], catalog['z'][haloId] ] )
      #haloPos = ( haloPos - 0.5*rockstarConf['TIPSY_LENGTH_CONVERSION'] )/h
      if positionsChanged: haloPos += sphereCenter
      haloPos = haloPos/h
      halo.attrs['pos'] = haloPos
      halo.attrs['vel'] = np.array( [ catalog['vx'][haloId], catalog['vy'][haloId], catalog['vz'][haloId] ] )
    allHalos = h5File_halos.create_group( 'all' )
    posAll = np.array([catalog['x'], catalog['y'], catalog['z']]).T
    #posAll = ( posAll - 0.5*rockstarConf['TIPSY_LENGTH_CONVERSION'] )/h
    if positionsChanged: posAll += sphereCenter
    posAll = posAll/h
    mvirAll = catalog['mvir']/h
    allHalos.create_dataset( 'pos',  data=posAll,  compression="gzip", compression_opts=9 )
    allHalos.create_dataset( 'mvir', data=mvirAll, compression="gzip", compression_opts=9 )
    h5File_halos.close()
  print " Time: ", time.time() - start   

print '\nLoading halos data...'
start = time.time()
ms_halosData = {}
ms_haloIds_catalog = {}
ms_halosMass = {}
ms_hostId = {}
for snapshot in snapshots:
  #Catalogs
  halosFile = outputDir + 'halosData_{0}.h5'.format(snapshot)
  ms_halosData[snapshot] = h5.File( halosFile, 'r' )
  ms_haloIds_catalog[snapshot]  = [ hId for hId in ms_halosData[snapshot].keys() if hId!='all' ]
  ms_halosMass[snapshot] = [ (hId, ms_halosData[snapshot][hId].attrs['mbound_vir']) for hId in ms_haloIds_catalog[snapshot]]
  ms_hostId[snapshot] = max( ms_halosMass[snapshot], key=lambda x: x[1] )
print " Time: ", time.time() - start 

print "\nLoading particles data..."
start = time.time()
ms_partsData = {}
ms_nPart = {}
for snapshot in snapshots:
  particlesFile = particlesDir + 'pData_{0}.h5'.format(snapshot)
  ms_partsData[snapshot] = h5.File( particlesFile, 'r' )
  ms_nPart[snapshot] = { pType : ms_partsData[snapshot][pType]['mass'].shape[0] for pType in [ 'gas', 'dm', 'star' ]}
print " Time: ", time.time() - start 

if linkHP:
  print "\nLinking particles to halos..."
  start = time.time() 
  for snapshot in snapshots:
    nGas, nDM = ms_nPart[snapshot]['gas'], ms_nPart[snapshot]['dm']
    print ' snapshot {0}:'.format(snapshot)
    print '  Loading particles...'
    parts_halo_data = mha.loadParticlesHalos( snapshot, outputDir )
    print '  Linking particles...'
    halos_particlesIds = mha.linkParticlesHalos_rks( snapshot, parts_halo_data, nGas, nDM )
    print '  Writing to file...'
    outputFile = outputDir + "halosParts_{0}.h5".format(snapshot)
    h5File_halosParts = h5.File( outputFile, 'w' )
    halosIds = [ int(k) for k in ms_halosData[snapshot].keys() if k != 'all' ]
    for haloId in halosIds:
      halo = h5File_halosParts.create_group( str( haloId ) )
      for pType in [ 'star', 'dm' ]:
	inHalosParts_id = halos_particlesIds[pType].get(haloId)
	name = '{0}_rks'.format(pType)
	data = inHalosParts_id if inHalosParts_id != None else np.array([])
	halo.create_dataset( name,  data=data,  compression="gzip", compression_opts=9 )
    del halos_particlesIds
    
    print '  Linking by position...'
    hData = ms_halosData[snapshot]
    pData = ms_partsData[snapshot]
    halosIds = [ int(k) for k in ms_halosData[snapshot].keys() if k != 'all' ]
    posHalos = { hId: hData[str(hId)].attrs['pos'] for hId in halosIds }
    haloRstring = {'dm':['rvir'], 'star':['rvmax', 'rvir10', 'rvir'] }
    print "   Making KDTrees..."
    for pType in [ 'dm', 'star' ]:
      print "    " + pType
      partsPos = pData[pType]['pos'][...] 
      tree = KDTree( partsPos )
      print "     Linking halo-particles..."
      for hId in halosIds:
	halo = h5File_halosParts[ str( hId ) ]
	haloPos =  hData[str(hId)].attrs['pos']
	for radius in haloRstring[pType]: 
	  if radius == 'rvir10': haloRadius = hData[str(hId)].attrs['rvir']/10
	  else: haloRadius = hData[str(hId)].attrs[radius]
	  inHaloIds = tree.query_ball_point( haloPos, haloRadius )
	  name = '{0}_pos_{1}'.format(pType, radius)
	  halo.create_dataset( name,  data=inHaloIds,  compression="gzip", compression_opts=9 )
      del tree

    print "  Getting in-halo bound particles..."
    particlesFile = outputDir + 'halos_{0}.0.particles'.format(snapshot)
    boundOutputFile = outputDir + 'boundParticles_{0}.dat'.format(snapshot)
    if boundOutputFile not in boundFiles: 
      print '  Getting bound particles: {0}'.format(boundOutputFile)
      call([ getBoundCommand, particlesFile, boundOutputFile ])
    boundString = open( boundOutputFile, 'r' ).read().splitlines()
    for line in boundString:
      data = np.fromstring( line, dtype=int, sep=' ')
      hId = data[0]
      halo = h5File_halosParts[ str( hId ) ]
      boundIds = data[1:]
      boundGas  = boundIds[boundIds<nGas]
      boundDM   = boundIds[(boundIds >= nGas) * (boundIds < (nGas+nDM))] - nGas
      boundStar = boundIds[boundIds >= (nGas+nDM)] - (nGas + nDM)
      halo.create_dataset( 'gas_bound',  data=boundGas,  compression="gzip", compression_opts=9 )
      halo.create_dataset( 'dm_bound',  data=boundDM,  compression="gzip", compression_opts=9 )
      halo.create_dataset( 'star_bound',  data=boundStar,  compression="gzip", compression_opts=9 ) 
    del boundString      
    h5File_halosParts.close()  


print '\nLoading halos-particles...'
start = time.time()
ms_halosPartsIds = {}
for snapshot in snapshots:
  halosPartsFile = outputDir + 'halosParts_{0}.h5'.format(snapshot)
  ms_halosPartsIds[snapshot] = h5.File( halosPartsFile, 'r' )
print " Time: ", time.time() - start


print "\nLoading trees hlists files..."
start = time.time()
ms_treesCatalog = {}
ms_treesCatalog, nHalos_tree = mha.load_treeFiles( ms_treesCatalog, outputDir, snapshots )
ms_haloIdsAll = {}  #( originalId , globalID , desc_id ) 
ms_haloIds_trees = {}
ms_globalToOriginalId = {}
ms_originalToDescId = {}
for snapshot in snapshots:
  tCtlg = ms_treesCatalog[snapshot] 
  ms_haloIds_trees[snapshot] = tCtlg['Orig_halo_ID'].astype(int)
  ms_haloIdsAll[snapshot] = zip( tCtlg['Orig_halo_ID'].astype(int), tCtlg['id'].astype(int), tCtlg['desc_id'].astype(int) )
  ms_globalToOriginalId[snapshot] = { ms_haloIdsAll[snapshot][i][1] : ms_haloIdsAll[snapshot][i][0] for i in range(len(ms_haloIdsAll[snapshot]) ) }
  ms_originalToDescId[snapshot] = { ms_haloIdsAll[snapshot][i][0] : ms_haloIdsAll[snapshot][i][2] for i in range(len(ms_haloIdsAll[snapshot]) ) }
print " Time: ", time.time() - start  

print "\nLoading deleted halos..."
start = time.time()
ms_deletedCatalog, nHalos_deleted = mha.load_deadHalos( ms_treesCatalog, outputDir, snapshots )
ms_haloIds_deleted = {}
for snapshot in snapshots:
  if snapshot == len(snapshots)-1: ms_haloIds_deleted[snapshot] = np.array([])
  else: ms_haloIds_deleted[snapshot] = ms_deletedCatalog[snapshot]['Original_ID'].astype(int) 
print " Time: ", time.time() - start

###############################################################
#SELECT WICH HALOS TO USE
ms_haloIds = ms_haloIds_catalog
###############################################################


print '\nLinking halos across snapshots...'
start = time.time()
ms_haloDesc_id = {}
for snapshot in snapshots:
  if snapshot == max(snapshots):
    ms_haloDesc_id[snapshot] = { k:-1 for k in ms_originalToDescId[snapshot].keys() }
  else:
    haloDesc_id = {}
    for hId, globalId, descId in ms_haloIdsAll[snapshot]:
      haloDesc_id[ hId ] = ms_globalToOriginalId[snapshot+1][descId]
    ms_haloDesc_id[snapshot] = haloDesc_id  
print " Time: ", time.time() - start



print '\nFinding halos with stars...'
start = time.time()
ms_halosWithStars = {}
for snapshot in snapshots:
  haloIds = ms_haloIds[snapshot]
  halosWithStars = {}
  inHalosPartsIds = ms_halosPartsIds[snapshot] 
  halosData = ms_halosData[snapshot]
  halosWithStarsAll = []
  for finder in ['pos_rvir', 'pos_rvir10', 'pos_rvmax', 'rks', 'SM', 'bound' ]:
    if finder == 'SM': hList = [ int(hId) for hId in haloIds if halosData[hId].attrs.get('SM')>0  ]
    else: hList = [ int(hId) for hId in inHalosPartsIds.keys() if len(inHalosPartsIds[hId]['star_'+finder])>0  ] 
    halosWithStars[finder] = hList
    halosWithStarsAll.extend( hList )
  halosWithStarsAll = set( halosWithStarsAll )
  ms_halosWithStars[snapshot] = halosWithStars
  ms_halosWithStars[snapshot]['all'] = halosWithStarsAll
print " Time: ", time.time() - start  

print '\nAnalysing in-halo star mass...'
start = time.time()
ms_inHaloStarMass = {}
ms_nStarsInHalos = {}
for snapshot in snapshots:
  inHaloStarMass = {}
  nStarInHalos = {}
  halosWithStars = ms_halosWithStars[snapshot]
  halosData = ms_halosData[snapshot]
  partsData = ms_partsData[snapshot]
  inHalosPartsIds = ms_halosPartsIds[snapshot]
  nGas, nDM = ms_nPart[snapshot]['gas'], ms_nPart[snapshot]['dm']
  starMasses = partsData['star']['mass'][...]
  for finder in [ 'pos_rvir10', 'pos_rvmax', 'rks', 'SM', 'pos_rvir', 'bound' ]:
    if finder == 'SM': inHaloStarMass[finder] = { hId: halosData[str(hId)].attrs['SM'] for hId in halosWithStars['SM']  }
    else:
      inHaloStarMass[finder] = {}
      nStarInHalos[finder] = {}
      for hId in halosWithStars[finder]:
	starsIds = inHalosPartsIds[str(hId)]['star_'+finder][...]
	starsIds.sort()
	stellarMass = (starMasses[starsIds]).sum()
	nStarInHalos[finder][hId] = len(starsIds)
	inHaloStarMass[finder][hId] = stellarMass
  ms_inHaloStarMass[snapshot] = inHaloStarMass
  ms_nStarsInHalos[snapshot] = nStarInHalos
print " Time: ", time.time() - start       

import visual as vi
def vis3d():
  #3D visualization
  
  snapshot = 2
  pPos = ms_partsData[snapshot]['star']['pos'][...]
  limits = np.array([ [pPos[:,0].min(), pPos[:,0].max() ],
		      [pPos[:,1].min(), pPos[:,1].max() ], 
		      [pPos[:,2].min(), pPos[:,2].max() ] ] )
  posRange = limits[:,1]-limits[:,0]
  center = limits.sum( axis=1 )/2.
  scene = vi.display(title='Galaxies', center=center, background=(0,0,0))
  center1 = limits[:,0]
  center2 = limits[:,1]
  center3 = ( limits[0,0], limits[1,0], limits[2,1] )
  center4 = ( limits[0,1], limits[1,1], limits[2,0] )
  center5 = ( limits[0,0], limits[1,1], limits[2,1] )
  center6 = ( limits[0,1], limits[1,0], limits[2,1] )
  r = 0.004
  box = [ vi.cylinder( pos=center1, axis=(posRange[0], 0, 0 ), radius=r ),
	  vi.cylinder( pos=center1, axis=(0, posRange[1], 0 ), radius=r ),
	  vi.cylinder( pos=center1, axis=(0, 0, posRange[2] ), radius=r ),
	  vi.cylinder( pos=center2, axis=(-posRange[0], 0, 0 ), radius=r ),
	  vi.cylinder( pos=center2, axis=(0, -posRange[1], 0 ), radius=r ),
	  vi.cylinder( pos=center2, axis=(0, 0, -posRange[2] ), radius=r ),
	  vi.cylinder( pos=center3, axis=(posRange[0], 0, 0 ), radius=r ),
	  vi.cylinder( pos=center3, axis=(0, posRange[1], 0 ), radius=r ),
	  vi.cylinder( pos=center4, axis=(-posRange[0], 0, 0 ), radius=r ),
	  vi.cylinder( pos=center4, axis=(0, -posRange[1], 0 ), radius=r ),
	  vi.cylinder( pos=center5, axis=(0, 0, -posRange[2] ), radius=r ),
	  vi.cylinder( pos=center6, axis=(0, 0, -posRange[2] ), radius=r ) ]
  hostId  = ms_hostId[snapshot][0]
  hostPos = ms_halosData[snapshot][hostId].attrs['pos']
  hostRadius = ms_halosData[snapshot][hostId].attrs['rvir']
  hostSphere = vi.sphere(pos=hostPos, radius=hostRadius, opacity=0.2)
  satelites = {}
  for hId in ms_halosWithStars[snapshot]['SM']:
    hId = str( hId )
    if hId == hostId: continue 
    sm = ms_halosData[snapshot][ hId ].attrs['SM']
    if sm < 1e6: continue
    satelites[hId] = {}
    inHaloPartIds  = ms_halosPartsIds[snapshot][ hId ]['star_bound'][...] 
    inHaloPartPos  = pPos[inHaloPartIds]
    color = np.random.rand(3)
    satelites[hId]['points'] = vi.points(pos=inHaloPartPos, size=2, color=color)
    disp = np.array([0., 0.1, 0.])
    #satelites[hId]['text'] = vi.text(pos=inHaloPartPos+disp, )
#vis3d()




#Virial threshold
CRITICAL_DENSITY = 2.77519737e11 # 3H^2/8piG in (Msun / h) / (Mpc / h)^3


if profiles:
  print "\nDensity and vel_distb profiles..."
  start = time.time()
  snapshot = 2
  output = outputDir + 'profiles/snap_{0}'.format(snapshot)
  if not os.path.exists( output ): os.makedirs( output )
  output1 = output + '/density'
  if not os.path.exists( output1 ): os.makedirs( output1 )
  output2 = output + '/vel_distb'
  if not os.path.exists( output2 ): os.makedirs( output2 )
  hostId  = ms_hostId[snapshot][0]
  hostPos = ms_halosData[snapshot][ hostId ].attrs['pos']*1e3
  for hId in ms_halosWithStars[snapshot]['SM']:
    #hId = hostId
    hId = str( hId )
    #density
    fig1 = plt.figure(1)
    fig1.clf()
    ax1 = fig1.add_subplot(111)
    #vel_distb
    fig2 = plt.figure(2)
    fig2.clf()
    ax2 = fig2.add_subplot(111)
    #Dens and vel dist
    fig3, ax3 = plt.subplots( 2, sharex=True )
    fig3.set_size_inches(8,10)
    #fig3.clf()
    legends, legends1 = [], []
    minBinCenter, maxBinCenter = 1000, 0
    sMass      = ms_halosData[snapshot][ hId ].attrs['SM']
    if sMass < 1e6: continue
    haloPos    = ms_halosData[snapshot][ hId ].attrs['pos']*1e3
    haloVel    = ms_halosData[snapshot][ hId ].attrs['vel']
    haloRadius = ms_halosData[snapshot][ hId ].attrs['rvir']*1e3
    haloMass   = ms_halosData[snapshot][ hId ].attrs['mbound_vir']
    dist       = np.sqrt( np.sum( (haloPos - hostPos)**2, axis=-1) )
    for pType in ['star','dm']:
      #pType = 'star'
      particlesPos  = ms_partsData[snapshot][pType]['pos'][...]*1e3
      particlesVel  = ms_partsData[snapshot][pType]['vel'][...]
      particlesMass = ms_partsData[snapshot][pType]['mass'][...]
      finder_list = [ 'bound', 'pos_rvir']
      for finder in finder_list:
      #finder = 'bound'
	inHaloPartIds  = ms_halosPartsIds[snapshot][ hId ][pType+ '_' + finder][...] 
	inHaloPartPos  = particlesPos[inHaloPartIds]
	inHaloPartVel  = particlesVel[inHaloPartIds]
	inHaloPartMass = particlesMass[inHaloPartIds]
	posRel       = inHaloPartPos - haloPos
	velRel       = inHaloPartVel - haloVel
	distToCenter = np.sqrt( np.sum(posRel**2, axis=-1) )
	sortedIndx   = distToCenter.argsort()
	distToCenter = distToCenter[sortedIndx]
	posRel       = posRel[sortedIndx]
	velRel       = velRel[sortedIndx]
	posNormalizer = np.tile( distToCenter, (3,1) ).T
	posRel = posRel/posNormalizer #Normalize position vector
	radialVel =np.array([ posRel[i].dot(velRel[i]) for i in range(len(distToCenter)) ]) 
	inHaloPartMass = inHaloPartMass[sortedIndx]
	rMin, rMax = 0.1, distToCenter.max()
	################################################
	#binCenters, densProf, kinmProf = mha.profiles_exp( rMin, rMax, distToCenter, inHaloPartMass, inHaloPartVel,  nBins=20 )
	nBins = 15
	distMaskIds = np.where( ( distToCenter > rMin ) * ( distToCenter <= rMax ) )[0]
	left, right = distMaskIds[0], distMaskIds[-1]
	#binEdges   = np.exp(np.linspace(np.log(1e-1), np.log(rMax*(1.00001)), nBins))
	binEdges    = np.logspace( -1, np.log10(rMax*1.01), base=10, num=nBins)
	#binCenters = np.sqrt(binEdges[:-1]*binEdges[1:])
	binCenters  = np.zeros(nBins-1)
	binVols     = np.zeros(nBins-1)
	hist, histBinEdges  = np.histogram(distToCenter, bins=binEdges)
	shellVol = 4.*np.pi/3*(binEdges[1:]**3 - binEdges[:-1]**3)
	#nParticles = inHaloPartMass.shape[0]
	massDist = np.zeros(nBins-1)
	kinmProf = np.zeros(nBins-1)
	start = left 
	for i in range(nBins-1):
	  end = start + hist[i]
	  binWidth = distToCenter[end-1] - distToCenter[start]
	  binCenter = distToCenter[start:end].mean()
	  binCenters[i] = binCenter
	  binVols[i] = 4. * np.pi * binCenter**2 * binWidth
	  massDist[i] = (inHaloPartMass[start:end]).sum()
	  velsBin = radialVel[start:end]
	  velAvg = velsBin.mean()
	  kinmProf[i] = np.sqrt( ( ( velsBin - velAvg )**2 ).sum()/hist[i] )
	  start = end
	densProf = massDist/shellVol
	###############################################
	minBinCenter = min( binCenters[0], minBinCenter )
	maxBinCenter = max( binCenters[-1], maxBinCenter )
	legends.append(pType + '_' + finder)
	ax1.loglog( binCenters, densProf, '-o' )
	ax2.plot( binCenters, kinmProf, '-o' )
	if (finder=='bound'): 
	  legends1.append(pType + '_' + finder)
	  ax3[0].loglog( binCenters, densProf, '-o' )
	  ax3[1].semilogx( binCenters, kinmProf, '-o' )
    ax1.axvline( x=haloRadius, color='y')
    ax1.set_xlim(minBinCenter, maxBinCenter)
    ax1.set_xlabel(r'r [kpc]')
    ax1.set_ylabel(r'Density $[{\rm M_{\odot} kpc^{-3}  }]$')
    ax1.set_title('id:{2}, mDM:{0:1.1e},  mS:{3:1.1e},  dist:{1:.0f} kpc'.format(haloMass, float(dist), hId, sMass) )
    ax1.legend(legends, loc=0)
    fig1.savefig( output1 + '/dPrf_{0}.png'.format(hId) ) 
    ax2.axvline( x=haloRadius, color='y')
    ax2.set_xlim(minBinCenter, maxBinCenter)
    ax2.set_xscale('log')
    ax2.set_xlabel(r'r [kpc]')
    ax2.set_ylabel(r'Radial Velocity Dispersion $[{\rm  km/seg  }]$')
    ax2.set_title('id:{2}, mDM:{0:1.1e},  mS:{3:1.1e},  dist:{1:.0f} kpc'.format(haloMass, float(dist), hId, sMass) )
    ax2.legend(legends, loc=0)
    fig2.savefig( output2 + '/kPrf_{0}.png'.format(hId) ) 
    ax3[0].axvline( x=haloRadius, color='y')
    ax3[0].set_ylabel(r'Density $[{\rm M_{\odot} kpc^{-3}  }]$')
    ax3[0].set_title('id:{2}, mDM:{0:1.1e},  mS:{3:1.1e},  dist:{1:.0f} kpc'.format(haloMass, float(dist), hId, sMass) )
    ax3[0].legend(legends1, loc=0) 
    ax3[1].axvline( x=haloRadius, color='y')
    ax3[1].set_xlim(minBinCenter, maxBinCenter)
    ax3[1].set_xlabel(r'r [kpc]')
    ax3[1].set_ylabel(r'Radial Velocity Dispersion $[{\rm  km/seg  }]$')
    #ax3[1].legend(legends, loc=0)
    fig3.subplots_adjust(hspace=0)
    fig3.savefig( output + '/prf_{0}.png'.format(hId) )
    fig3.clf()
  print " Time: ", time.time() - start  











if plotMF:
  print "\nPlotting mass function..."
  def powerFit( x, a, n ):
    return a * x**n
  start = time.time()
  output = outputDir + '/massFunction'
  if not os.path.exists( output ): os.makedirs(output)
  for snapshot in snapshots:
    hostId = ms_hostId[snapshot][0]
    hostPos = ms_halosData[snapshot][hostId].attrs['pos']
    hostRadius = ms_halosData[snapshot][ hostId ].attrs['rvir']
    halosIds = ms_haloIds[snapshot]
    halosDist = { hId: np.sqrt( ( (ms_halosData[snapshot][hId].attrs['pos'] - hostPos)**2 ).sum() ) for hId in halosIds }
    halosFilter = [ hId for hId in halosIds 
		  if hId != hostId ]
		  #and halosDist[hId] < hostRadius ]
    nBins = 50
    boxSize = 1
    comulative = True
    addName = ''
    masses = np.array([ ms_halosData[snapshot][hId].attrs['mbound_vir'] for hId in halosFilter ])  
    minMass, maxMass = masses.min(), masses.max()
    binEdges = np.exp(np.linspace(np.log(minMass), np.log(maxMass*(1.01)), nBins))
    binCenters = np.sqrt(binEdges[:-1]*binEdges[1:])
    hist = np.histogram(masses, bins=binEdges)[0]
    if comulative: massFunction = (np.sum(hist) - np.cumsum(hist)) / boxSize**3
    else: massFunction = hist
    #Fit
    fitEdge = 21
    popt1, pcov1 = curve_fit(powerFit, binCenters[:fitEdge], massFunction[:fitEdge])
    popt2, pcov2 = curve_fit(powerFit, binCenters[fitEdge:], massFunction[fitEdge:])
    fig = plt.figure(0)
    fig.clf()
    ax = fig.add_subplot(111)
    ax.plot(binCenters, massFunction)
    ax.plot(binCenters[:fitEdge], powerFit( binCenters[:fitEdge], popt1[0], popt1[1] ) ) 
    ax.plot(binCenters[fitEdge:], powerFit( binCenters[fitEdge:], popt2[0], popt2[1] ) ) 
    #axvline(x=.5, ymin=0.25, ymax=0.75)
    #ax.axvline(x=binCenters[21])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Halo Mass$_{\rm DM}\,[{\rm M_{\odot}}]$')
    if comulative: ax.set_ylabel(r'n(>M) $[{\rm Mpc^{-3}}]$')
    else: ax.set_ylabel(r'n $[{\rm Mpc^{-3}}]$')
    ax.legend(['m_func', 'n:{0:.2f}'.format(float(popt1[1])), 'n:{0:.2f}'.format(float(popt2[1]))], loc=0)
    ax.set_title("massRange[ {0} , {1} ]".format(minMass, maxMass ))
    plt.savefig( output + '/massFunc{0}_{1}.png'.format(addName,snapshot) )
  print " Time: ", time.time() - start


if plotDensityProfile:
  def toFit_NFW(x, rho0):
    global rs
    return rho0/( x/rs*( 1 + x/rs )**2 )
  print '\nGetting density profiles...'
  start = time.time()
  output = outputDir + '/densityProfile'
  if not os.path.exists( output ): os.makedirs(output) 
  DM_particleMass = 92496.7  
  #ms_dProfiles = {}
  ms_dProfiles_dif = {}
  ms_dProfiles_com = {}
  #for snapshot in snapshots:
  for snapshot in [2]:
    #dProfiles = {}
    dProfiles_dif = {}
    dProfiles_com = {}
    particlesPos = ms_partsData[snapshot]['dm']['pos'][...]*1e3
    for hId in ms_halosWithStars[snapshot]['SM']:
      hId = str(hId)
      #dProfiles[hId] = {}
      dProfiles_dif[hId] = {}
      dProfiles_com[hId] = {}
      for finder in  ['pos_rvir',  'bound']:
	haloPos = ms_halosData[snapshot][ hId ].attrs['pos']*1e3
	haloRadius = ms_halosData[snapshot][ hId ].attrs['rvir']*1e3
	inHaloPartIds = ms_halosPartsIds[snapshot][ hId ]['dm_'+finder][...] 
	inHaloPartPos = particlesPos[inHaloPartIds]
	#dProf = mha.getDensityProfile( haloPos, haloRadius, 
				#inHaloPartPos, DM_particleMass, nBins=30)
	#dProfiles[hId][finder] = dProf
	#Extended
	dProf_dif = mha.getDensityProfile_dif(haloPos, haloRadius, 
				inHaloPartPos, DM_particleMass, nBins=30)
	dProfiles_dif[hId][finder] = dProf_dif
	#Comulative
	dProf_com = mha.getDensityProfile_com(haloPos, haloRadius, 
				inHaloPartPos, DM_particleMass, nBins=30)
	dProfiles_com[hId][finder] = dProf_com
    #ms_dProfiles[snapshot] = dProfiles
    ms_dProfiles_dif[snapshot] = dProfiles_dif
    ms_dProfiles_com[snapshot] = dProfiles_com
    
    
  print " Saving density profiles..."
  #for snapshot in snapshots:
  for snapshot in [2]:
    output1 = outputDir + '/densityProfile/snap_{0}'.format(snapshot)
    if not os.path.exists( output1 ): os.makedirs(output1) 
    ##NFW fit
    #output2 = output1 + '/NFW_fit'
    #if not os.path.exists( output2 ): os.makedirs(output2)
    #Extended
    output3 = output1 + '/differential'
    if not os.path.exists( output3): os.makedirs(output3)
    #Comulative
    output4 = output1 + '/comulative'
    if not os.path.exists( output4): os.makedirs(output4)
    hostId = ms_hostId[snapshot][0]
    hostPos = ms_halosData[snapshot][ hostId ].attrs['pos']*1e3
    for hId in ms_halosWithStars[snapshot]['SM']:
      hId = str(hId)
      #fig1 = plt.figure(1)
      #fig1.clf()
      #ax1 = fig1.add_subplot(111)
      ##NFW_fit
      #fig2 = plt.figure(2)
      #fig2.clf()
      #ax2 = fig2.add_subplot(111)
      #Differential
      fig3 = plt.figure(3)
      fig3.clf()
      ax3 = fig3.add_subplot(111)
      #Comulative
      fig4 = plt.figure(4)
      fig4.clf()
      ax4 = fig4.add_subplot(111)
      haloPos = ms_halosData[snapshot][ hId ].attrs['pos']*1e3
      haloMass = ms_halosData[snapshot][ hId ].attrs['mbound_vir']
      posRel = haloPos - hostPos
      dist = np.sqrt( np.sum(posRel**2, axis=-1) )
      haloRadius = ms_halosData[snapshot][ hId ].attrs['rvir']*1e3
      rs = ms_halosData[snapshot][ hId ].attrs['Rs']*1e3
      for finder in  ['pos_rvir',  'bound']:
	##Normal
	#dProf = ms_dProfiles[snapshot][hId][finder] 
	#ax1.loglog(dProf[1], dProf[0] )    
	#NFW fitting
	#if finder == 'rks':	  
	  ##Fit
	  #popt, pcov = curve_fit(toFit_NFW, dProf[1], dProf[0])
	  #rho0 = popt[0]
	  #ax2.loglog(dProf[1], dProf[0] )
	  #ax2.loglog(dProf[1], mha.densityNFW( dProf[1], rho0 ,rs ) )
	#Extended
	dProf_dif = ms_dProfiles_dif[snapshot][hId][finder]
	ax3.loglog(dProf_dif[1], dProf_dif[0] )    
	#Comulative
	dProf_com = ms_dProfiles_com[snapshot][hId][finder]
	ax4.loglog(dProf_com[1], dProf_com[0] ) 
      #ax1.set_xlabel(r'r [kpc]')
      #ax1.set_ylabel(r'Density $[{\rm M_{\odot} kpc^{-3}  }]$')
      #ax1.set_title('id:{2},    mass:{0:1.2e} Msun,    dist:{1:.0f} kpc'.format(haloMass, float(dist), hId) )
      #ax1.legend(['pos_rvir', 'rks', 'bound'], loc=0)
      #fig1.savefig( output1 + '/dPrf_{0}.png'.format(hId) )
      #ax2.set_xlabel(r'r [kpc]')
      #ax2.set_ylabel(r'Density $[{\rm M_{\odot} kpc^{-3}  }]$')
      #ax2.set_title('id:{2},    mass:{0:1.2e} Msun,    dist:{1:.0f} kpc'.format(haloMass, float(dist), hId) )
      #ax2.legend([ 'rks', 'fit_NFW'], loc=0)
      #fig2.savefig( output2 + '/dPrf_{0}.png'.format(hId) )	
      ax3.set_xlabel(r'r [kpc]')
      ax3.set_ylabel(r'Density $[{\rm M_{\odot} kpc^{-3}  }]$')
      ax3.set_title('id:{2},    mass:{0:1.2e} Msun,    dist:{1:.0f} kpc'.format(haloMass, float(dist), hId) )
      ax3.legend(['pos_rvir',  'bound'], loc=0)
      fig3.savefig( output3 + '/dPrf_{0}.png'.format(hId) )
      ax4.set_xlabel(r'r [kpc]')
      ax4.set_ylabel(r'Comulative Density $[{\rm M_{\odot} kpc^{-3}  }]$')
      ax4.set_title('id:{2},    mass:{0:1.2e} Msun,    dist:{1:.0f} kpc'.format(haloMass, float(dist), hId) )
      ax4.legend(['pos_rvir', 'bound'], loc=0)
      ax4.axvline( x=haloRadius, color='y')
      fig4.savefig( output4 + '/dPrf_{0}.png'.format(hId) )      
  print " Time: ", time.time() - start 


if starsProfiles:
  print "\nStellar density profiles..."
  start = time.time()
  for snapshot in snapshots:
    output = outputDir + '/densityProfile/snap_{0}'.format(snapshot)
    output5 = output + '/stars/density'
    if not os.path.exists( output5): os.makedirs(output5)
    hostId = ms_hostId[snapshot][0]
    hostPos = ms_halosData[snapshot][ hostId ].attrs['pos']*1e3
    particlesPos = ms_partsData[snapshot]['star']['pos'][...]*1e3
    particlesVel = ms_partsData[snapshot]['star']['vel'][...]
    particlesMass = ms_partsData[snapshot]['star']['mass'][...]
    for hId in ms_halosWithStars[snapshot]['SM']:
      fig1 = plt.figure(1)
      fig1.clf()
      ax1 = fig1.add_subplot(111)
      hId = str(hId)
      haloPos = ms_halosData[snapshot][ hId ].attrs['pos']*1e3
      haloRadius = ms_halosData[snapshot][ hId ].attrs['rvir']*1e3
      haloMass = ms_halosData[snapshot][ hId ].attrs['mbound_vir']
      dist = np.sqrt( np.sum( (haloPos - hostPos)**2, axis=-1) )
      for finder in  [ 'bound' ]:
	inHaloPartIds = ms_halosPartsIds[snapshot][ hId ]['star_'+finder][...] 
	inHaloPartPos = particlesPos[inHaloPartIds]
	inHaloPartVel = particlesVel[inHaloPartIds]
	inHaloPartMass = particlesMass[inHaloPartIds]
	dProf = mha.getDensityProfile_mass(haloPos, haloRadius, inHaloPartPos, inHaloPartMass, nBins=20 )  
	ax1.loglog( dProf[1], dProf[0] )
      ax1.set_xlabel(r'r [kpc]')
      ax1.set_ylabel(r'Stellar Density $[{\rm M_{\odot} kpc^{-3}  }]$')
      ax1.set_title('id:{2}, DMmass:{0:1.2e},  Smass:{3:1.2e},  dist:{1:.0f} kpc'.format(haloMass, float(dist), hId, inHaloPartMass.sum()) )
      ax1.legend(['bound'], loc=0)
      #ax1.axvline( x=haloRadius, color='y')
      fig1.savefig( output5 + '/dPrf_{0}.png'.format(hId) ) 
  print " Time: ", time.time() - start 


#print "\nStellar vel_distb profiles..."
#start = time.time()
#snapshot = 2
#output = outputDir + 'densityProfile/snap_{0}'.format(snapshot)
#output1 = output + '/stars/vel_distb'
#if not os.path.exists( output1): os.makedirs(output1)
#hostId = ms_hostId[snapshot][0]
#hostPos = ms_halosData[snapshot][ hostId ].attrs['pos']*1e3
#particlesPos = ms_partsData[snapshot]['star']['pos'][...]*1e3
#particlesVel = ms_partsData[snapshot]['star']['vel'][...]
#for hId in ms_halosWithStars[snapshot]['SM']:
  #hId = str( hId )
  #haloPos = ms_halosData[snapshot][ hId ].attrs['pos']*1e3
  #haloVel = ms_halosData[snapshot][ hId ].attrs['vel']
  #haloRadius = ms_halosData[snapshot][ hId ].attrs['rvir']*1e3
  #haloMass = ms_halosData[snapshot][ hId ].attrs['mbound_vir']
  #dist = np.sqrt( np.sum( (haloPos - hostPos)**2, axis=-1) )
  #sMass = ms_halosData[snapshot][ hId ].attrs['SM']
  #finder = 'bound'
  #inHaloPartIds = ms_halosPartsIds[snapshot][ hId ]['star_'+finder][...] 
  #inHaloPartPos = particlesPos[inHaloPartIds]
  #inHaloPartVel = particlesVel[inHaloPartIds]
  ##inHaloPartMass = particlesMass[inHaloPartIds]
  ########################################
  #nBins = 10
  ##prof = mha.getKinematicProfile( haloPos, haloRadius, haloVel, particlesPos, particlesVel, nBins=20)
  
  #posRel = (inHaloPartPos - haloPos)[...]
  #velRel = (inHaloPartVel - haloVel)[...]
  #distToCenter = np.sqrt( np.sum(posRel**2, axis=-1) )
  #rMin, rMax = 0.1, haloRadius/2
  #distMask = np.where( (distToCenter > rMin) * (distToCenter < rMax) )
  #distToCenter = distToCenter[distMask]
  #posRel = posRel[distMask]
  #velRel = velRel[distMask]
  #sortedIndx = distToCenter.argsort()
  #distToCenter = distToCenter[sortedIndx]
  #posRel = posRel[sortedIndx]
  #velRel = velRel[sortedIndx]
  #nParticles = distToCenter.shape[0]
  #pos = posRel
  #vel = velRel
  #posNormalizer = np.tile( distToCenter, (3,1) ).T
  #pos = pos/posNormalizer #Normalize position vector
  #radialVel = np.array([ pos[i].dot(vel[i]) for i in range(nParticles) ])
  #binEdges = np.exp(np.linspace(np.log(1e-1), np.log(rMax*(1.001)), nBins))
  #binCenters = np.sqrt(binEdges[:-1]*binEdges[1:])
  #hist, histBinEdges  = np.histogram(distToCenter, bins=binEdges)
  ##shellVol = 4.*np.pi/3*(binEdges[1:]**3 - binEdges[:-1]**3)
  #velDisp = np.zeros_like(binCenters)
  #start = 0
  #for i in range(velDisp.shape[0]):
    #end = start + hist[i]
    #vels = radialVel[start:end]
    #vAvg = np.average( vels )
    #velDisp[i] = np.sqrt(( (vels - vAvg )**2 ).sum()/hist[i]) 
    #start = end 
#############################################################
  #prof = velDisp, binCenters
  #fig1 = plt.figure(2)
  #fig1.clf()
  #ax1 = fig1.add_subplot(111)
  #ax1.plot( prof[1], prof[0] )
  #ax1.set_xscale('log')
  #ax1.set_xlabel(r'r [kpc]')
  #ax1.set_title('id:{2}, DMmass:{0:1.2e},  Smass:{3:1.2e},  dist:{1:.0f} kpc'.format(haloMass, float(dist), hId, sMass) )
  #fig1.savefig( output1 + '/kPrf_{0}.png'.format(hId) ) 

#sradialVel = posRel.dot()
#rmax = haloRadius
#rMax = distToCenter.max()
#binWidth = rMax/(nBins-1)
#binEdges = np.exp(np.linspace(np.log(1e-1), np.log(rMax*(1.001)), nBins))
#binCenters = np.sqrt(binEdges[:-1]*binEdges[1:])
#hist, histBinEdges  = np.histogram(distToCenter, bins=binEdges)
#shellVol = 4.*np.pi/3*(binEdges[1:]**3 - binEdges[:-1]**3)
#nParticles = distToCenter.shape[0]
#velDisp = np.zeros_like(shellVol)
#start = nParticles - hist.sum()
##for i in range(velDisp.shape[0]):
  ##end = start + hist[i]
  ##vels = particlesVel[start:end]
  ##vAvg = np.average(vels)
  ##velDisp[i] = (particlesMass[start:end]).sum()
  ##start = end


if writeTables:
  print '\nSaving halo-star table...'
  start = time.time()
  output = outputDir + '/tables'
  if not os.path.exists( output ): os.makedirs(output) 
  for snapshot in snapshots:
    hostId = ms_hostId[snapshot][0]
    halosData = ms_halosData[snapshot]
    dmMass, starMass = ms_partsData[snapshot]['dm']['mass'][...], ms_partsData[snapshot]['star']['mass'][...]
    allHalosWithStars = ms_halosWithStars[snapshot]['all']
    starMassInHalos = ms_inHaloStarMass[snapshot]
    nStarInHalos = ms_nStarsInHalos[snapshot]
    inHaloPartsIds = ms_halosPartsIds[snapshot]
    nGas, nDM = ms_nPart[snapshot]['gas'], ms_nPart[snapshot]['dm']
    writeHaloTable( output, snapshot, hostId, halosData, allHalosWithStars, inHaloPartsIds, dmMass, starMass, nGas, starMassInHalos, nStarInHalos )
  print " Time: ", time.time() - start  


if plotDensityProjection:
  print "\nPlotting density projection..."
  print "Loading density..."
  start = time.time()
  output = outputDir + '/densityProjection'
  if not os.path.exists( output ): os.makedirs(output)
  for snapshot in snapshots:
    print " Snapshot: ", snapshot
    halosData = ms_halosData[snapshot]
    halosIds = ms_haloIds[snapshot]
    #halosIds = [ hId for hId in halosData.keys() if hId!='all'
		#and hId==str(halosData[hId].attrs['id']) ]
    #Load Density
    densityName = particlesDir + 'density_{0}.h5'.format(snapshot) 
    try: densityData =  h5.File( densityName, 'r' )
    except IOError : 
      print " Getting Density snapshot: {0} ...".format(snapshot)
      densityData =  h5.File( densityName , 'w' )
      for pType in ["dm", "star"]:
	print "  " + pType
	group = densityData.create_group( pType )
	pos = ms_partsData[snapshot][pType]['pos'][...]
	mass = ms_partsData[snapshot][pType]['mass'][...]
	density, limits = tools.getDensity( pos, mass, nPoints=256 )
	group.create_dataset( 'density', data=density, compression="gzip", compression_opts=9 )
	group.create_dataset( 'limits', data=limits,   compression="gzip", compression_opts=9 )
    for pType in [ 'dm', 'star' ]:
      print "  Plotting density  " + pType
      density = densityData[pType].get("density")[...]
      limits  = densityData[pType].get("limits")[...]
      if pType == 'dm':
	halosFilter = [ hId for hId in halosIds if halosData[hId].attrs['mvir'] > 1e7]
	halosStarMass = None
	haloRadius = 'rvir'
      if pType == 'star': 
	halosFilter = [ hId for hId in halosIds if halosData[hId].attrs['SM'] > 1e5]
	halosStarMass = np.array([ halosData[hId].attrs['SM'] for hId in halosFilter ])
	haloRadius = 'rvir'
      tools.plotDensitySimple( density, limits, halosData, halosFilter, pType, halosStarMass,
			hRadius=haloRadius, output=output, alpha=0.15, 
			title='', addName='{0}_{1}'.format(pType, snapshot))

  print '\nPlotting stars density projection...'
  start = time.time()
  output = outputDir + '/stars'
  if not os.path.exists( output ): os.makedirs(output) 
  for snapshot in snapshots:
    hostId = int( ms_hostId[snapshot][0] )
    starPos = ms_partsData[snapshot]['star']['pos'][...]
    starMass = ms_partsData[snapshot]['star']['mass'][...]
    nGas, nDM = ms_nPart[snapshot]['gas'], ms_nPart[snapshot]['dm']
    halosData = ms_halosData[snapshot]
    for finder in [ 'pos_rvir10', 'pos_rvmax', 'rks', 'bound' ]:
      starsIds = []
      halosIds = []
      for hId in ms_inHaloStarMass[snapshot][finder].keys():
	if hId == hostId: continue
	if ms_inHaloStarMass[snapshot][finder][hId] < 1e5: continue
	halosIds.append( hId )
	starsIds.extend( ms_halosPartsIds[snapshot][str(hId)]['star_'+finder][...].tolist() )
      starsIds = np.array( list( set( starsIds ) ) )
      #if finder == 'rks': starsIds -= ( nGas + nDM )
      pos = starPos[starsIds]
      mass = starMass[starsIds]
      density, limits = tools.getDensity( pos, mass, nPoints=256 )
      halosFilter = [ str( hId ) for hId in halosIds ]
      halosStarMass = np.array([ ms_inHaloStarMass[snapshot][finder][int(hId)] for hId in halosFilter ])
      haloRadius = 'rvir'
      tools.plotDensitySimple( density, limits, halosData, halosFilter, 'star', halosStarMass,
		      hRadius=haloRadius, output=output, alpha=0.15, 
		      title=' halo stars by {0}   snap: {1}'.format(finder, snapshot), addName='{0}_{1}'.format(finder, snapshot))
  print " Time: ", time.time() - start  
