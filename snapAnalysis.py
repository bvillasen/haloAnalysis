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
import plots
import tools
from ms_haloTable import writeHaloTable
from geometry import *
from findCenters import *


paramNumber = 'rg00'

plotDensityProjection = False
plotDensityProfile = False
fileHalos = False
linkHP = False
writeTables = False
plotMF = False
starsProfiles = False
profiles = False
errorBar_prof = True
profiles_2d = False
anim = False
densProj = False
moveCenter = False
moveCM = False
for option in sys.argv:
  if option == "moveCM": moveCM = True
  if option == "move": moveCenter = True
  if option == "proj": densProj = True
  if option == "dproj": plotDensityProjection = True
  if option == "dprof": plotDensityProfile = True
  if option == "prof": profiles = True
  if option == "error": errorBar_prof = True
  if option == "prof2d": profiles_2d = True
  if option == "anim": anim = True
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
#Use RKS/examples/calc_potentials to get bound pairticls ids
getBoundCommand = '/home/bruno/apps/rockstar-galaxies_new/examples/calc_potentials'

listFiles = [outputDir + f for f in os.listdir( outputDir ) if f.find(".list")>=0 ]
boundFiles = [outputDir + f for f in os.listdir( outputDir ) if f.find("boundParticles")>=0 ]
snapshots = sorted( [ int(f[f.find('.list')-1]) for f in listFiles ], reverse=True)
snapshots = [ 3 ]
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

print '\nLoading halos data from rks...'
start = time.time()
paramNumber_rks = 'r00'
inputDir = dataDir + 'halos/{0}/multi_snap/'.format(paramNumber_rks)
centersDir = outputDir + 'moveCenters/'
ms_halosData_rks = {}
ms_rksCenters = {}
findCenter_rks = False
if findCenter_rks:
  for snapshot in snapshots:
    halosFile = inputDir + 'halosData_{0}.h5'.format(snapshot)
    ms_halosData_rks[snapshot] = h5.File( halosFile, 'r' )
    print ' Finding rks centers...'
    snapshot = max(snapshots)
    ms_halos_rksId = {}
    data_rks = ms_halosData_rks[snapshot]
    data     = ms_halosData[snapshot]
    allIds = set([hId for hId in data_rks.keys() if hId != 'all'])
    param_list = [ 'x', 'y', 'z', 'vx', 'vy', 'vz' ]
    ms_halos_rksId[snapshot] = {}
    outData = []
    for hId in ms_halosWithStars[snapshot]['SM']:
      posibleIds = allIds.copy()
      diff = 0.20
      while len( posibleIds ) > 1:
	for param in param_list:
	  val = data[str(hId)].attrs[param]
	  posibleIds_p = tools.getCloseIdsParam( val, data_rks, param, diff, posibleIds)
	  posibleIds = posibleIds.intersection(posibleIds_p)
	diff -= 0.005
      ms_halos_rksId[snapshot][hId] = posibleIds
      if len( posibleIds ) > 0:
	id_rks = posibleIds.pop()
	center_rks = data_rks[id_rks].attrs['pos']*1e3
      else:
	id_rks = -1
	center_rks = ms_halosData[snapshot][str(hId)].attrs['pos']*1e3
      outData.append( [ int( hId ), int( id_rks ), center_rks[0], center_rks[1], center_rks[2] ])
    outData = np.array(outData)
    output = outputDir + 'moveCenters/'
    if not os.path.exists( output ): os.makedirs( output )
    np.savetxt( output + 'rks_centers_{0}.dat'.format(snapshot), outData )
for snapshot in snapshots:
  ms_rksCenters[snapshot] = {}
  halosFile = inputDir + 'halosData_{0}.h5'.format(snapshot)
  ms_halosData_rks[snapshot] = h5.File( halosFile, 'r' )
  centersData = np.loadtxt( centersDir + 'rks_centers_{0}.dat'.format(snapshot)  )
  for center in centersData:
    hId = str( int(center[0]) )
    ms_rksCenters[snapshot][hId] = {}
    ms_rksCenters[snapshot][hId]['id_rks'] = int(center[1])
    ms_rksCenters[snapshot][hId]['pos'] = center[2:]
print " Time: ", time.time() - start 



findCenters = False
if findCenters:
  print '\nGetting halo-galaxy centers...'
  start = time.time()
  output = outputDir + 'moveCenters/'
  if not os.path.exists( output ): os.makedirs( output )
  snapshot = max(snapshots)
  hostId = ms_hostId[snapshot][0]
  centers = {}
  for pType in [ 'star', 'dm' ]:
      centers[pType] = {}
      partsPos  = ms_partsData[snapshot][pType]['pos'][...]*1e3
      partsMass = ms_partsData[snapshot][pType]['mass'][...]
      for hId in ms_halosWithStars[snapshot]['SM']:
	hId = str( hId )
	if hId == hostId: continue
	print 'Halo : {0}'.format( hId )
	haloPos   = ms_halosData[snapshot][hId].attrs['pos']*1e3
	inHaloPartIds  = ms_halosPartsIds[snapshot][hId][pType+'_bound'][...]
	inHaloPartPos  = partsPos[inHaloPartIds] 
	inHaloPartMass = partsMass[inHaloPartIds]
	center = getCenter( haloPos, inHaloPartPos, inHaloPartMass, rLast_cm=0.2, nPartLast_cm=40 )
	centers[pType][hId] = center
  print ' Saving centers...'
  centersData = []
  for hId in centers[pType].keys():
    line = [ int(hId), centers['dm'][hId][0], centers['dm'][hId][1], centers['dm'][hId][2], centers['star'][hId][0], centers['star'][hId][1], centers['star'][hId][2] ]
    centersData.append( line )  
  centersData = np.array( centersData )
  h = '#hId, c_dm_x, c_dm_y, c_dm_z,  c_st_x, c_st_y, c_st_z   '
  np.savetxt(output + 'centers_{0}.dat'.format(snapshot), centersData, header=h )     
  print ' Centers saved: ', output + 'centers_{0}.dat'.format(snapshot)
  print " Time: ", time.time() - start  
  


print '\nLoading centers...'
start = time.time() 
inputDir = outputDir + 'moveCenters/'
ms_centers = {}
ms_offsets = {}
for snap in [ 0, 1, 2, 3]:
  ms_centers[snap] = { 'dm':{}, 'star':{} }
  ms_offsets[snap] = {}
  centersData = np.loadtxt( inputDir + 'centers_{0}.dat'.format(snapshot) )
  for line in centersData:
    hId = str( int( line[0] ) )
    center_dm = line[1:4]
    center_st = line[4:7]
    ms_centers[snap]['dm'][hId] = center_dm
    ms_centers[snap]['star'][hId] = center_st
    ms_offsets[snap][hId] = np.sqrt( ( ( center_dm - center_st )**2 ).sum() )
print " Time: ", time.time() - start  

def printOffsets():
  for hId in ms_offsets[snap].keys():
    line = '{0}  {1:.1f}  {2:.1f}  {3:.1f}  {4:.1f}'.format(hId, ms_offsets[3][hId]*1e3, ms_offsets[2][hId]*1e3, ms_offsets[1][hId]*1e3, ms_offsets[0][hId]*1e3 )          
    print line


def printCenters( CM_r, snapshot=3 ):
  for hId in ms_halosWithStars[snapshot]['SM']:
    hId = str( hId )
    rksFound = True
    haloPos = ms_halosData[snapshot][hId].attrs['pos']*1e3
    cm_s    = ms_centers[snapshot][CM_r]['star'][hId]['CM']
    cm_dm   = ms_centers[snapshot][CM_r]['dm'][hId]['CM']
    if ms_rksCenters[snapshot][hId]['id_rks']>0: c_rks   = ms_rksCenters[snapshot][hId]['pos'] 
    else:
      rksFound = False
      c_rks = ' RKS-halo not found'
    dist_DM_S = np.sqrt( ( (cm_s - cm_dm )**2 ).sum() )*1e3
    print '\nHalo: {0}\n rks: {1}\n cm_d:{2}\n cm_s:{3}\n pos: {4}'.format(hId, c_rks, cm_dm, cm_s, haloPos)
    if rksFound:
      dist_rksDM_cmDM = np.sqrt( ( (c_rks - cm_dm )**2 ).sum() )*1e3
      print 'dist rks_DM-cm_DM: {0:.0f} pc'.format( dist_rksDM_cmDM )
    print 'dist cm_DM-S: {0:.0f} pc'.format( dist_DM_S )
############################################################################### 







if profiles:
  useRKScenters = False
  CM_r = 130
  binByPart = True
  nBins = 15
  profDim = 2 if profiles_2d else 3
  print "\nDensity and vel_distb profiles..."
  startT = time.time()
  snapshot = max(snapshots)
  output = outputDir + 'profiles_{0}'.format( CM_r)
  if useRKScenters: output += '_rks'
  if binByPart: output += '_equiP'
  output += '/snap_{0}'.format(snapshot)
  if not os.path.exists( output ): os.makedirs( output )
  output1 = output + '/kinematic'
  if not os.path.exists( output1 ): os.makedirs( output1 )
  output2 = output + '/vel_distb'
  if not os.path.exists( output2 ): os.makedirs( output2 )
  hostId  = ms_hostId[snapshot][0]
  hostPos = ms_halosData[snapshot][ hostId ].attrs['pos']*1e3
  for hId in ms_halosWithStars[snapshot]['SM']:
    hId = str( hId )
    if hId == hostId: continue
    print " Halo: {0}".format(hId)
    #kinematic
    fig1, ax1 = plt.subplots( 2, 2, sharex=True )
    fig1.set_size_inches(16,10)
    ##vel_distb
    #fig2 = plt.figure(2)
    #ax2 = fig2.add_subplot(111)
    #Dens and vel dist
    fig3, ax3 = plt.subplots( 2,2)
    fig3.set_size_inches(16,10)
    legends1, legends2, legends3 = [], [], []
    minBinCenter, maxBinCenter = 1000, 0
    sMass      = ms_halosData[snapshot][ hId ].attrs['SM']
    if sMass < 1e6: continue
    haloPos    = ms_halosData[snapshot][ hId ].attrs['pos']*1e3
    haloVel    = ms_halosData[snapshot][ hId ].attrs['vel']
    haloRadius = ms_halosData[snapshot][ hId ].attrs['rvir']*1e3
    haloRvmax = ms_halosData[snapshot][ hId ].attrs['rvmax']*1e3
    haloVmax = ms_halosData[snapshot][ hId ].attrs['vmax']
    haloMass   = ms_halosData[snapshot][ hId ].attrs['mbound_vir']
    dist       = np.sqrt( np.sum( (haloPos - hostPos)**2, axis=-1) )
    haloBulkVel = [ ms_halosData[snapshot][ hId ].attrs['bulk_vx'],
		    ms_halosData[snapshot][ hId ].attrs['bulk_vy'],
		    ms_halosData[snapshot][ hId ].attrs['bulk_vz'] ]
    haloBulkVel = np.array( haloBulkVel )
    beta, nParticles, centers = {}, {}, {}
    for pType in [  'dm', 'star']:
      #new_center = haloPos
      new_center = ms_centers[snapshot][CM_r][pType][hId]['CM']
      # RKS CENTERS FOR DM
      if pType == 'dm' and useRKScenters:
	if ms_rksCenters[snapshot][hId]['id_rks']>0:
	  new_center = ms_rksCenters[snapshot][hId]['pos']
	else: print '  No RKS-halo found: using CM center '
      centers[pType] = new_center
      particlesPos  = ms_partsData[snapshot][pType]['pos'][...]*1e3
      particlesVel  = ms_partsData[snapshot][pType]['vel'][...]
      particlesMass = ms_partsData[snapshot][pType]['mass'][...]
      if profiles_2d:
	particlesPos = particlesPos[:,:-1]
	particlesVel = particlesPos[:,:-1]
      finder_list = [ 'bound' ]
      for finder in finder_list:
      #finder = 'bound'
	inHaloPartIds  = ms_halosPartsIds[snapshot][ hId ][pType+ '_' + finder][...] 
	nParticles[pType] = len( inHaloPartIds )
	inHaloPartPos  = particlesPos[inHaloPartIds]
	inHaloPartVel  = particlesVel[inHaloPartIds]
	inHaloPartMass = particlesMass[inHaloPartIds]
	posRel       = inHaloPartPos - new_center
	velRel       = inHaloPartVel - haloVel
	distToCenter = np.sqrt( np.sum(posRel**2, axis=-1) )
	sortedIndx   = distToCenter.argsort()
	distToCenter   = distToCenter[sortedIndx]
	inHaloPartMass = inHaloPartMass[sortedIndx]
	posRel         = posRel[sortedIndx]
	velRel         = velRel[sortedIndx]
	posRel       = posRel/(distToCenter[:,None]) #Normalize position vector
	r_vec, theta_vec, phi_vec = spherical_vectors(posRel)
	vel_radial_val =np.array([ posRel[i].dot(velRel[i]) for i in range(len(distToCenter)) ]) 
	#vel_radial = vel_radial_val[:,None]*posRel
	#vel_ortho  = velRel - vel_radial	
	vel_theta_val  = np.array([ theta_vec[i].dot(velRel[i]) for i in range(len(distToCenter)) ])
	vel_phi_val    = np.array([ phi_vec[i].dot(velRel[i])   for i in range(len(distToCenter)) ])
	sigma_radial2 = ( ( vel_radial_val - vel_radial_val.mean() )**2 ).mean() 
	sigma_phi2    = ( ( vel_phi_val - vel_phi_val.mean() )**2 ).mean() 
	beta[pType] = 1 - ( sigma_phi2 / sigma_radial2 )
	#beta[pType] = 1 - ( (vel_phi_val**2).mean() / (vel_radial_val**2).mean() )
	rMin, rMax = 0.1, distToCenter.max()
	################################################
	binCenters, densProf, velProf_rad, velProf_theta, velProf_phi, betaProf, nPartDist = mha.get_profiles( rMin, rMax, 
			distToCenter, inHaloPartMass, vel_radial_val, vel_theta_val, vel_phi_val,
			nBins=nBins, binByPart=binByPart )
	###############################################
	#plotting##########################################
	minBinCenter = min( binCenters[0], minBinCenter )
	maxBinCenter = max( binCenters[-1], maxBinCenter )
	#fig1##########################################
	if pType == 'dm':
	  ax1[0,0].set_xscale('log')
	  ax1[0,1].set_xscale('log')
	  ax1[0,0].errorbar( binCenters, velProf_rad,  yerr=velProf_rad/np.sqrt(nPartDist) )
	  ax1[0,0].errorbar( binCenters, velProf_phi,  yerr=velProf_phi/np.sqrt(nPartDist) )
	  ax1[0,0].legend(['dm_radial', 'dm_tang'], loc=0)
	  ax1[0,1].errorbar( binCenters, betaProf,     yerr=betaProf/np.sqrt(nPartDist) )
	  
	if pType == 'star':
	  ax1[1,0].set_xscale('log')
	  ax1[1,1].set_xscale('log')
	  ax1[1,0].errorbar( binCenters, velProf_rad,  yerr=velProf_rad/np.sqrt(nPartDist)  )
	  ax1[1,0].errorbar( binCenters, velProf_phi,  yerr=velProf_phi/np.sqrt(nPartDist) )
	  ax1[1,0].legend(['st_radial', 'st_tang'], loc=0)	
	  ax1[1,1].errorbar( binCenters, betaProf,     yerr=betaProf/np.sqrt(nPartDist) )
	###############################################
	if (finder=='bound'): 
	  #if pType == 'star':
	    ##ax2.plot( binCenters, velProf_rad, '-o' )
	    ##ax2.plot( binCenters, velProf_radAbs, '-o' )
	    ##ax2.plot( binCenters, velProf_ort, '-o' )
	    #legends2.append( pType + '_' + finder+'_rad')
	    #legends2.append( pType + '_' + finder+'_radAbs')
	    #legends2.append( pType + '_' + finder+'_ort')
	  ax3[1,0].set_xscale('log')
	  ax3[0,0].loglog(   binCenters, densProf, '-o' )
	  ax3[1,0].errorbar( binCenters, velProf_rad, yerr=velProf_rad/np.sqrt(nPartDist) )
	  #ax3[1,1].set_xscale('log')
	  ax3[0,1].set_yscale('log')
	  ax3[0,1].plot(  nPartDist, '-o'  )
	  ax3[1,1].plot(  binCenters, '-o' )
	  legends3.append(pType + '_' + finder)
    CM_dist = np.sqrt( ( ( centers['dm'] - centers['star'] )**2 ).sum() )*1000
    ax1[0,0].axvline( x=haloRadius, color='y')
    ax1[0,0].axvline( x=haloRvmax, color='m')
    ax1[1,0].axvline( x=haloRadius, color='y')
    ax1[1,0].axvline( x=haloRvmax, color='m')
    ax1[0,1].axvline( x=haloRadius, color='y')
    ax1[0,1].axvline( x=haloRvmax, color='m')
    ax1[1,1].axvline( x=haloRadius, color='y')
    ax1[1,1].axvline( x=haloRvmax, color='m')
    ax1[0,0].axhline( y=haloVmax, color='c')
    ax1[1,0].axhline( y=haloVmax, color='c')
    ax1[0,1].axhline( y=0, color='g')
    ax1[1,1].axhline( y=0, color='g')
    ax1[0,0].set_xlim(minBinCenter, maxBinCenter)
    ax1[0,1].set_ylim(-1, 1)
    ax1[1,1].set_ylim(-1, 1)
    #ax1[0].set_xlabel(r'r [kpc]')
    ax1[1,0].set_xlabel(r'r [kpc]')
    ax1[1,1].set_xlabel(r'r [kpc]')
    ax1[0,0].set_ylabel(r'DM Velocity Dispersion $[{\rm  km/seg  }]$')
    ax1[1,0].set_ylabel(r'Stars Velocity Dispersion $[{\rm  km/seg  }]$')
    ax1[0,1].set_ylabel(r'$\beta_{\rm DM}$', rotation="horizontal")
    ax1[1,1].set_ylabel(r'$\beta_{\rm S}$' , rotation="horizontal")
    ax1[0,0].set_title('id:{2}, mDM:{0:1.1e},  mS:{3:1.1e},  d:{1:.0f} kpc,  nS:{4:1.1e}'.format(haloMass, float(dist), hId, sMass, nParticles['star'] ) )
    ax1[0,1].set_title('Bdm: {0:.2f}  Bs:{1:.2f}'.format( float(beta['dm']), float(beta['star']) ) )
    
    #legends1 = [ 'radial', 'theta' ]
    #ax1.legend(legends1, loc=0)
    fig1.subplots_adjust(hspace=0)
    fig1.savefig( output1 + '/kPrf_{0}.png'.format(hId) ) 
    fig1.clf()
    #ax2.axvline( x=haloRadius, color='y')
    #ax2.set_xlim(minBinCenter, maxBinCenter)
    #ax2.set_xscale('log')
    #ax2.set_xlabel(r'r [kpc]')
    #ax2.set_ylabel(r'Radial Velocity Dispersion $[{\rm  km/seg  }]$')
    #ax2.set_title('id:{2}, mDM:{0:1.1e},  mS:{3:1.1e},  dist:{1:.0f} kpc'.format(haloMass, float(dist), hId, sMass) )
    #ax2.legend(legends2, loc=0)
    #fig2.savefig( output2 + '/kPrf_{0}.png'.format(hId) )
    #fig2.clf()
    ax3[0,0].axvline( x=haloRadius, color='y')
    ax3[0,0].axvline( x=haloRvmax, color='m')
    ax3[0,0].set_ylabel(r'Density $[{\rm M_{\odot} kpc^{-3}  }]$')
    ax3[0,0].set_title('id:{2}, mDM:{0:1.1e},  mS:{3:1.1e},  d:{1:.0f} kpc\nnS:{4:1.1e} CM_dist:{5:.0f}'.format(haloMass, float(dist), hId, sMass, nParticles['star'], CM_dist  ) )
    ax3[0,0].legend(legends3, loc=0) 
    ax3[1,0].axvline( x=haloRadius, color='y')
    ax3[1,0].axvline( x=haloRvmax, color='m')
    ax3[1,0].set_xlim(rMin - 1e-2, maxBinCenter)
    ax3[0,0].set_xlim(rMin - 1e-2, maxBinCenter)
    ax3[1,1].set_ylim(0 , haloRadius/3)
    ax3[1,0].set_ylabel(r'Radial Velocity Dispersion $[{\rm  km/seg  }]$')
    ax3[1,0].set_xlabel(r'r [kpc]')
    ax3[1,1].set_xlabel(r'Bin number')
    ax3[1,1].set_ylabel(r'Bin center [kpc]')
    ax3[0,1].set_ylabel(r'num particles in bin ')
    ax3[1,1].axhline( y=haloRadius, color='y')
    ax3[1,1].axhline( y=haloRvmax, color='m')
    #ax3[1].legend(legends, loc=0)
    fig3.subplots_adjust(hspace=0)
    fig3.savefig( output + '/prf_{0}.png'.format(hId) )
    fig3.clf()
  print " Time: ", time.time() - startT  


#snapshot = max(snapshots)
#hostId  = ms_hostId[snapshot][0]
##hId = hostId
#hId = '737'  
#haloPos    = ms_halosData[snapshot][ hId ].attrs['pos']*1e3
#haloVel    = ms_halosData[snapshot][ hId ].attrs['vel']
#beta = {}
#pType = 'dm'
#particlesPos  = ms_partsData[snapshot][pType]['pos'][...]*1e3
#particlesVel  = ms_partsData[snapshot][pType]['vel'][...]
#particlesMass = ms_partsData[snapshot][pType]['mass'][...]
#finder = 'bound'
#inHaloPartIds  = ms_halosPartsIds[snapshot][ hId ][pType+'_'+finder][...] 
#inHaloPartPos  = particlesPos[inHaloPartIds]
#inHaloPartVel  = particlesVel[inHaloPartIds]
#inHaloPartMass = particlesMass[inHaloPartIds]
#new_center = ms_centers[snapshot][pType][hId]['CM']
#new_center     = haloPos
#posRel       = inHaloPartPos - new_center
#velRel       = inHaloPartVel - haloVel
#distToCenter = np.sqrt( np.sum(posRel**2, axis=-1) )
#sortedIndx   = distToCenter.argsort()
#distToCenter   = distToCenter[sortedIndx]
#inHaloPartMass = inHaloPartMass[sortedIndx]
#posRel         = posRel[sortedIndx]
#velRel         = velRel[sortedIndx]
##posRel         = posRel/(distToCenter[:,None]) #Normalize position vector
#radial_vec, theta_vec, phi_vec = spherical_vectors(posRel)
#vel_radial_val = np.array([ radial_vec[i].dot(velRel[i]) for i in range(len(distToCenter)) ]) 
#vel_theta_val  = np.array([  theta_vec[i].dot(velRel[i]) for i in range(len(distToCenter)) ])
#vel_phi_val    = np.array([    phi_vec[i].dot(velRel[i]) for i in range(len(distToCenter)) ])
#beta[pType] = 1 - (vel_phi_val**2).mean() / (vel_radial_val**2).mean()
#rMin, rMax = 0.1, distToCenter.max()
#binCenters, densProf, velProf_rad, velProf_theta, velProf_phi, nPartDist = mha.get_profiles( rMin, rMax, 
			#distToCenter, inHaloPartMass, vel_radial_val, vel_theta_val, vel_phi_val,
			#nBins=300, binByPart=True )
##fitting##########################################
#def hernquist( r, rho0, r0, a, b, c ):
  #return rho0 * ( r0/r )**a * ( 1 + ( r/r0 )**b )**( (a-c)/b )

#def NFW( r, rho0, r0 ):
  #return rho0 / ( ( r/r0 ) * ( 1 + ( r/r0 )**2 ) )

#fit_values_1, fit_cov_1 = curve_fit( hernquist, binCenters[1:], densProf[1:] )
#rho0, r0, a, b, c = fit_values_1
#densFit_1 = hernquist( binCenters[1:], rho0, r0, a, b, c )

#fit_values_2, fit_cov_2 = curve_fit( NFW, binCenters[1:], densProf[1:] )
#rho0, r0 = fit_values_2
#densFit_2 = NFW( binCenters[1:], rho0, r0 )

#fig = plt.figure(1)
#fig.clf()
#plt.loglog( binCenters, densProf, '-o' )
#plt.loglog( binCenters[1:], densFit_1, '--' )
#plt.loglog( binCenters[1:], densFit_2, '--' )
#fig.show()


if moveCenter:
  def getBallFromMassRatio( ballCenter, radiusInitial, massRatio, partMass_inHalo, tree):
    ball_pos    = ballCenter
    ball_radius = radiusInitial
    mass_total      = partMass_inHalo.sum()
    massRatio_current = 0 
    while massRatio_current < massRatio:
      inBall_ids = tree.query_ball_point( ball_pos, ball_radius )
      inBall_mass = partMass_inHalo[inBall_ids].sum()
      massRatio_current = inBall_mass/mass_total
      #print massRatio
      ball_radius += 0.05 
    return ball_radius, massRatio_current  
  
  def getMassRatio(ball_pos, ball_radius, partMass_inHalo, tree):
    mass_total      = partMass_inHalo.sum()
    inBall_ids = tree.query_ball_point( ball_pos, ball_radius )
    inBall_mass = partMass_inHalo[inBall_ids].sum()
    massRatio = inBall_mass/mass_total
    return massRatio
    
  def getCM( ball_pos, ball_radius, partMass_inHalo, partPos_inHalo, tree  ):
    inBall_ids = tree.query_ball_point( ball_pos, ball_radius )
    p_mass = partMass_inHalo[inBall_ids]
    p_pos  = partPos_inHalo[inBall_ids]
    cm = np.sum( (p_mass[:,None]*p_pos)/(p_mass.sum()), axis=0 )
    return cm  
    
  def evaluateCost( ball_pos, ball_radius, partMass_inHalo, partPos_inHalo, tree  ):
    inBall_ids = tree.query_ball_point( ball_pos, ball_radius )
    p_mass = partMass_inHalo[inBall_ids]
    p_pos  = partPos_inHalo[inBall_ids]
    dist   = np.sqrt( ( (p_pos - ball_pos)**2 ).sum(axis=1) )
    cost = ( p_mass * ( ball_radius - dist )/ball_radius ).sum()
    return cost

  
  def getDeltaPos(ball_pos, ball_radius, partMass_inHalo, partPos_inHalo, tree ):
    moveVec = np.zeros( 3 )
    cost_0 = evaluateCost( ball_pos, ball_radius, partMass_inHalo, partPos_inHalo, tree  )
    #print cost_0
    moveDist = 0.03
    moveDirections = np.array([ [ 1., 0, 0 ], [ 0, 1., 0 ], [ 0, 0, 1. ] ]) 
    for direction in moveDirections:
      new_pos = ball_pos + moveDist*direction
      cost_r = evaluateCost( new_pos, ball_radius, partMass_inHalo, partPos_inHalo, tree  )
      new_pos = ball_pos - moveDist*direction
      cost_l = evaluateCost( new_pos, ball_radius, partMass_inHalo, partPos_inHalo, tree  )
      #moveVec += ( (cost_r-cost_0)/(2*cost_0) + ( cost_0-cost_l )/(2*cost_0) ) * moveDist * direction 
      moveVec +=  (cost_r-cost_l )/(2*cost_0)  * 40*moveDist * direction 
    return moveVec

  def moveBall( ball_posInit, ball_radius, partMass_inHalo, partPos_inHalo, tree ):
    ball_pos = ball_posInit.copy()
    maxIter = 100000
    for i in range(maxIter):
      #print i
      deltaPos_ball = getDeltaPos(ball_pos, ball_radius, partMass_inHalo, partPos_inHalo, tree )
      #print deltaPos_ball
      change = np.sqrt( ( deltaPos_ball**2  ).sum() ) 
      if change < 1e-5: break
      ball_pos += deltaPos_ball
    if i == maxIter-1: print "Ball move not converged"
    print "   steps: {0}".format(i)
    return ball_pos  

  for pType in [  'dm']:
    partsPos  = ms_partsData[snapshot][pType]['pos'][...]*1e3
    partsMass = ms_partsData[snapshot][pType]['mass'][...]
    moveData = []
    for hId in ms_halosWithStars[snapshot]['SM']:
      hId = str( hId )
      if hId == hostId: continue
      #if hId == '1999': continue
      print 'Halo : {0}'.format( hId )
      haloPos   = ms_halosData[snapshot][hId].attrs['pos']*1e3
      haloRvir  = ms_halosData[snapshot][hId].attrs['rvir']*1e3
      haloRvmax = ms_halosData[snapshot][hId].attrs['rvmax']*1e3
      rMin = 0.1 
      inHaloPartIds   = ms_halosPartsIds[snapshot][hId][pType+'_bound'][...]
      partPos_inHalo  = partsPos[inHaloPartIds] 
      partMass_inHalo = partsMass[inHaloPartIds]
      #mass_total      = partMass_inHalo.sum()
      tree = KDTree( partPos_inHalo )
      print "   Initial sphere:"
      ball_pos = haloPos
      massRatio = 0.3 if pType == 'star' else 0.2
      ball_radius, massRatio_i = getBallFromMassRatio( haloPos, rMin, massRatio, partMass_inHalo, tree )
      CM_i = getCM( ball_pos, ball_radius, partMass_inHalo, partPos_inHalo, tree )
      print '    ball_radius: {0}, rvmax: {1}, rvir: {2}'.format( ball_radius, haloRvmax, haloRvir )
      #cost = evaluateCost(  ball_pos, ball_radius, partMass_inHalo, partPos_inHalo, tree )
      print '   Moving sphere'
      ball_pos_new = moveBall( haloPos, ball_radius, partMass_inHalo, partPos_inHalo, tree ) 
      CM_f = getCM( ball_pos_new, ball_radius, partMass_inHalo, partPos_inHalo, tree )
      massRatio_f = getMassRatio(ball_pos_new, ball_radius, partMass_inHalo, tree)
      print '    massRatio_i: {0}'.format(massRatio_i)
      print '    massRatio_f: {0}'.format(massRatio_f)
      print '    pos_i: {0} \n    pos_f: {1}'.format( haloPos, ball_pos_new )
      print '    cm_i:  {0} \n    cm_f:  {1}'.format( CM_i, CM_f )
      
      moveData.append([ int(hId), ball_radius, ball_pos_new[0],  ball_pos_new[1], ball_pos_new[2], CM_i[0], CM_i[1], CM_i[2], CM_f[0], CM_f[1], CM_f[2], massRatio_i, massRatio_f])
    moveData = np.array( moveData )
    h = '#hId, ball_radius, ball_pos_new, CM_ball_initial, CM_ball_final, massRatio_i, massRatio_f'
    np.savetxt(output + 'centers_{0}_{1}.dat'.format(snapshot, pType), moveData, header=h )
  print " Time: ", time.time() - start

############################################################################## 
if densProj:
  partsPos_dm  = ms_partsData[snapshot]['dm']['pos'][...]*1e3
  partsMass_dm = ms_partsData[snapshot]['dm']['mass'][...]
  partsPos_star  = ms_partsData[snapshot]['star']['pos'][...]*1e3
  partsMass_star = ms_partsData[snapshot]['dm']['mass'][...]
  #density, limits, delta, cellOccup = tools.starsDensity( hostPos, snapshot, hostId, 
			  #partsPos_star, partsMass_star, partsPos_dm, partsMass_dm, 
			  #ms_halosData, ms_halosPartsIds, output, nPoints=512, plot=True, view=False, alpha=0.2)
  for hId in ms_halosWithStars[snapshot]['SM']:
    hId = str( hId )
    if hId == hostId: continue
    sMass      = ms_halosData[snapshot][ hId ].attrs['SM']
    if sMass < 1e6: continue
    haloPos = ms_halosData[snapshot][hId].attrs['pos']*1e3
    density, limits, delta, cellOccup = tools.starsDensity( hostPos, snapshot, hId, 
			  partsPos_star, partsMass_star, partsPos_dm, partsMass_dm, 
			  ms_halosData, ms_halosPartsIds, 
			  ms_centers[snapshot], output, nPoints=256, plot=True, view=False, alpha=0.2)
##############################################################################
def wiewDensity(hId):
  hId = str( hId )
  sMass      = ms_halosData[snapshot][ hId ].attrs['SM']
  haloPos = ms_halosData[snapshot][hId].attrs['pos']*1e3
  density, limits, delta, cellOccup = tools.starsDensity( hostPos, snapshot, hId, 
			pos_stars, mass_stars, pos_dm, mass_dm, 
			ms_halosData, ms_halosPartsIds,
			ms_centers[snapshot], output, nPoints=256, plot=True, view=True, alpha=0.2)
################################################################################


##############################################################################
import visual as vi
def initWindow():
  #snapshot = 2
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
  return scene

def loadAnimData( colors, show=False ):
  anim_satelites = {}
  for snapshot in snapshots:
    anim_satelites[snapshot] = {}
    pPos = ms_partsData[snapshot]['star']['pos'][...]
    hostId  = ms_hostId[snapshot][0]
    hostPos = ms_halosData[snapshot][hostId].attrs['pos']
    hostRadius = ms_halosData[snapshot][hostId].attrs['rvir']
    anim_satelites[snapshot][hostId] = {}
    anim_satelites[snapshot][hostId]['sphere'] = vi.sphere(pos=hostPos, radius=hostRadius, opacity=0.2, visible=show )
    halosFilter = [ ( str(hId),  ms_halosData[snapshot][ str(hId) ].attrs['SM'] )
		  for hId in ms_halosWithStars[snapshot]['SM'] 
		  if ms_halosData[snapshot][ str(hId) ].attrs['SM'] >= 1e6
		  and str(hId) != hostId ]
    sortedHalos = sorted( halosFilter, key= lambda x: x[1]  )
    for i, (hId, sm ) in enumerate(sortedHalos):
      anim_satelites[snapshot][hId] = {}
      inHaloPartIds  = ms_halosPartsIds[snapshot][ hId ]['star_bound'][...] 
      inHaloPartPos  = pPos[inHaloPartIds]
      anim_satelites[snapshot][hId]['points'] = vi.points(pos=inHaloPartPos, size=2, color=colors[i], visible=show)
  return anim_satelites

def viewSnapshot(  snapshot, anim_satelites, scene, visible=True   ):
  if visible: print ' snapshot: {0}'.format(snapshot)
  for hId in anim_satelites[snapshot].keys():
    for obj in anim_satelites[snapshot][hId].keys():
      anim_satelites[snapshot][hId][obj].visible = visible

def vis3d():
  #3D visualization
  print '\nStars 3D'
  colors = [ ( 0.3,0.3,1 ), ( 1,1,0 ), ( 1,0,1 ), ( 0,1,1 ), 
	     ( 1,0,0 ), ( 0,1,0 ), ( 0,0,1 ), ( 0.5,1,1 ), 
	     ( 1,0.5,1 ), ( 1,1,0.5 ), ( 1,0.5,0 ), ( 0.8,1,0 ), ( 1,0,0.5 ),    ]
  scene = initWindow()
  anim_satelites = loadAnimData( colors, show=False )
  anim_snap = snapshot
  viewSnapshot( anim_snap, anim_satelites, scene, visible=True)
  
  while True:
    key = scene.kb.getkey()
    if key == 'right':
      viewSnapshot( anim_snap, anim_satelites, scene, visible=False)
      anim_snap = anim_snap+1 if anim_snap < max( snapshots ) else min( snapshots )
      viewSnapshot( anim_snap, anim_satelites, scene, visible=True)
    if key == 'left':
      viewSnapshot( anim_snap, anim_satelites, scene, visible=False)
      anim_snap = anim_snap-1 if anim_snap > min( snapshots ) else max( snapshots )
      viewSnapshot( anim_snap, anim_satelites, scene, visible=True)
if anim: vis3d()
##############################################################################

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


if writeTables:
  print '\nSaving halo-star tables...'
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
	density, limits, delta, cellOccup = tools.getDensity( pos, mass, nPoints=256 )
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
      density, limits, delta, cellOccup = tools.getDensity( pos, mass, nPoints=256 )
      halosFilter = [ str( hId ) for hId in halosIds ]
      halosStarMass = np.array([ ms_inHaloStarMass[snapshot][finder][int(hId)] for hId in halosFilter ])
      haloRadius = 'rvir'
      tools.plotDensitySimple( density, limits, halosData, halosFilter, 'star', halosStarMass,
		      hRadius=haloRadius, output=output, alpha=0.15, 
		      title=' halo stars by {0}   snap: {1}'.format(finder, snapshot), addName='{0}_{1}'.format(finder, snapshot))
  print " Time: ", time.time() - start  
