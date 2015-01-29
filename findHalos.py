import os, sys, tempfile, time
from subprocess import call
import numpy as np
import h5py as h5
currentDirectory = os.getcwd()
#Add Modules from other directories
parentDirectory = currentDirectory[:currentDirectory.rfind("/")]
toolsDirectory = currentDirectory + "/tools"
sys.path.append( toolsDirectory )
from mpi4py import MPI

print "\nHalo Finder multi-snapshot" 
MPIcomm = MPI.COMM_WORLD
pId = MPIcomm.Get_rank()
nProc = MPIcomm.Get_size()

paramNumber = 's00'
h = 0.7

dataDir = "/home/bruno/Desktop/data/galaxy/"
rockstarDir = '/home/bruno/apps/rockstar-galaxies/'
rockstarComand = rockstarDir + 'rockstar-galaxies'
rockstarOutput = dataDir + 'halos/{0}/multi_snap'.format( paramNumber )

DM_particleMass = 92496.7        #Msun

rockstarConf = {
'FILE_FORMAT': "TIPSY", 
'h0': 0.7,                           #<hubble constant today> # in units of 100 km/s/Mpc
'Ol': 0.73,                          #<Omega_Lambda> # in units of the critical density
'Om': 0.27,                          #<Omega_Matter> # in units of the critical density
#'BOX_SIZE': 1.7*h,                    #Mpc/h
'BOX_SIZE': 30*h,
'TOTAL_PARTICLES': 28e6,
'PARTICLE_MASS': DM_particleMass*h,             #Msun/h
'TIPSY_LENGTH_CONVERSION': 28.7*h,   #Typically, box size in Mpc/h
'TIPSY_VELOCITY_CONVERSION': 691,      #Conversion from TIPSY units to km/s at z=0
'TIPSY_MASS_CONVERSION': 3.17e15*h, #Msun/h.
'FORCE_RES': 1e-4*h,                 #Mpc/h
#'SUPPRESS_GALAXIES': 0,
'FULL_PARTICLE_CHUNKS': 1,
'OUTBASE': rockstarOutput
}
parallelConf = {
'PARALLEL_IO': 1,
'PERIODIC': 0,                                #non-periodic boundary conditions
'NUM_BLOCKS': 1,                              # <number of files per snapshot>
'NUM_SNAPS': 1,                               # <total number of snapshots>
'INBASE': '"' + dataDir + 'snapshots"',               #"/directory/where/files/are/located"
'FILENAME': '"Galbox_<snap>.<block>.bin"',       #"my_sim.<snap>.<block>"
'NUM_WRITERS': 1,                             #<number of CPUs>
'FORK_READERS_FROM_WRITERS': 1,
'FORK_PROCESSORS_PER_MACHINE': 1,             #<number of processors per node>
}

if pId == 0:
  print "\nFinding halos..."
  print "Output: ", rockstarOutput + '\n'
  if not os.path.exists( rockstarOutput): os.makedirs(rockstarOutput)
  rockstarconfigFile = rockstarOutput + '/rockstar_param.cfg'
  rckFile = open( rockstarconfigFile, "w" )
  for key in rockstarConf.keys():
    rckFile.write( key + " = " + str(rockstarConf[key]) + "\n" )
  for key in parallelConf.keys():
    rckFile.write( key + " = " + str(parallelConf[key]) + "\n" )
  rckFile.close()
  
MPIcomm.Barrier()
start = time.time()
if pId == 0: call([rockstarComand, "-c", rockstarconfigFile ])
if pId == 1:
  time.sleep(5)
  call([rockstarComand, "-c", rockstarOutput + '/auto-rockstar.cfg' ])
print "Time: ", time.time() - start

if pId == 0:
  print "\nMaking consistent-trees..."
  start = time.time()
  outputDir = rockstarOutput + '/'
  #if not ( 'outputs' in os.listdir( outputDir  ) ):
    #print " Trees already there"
  rockstarDir = '/home/bruno/apps/rockstar-galaxies/'
  print " Writing trees"
  treesConfigComand = rockstarDir + 'scripts/gen_merger_cfg.pl'
  rockstarconfigFile = outputDir + 'rockstar.cfg'
  treesConfigFile = outputDir + 'outputs/merger_tree.cfg'
  treesDir = '/home/bruno/apps/consistent-trees'
  treesCommand = treesDir + '/do_merger_tree_np.pl'
  call(['perl', treesConfigComand, rockstarconfigFile ])
  os.chdir( treesDir )
  call(['perl', treesCommand, treesConfigFile ])
  treesToHalosCommand = treesDir + '/halo_trees_to_catalog.pl'
  call(['perl', treesToHalosCommand, treesConfigFile ])
  os.chdir( currentDirectory )
  print " Time: ", time.time() - start