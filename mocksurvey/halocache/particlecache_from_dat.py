# Read ascii table into halotools cache
from sys import argv, exit
import pathlib
import numpy as np
from halotools.sim_manager import UserSuppliedPtclCatalog

overwrite = False
for i in range(len(argv)):
    if argv[i - overwrite] == "--overwrite":
        overwrite = True
        del argv[i]
if not len(argv) == 6:
    print(
        'Usage: halocache_from_cosmosim.py [--overwrite] [datafile] '
        '[simname] [redshift] [Lbox] [particle mass]')
    exit()

input_fname = argv[1]
simname = argv[2]  # e.g. smdpl
redshift = float(argv[3])  # e.g. 1.0
Lbox = float(argv[4])  # e.g. 400.0
particle_mass = float(argv[5])  # e.g. 9.63e7

version_name = 'my_cosmosim_halos'
processing_notes = f'processed via {" ".join(argv)}'
output_file = str(pathlib.Path(input_fname).absolute()) + ".hdf5"

names = ["x", "y", "z", "vx", "vy", "vz", "ptcl_ids"]
types = [*(np.float32,)*6, np.int64]

dtype = np.dtype(list(zip(names, types)))
dat = np.loadtxt(input_fname, dtype=dtype)

ptcl_catalog = UserSuppliedPtclCatalog(
    redshift=redshift, Lbox=Lbox, particle_mass=particle_mass,
    **{name: dat[name] for name in names})

ptcl_catalog.add_ptclcat_to_cache(output_file, simname, version_name,
                                  processing_notes, overwrite=overwrite)
