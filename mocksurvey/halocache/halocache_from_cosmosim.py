# Read ascii table into halotools cache
from sys import argv, exit
from halotools.sim_manager import RockstarHlistReader

overwrite = False
for i in range(len(argv)):
    if argv[i] == "--overwrite":
        overwrite = True
        del argv[i]
if not len(argv) == 6:
    print('Usage: halocache_from_cosmosim.py [--overwrite] [datafile] [simname] [redshift] [Lbox] [particle mass]')
    exit()

input_fname = argv[1]
simname = argv[2]  # e.g. smdpl
redshift = float(argv[3])  # e.g. 1.0
Lbox = float(argv[4])  # e.g. 400.0
particle_mass = float(argv[5])  # e.g. 9.63e7

output_fname = 'std_cache_loc'
halo_finder = 'rockstar'
version_name = 'my_cosmosim_halos'

# data = np.genfromtxt(filename, names=True)
# print(data.dtype)

columns_to_keep_dict = {
    'halo_rowid': (0, 'i8'),
    'halo_id': (1, 'i8'),
    'halo_upid': (2, 'i4'),
    'halo_mvir': (3, 'f4'),
    'halo_rvir': (4, 'f4'),
    'halo_rs': (5, 'f4'),
    'halo_vmax': (6, 'f4'),
    'halo_x': (7, 'f4'),
    'halo_y': (8, 'f4'),
    'halo_z': (9, 'f4'),
    'halo_vx': (10, 'f4'),
    'halo_vy': (11, 'f4'),
    'halo_vz': (12, 'f4')
}

reader = RockstarHlistReader(input_fname, columns_to_keep_dict, output_fname, simname,
                             halo_finder, redshift, version_name, Lbox, particle_mass,
                             header_char='"', overwrite=overwrite)
reader.read_halocat(['halo_rvir', 'halo_rs'], write_to_disk=True, update_cache_log=True)
