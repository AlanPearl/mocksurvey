# Read ascii table into halotools cache
from sys import argv, exit

overwrite = False
for i in range(len(argv)):
    if argv[i] == "--overwrite":
        overwrite = True
        del argv[i]
if not len(argv) == 6:
    print('Usage: halocache_from_cosmosim.py [--overwrite] [datafile] [simname] [redshift] [Lbox] [particle mass]')
    exit()
from halotools.sim_manager import RockstarHlistReader

input_fname = argv[1]
simname = argv[2] # e.g. smdpl
redshift = float(argv[3]) # e.g. 1.0
Lbox = float(argv[4]) # e.g. 400.0
particle_mass = float(argv[5]) # e.g. 9.63e7

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

# Error Output:
# =============
# Traceback (most recent call last):
#   File "halos_from_ascii.py", line 40, in <module>
#     reader.read_halocat(['halo_rvir', 'halo_rs'], write_to_disk=True, update_cache_log=True)
#   File "/home/alan/local/anaconda3/lib/python3.6/site-packages/halotools/sim_manager/rockstar_hlist_reader.py", line 622, in read_halocat
#     result = self._read_ascii(**kwargs)
#   File "/home/alan/local/anaconda3/lib/python3.6/site-packages/halotools/sim_manager/rockstar_hlist_reader.py", line 685, in _read_ascii
#     return TabularAsciiReader.read_ascii(self, **kwargs)
#   File "/home/alan/local/anaconda3/lib/python3.6/site-packages/halotools/sim_manager/tabular_ascii_reader.py", line 588, in read_ascii
#     self.data_chunk_generator(num_rows_in_chunk, f)), dtype=self.dt)
#   File "/home/alan/local/anaconda3/lib/python3.6/site-packages/halotools/sim_manager/tabular_ascii_reader.py", line 493, in data_chunk_generator
#     yield tuple(parsed_line[i] for i in self.column_indices_to_keep)
#   File "/home/alan/local/anaconda3/lib/python3.6/site-packages/halotools/sim_manager/tabular_ascii_reader.py", line 493, in <genexpr>
#     yield tuple(parsed_line[i] for i in self.column_indices_to_keep)
# IndexError: list index out of range
