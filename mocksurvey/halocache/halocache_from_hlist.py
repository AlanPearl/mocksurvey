# Read ascii table into halotools cache
from sys import argv, exit
from halotools.sim_manager import RockstarHlistReader

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

output_fname = 'std_cache_loc'
halo_finder = 'rockstar'
version_name = 'my_cosmosim_halos'

# scale(0) id(1) desc_scale(2) desc_id(3) num_prog(4) pid(5) upid(6)
# desc_pid(7) phantom(8) sam_mvir(9) mvir(10) rvir(11) rs(12) vrms(13)
# mmp?(14) scale_of_last_MM(15) vmax(16) x(17) y(18) z(19) vx(20) vy(21)
# vz(22) Jx(23) Jy(24) Jz(25) Spin(26) Breadth_first_ID(27) Depth_first_ID(28)
# Tree_root_ID(29) Orig_halo_ID(30) Snap_num(31)
# Next_coprogenitor_depthfirst_ID(32) Last_progenitor_depthfirst_ID(33)

columns_to_keep_dict = {
    'halo_id': (1, 'i8'),
    'halo_pid': (5, 'i8'),
    'halo_upid': (6, 'i4'),
    'halo_mvir': (10, 'f4'),
    'halo_rvir': (11, 'f4'),
    'halo_rs': (12, 'f4'),
    'halo_vmax': (16, 'f4'),
    'halo_x': (17, 'f4'),
    'halo_y': (18, 'f4'),
    'halo_z': (19, 'f4'),
    'halo_vx': (20, 'f4'),
    'halo_vy': (21, 'f4'),
    'halo_vz': (22, 'f4')
}

reader = RockstarHlistReader(input_fname, columns_to_keep_dict, output_fname, simname,
                             halo_finder, redshift, version_name, Lbox, particle_mass,
                             header_char='#', overwrite=overwrite)
reader.read_halocat(['halo_rvir', 'halo_rs'],
                    write_to_disk=True, update_cache_log=True)
