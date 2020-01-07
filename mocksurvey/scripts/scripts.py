from sys import argv
import argparse

def lightcone():
    from ..ummags.ummags import makelightcones

    desc = ("Creates a UniverseMachine lightcone that includes magnitudes "
            "matched to UltraVISTA.")
    parser = argparse.ArgumentParser(description=desc,
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Mandatory positional arguments
    parser.add_argument("z_low", type=float, help="Minimum redshift")
    parser.add_argument("z_high", type=float, help="Maximum redshift")
    parser.add_argument("x_arcmin", type=float, help="Horizontal field "
                                                     "of view sidelength")
    parser.add_argument("y_arcmin", type=float, help="Vertical field "
                                                     "of view sidelength")
    # Optional positional arguments
    parser.add_argument("samples", type=int, default=1, nargs="?",
                        help="Number of realizations to create")
    parser.add_argument("photbands", type=str, nargs="?",
        help="String of characters specifying which magnitude bands to get")

    # Arguments to specify paths
    parser.add_argument("--outfilepath", metavar="PATH", help="directory to "
                "place output files")
    parser.add_argument("--outfilebase", metavar="NAME", help="base of the "
                "filename to construct the output lightcones")
    parser.add_argument("--id-tag", metavar="NAME", help="name of the survey "
                "in filename (ignored if outfilebase specified)")

    parser.add_argument("--executable", metavar="PATH", help="path to the "
                "lightcone executable from the UniverseMachine package")
    parser.add_argument("--umcfg", metavar="PATH", help="path to the "
                "configuration file required by the lightcone executable")

    # Specify mass limit / random seed
    parser.add_argument("--obs-mass-limit", type=float, default=8e8,
                    metavar="CUT", help="cut to place on 'obs_sm' column")
    parser.add_argument("--true-mass-limit", type=float, default=0,
                    metavar="CUT", help="cut to place on 'true_sm' column")
    parser.add_argument("--rseed", type=int, metavar="R", help="Random "
                                            "seed for this realization")

    # Arguments with not much use
    parser.add_argument("--ra-center", type=float, default=0, metavar="X",
                help="center around this right-ascension")
    parser.add_argument("--dec-center", type=float, default=0, metavar="Y",
                help="center around this declination")
    parser.add_argument("--theta-center", type=float, default=0, metavar="Z",
                help="third rotation angle around ra/dec center")
    parser.add_argument("--do-collision-test", action="store_true",
                        help="not recommended for large lightcones")

    a = parser.parse_args()

    makelightcones(a.z_low, a.z_high, a.x_arcmin, a.y_arcmin,
        executable=a.executable, umcfg=a.umcfg, samples=a.samples,
        photbands=a.photbands, keep_ascii_files=a.keep_ascii_files,
        obs_mass_limit=a.obs_mass_limit, true_mass_limit=a.true_mass_limit,
        outfilepath=a.outfilepath, outfilebase=a.outfilebase, id_tag=a.id_tag,
        do_collision_test=a.do_collision_test, ra=a.ra_center,
        dec=a.dec_center, theta=a.theta_center, rseed=a.rseed)
