import matplotlib.pyplot as plt

from mocksurvey import *

def mock_testsuite(simbox=None, center=None, scheme=None, sqdeg=None):
    """
    Test a variety of centers and selection limits over the default SimBox.
    The SimBox/centers/lims can be given, but the outputs will not be
    tested against a known output
    """
    if (simbox is None) and (center is None) and (scheme is None) and (sqdeg is None):
        test_exact_output = True
    else:
        test_exact_output = False
    if simbox is None:
        simbox = SimBox()
    if center is None:
        center = [200,200,200]
    if scheme is None:
        scheme = "hex"
    if sqdeg is None:
        sqdeg = 15.
    
    test_field(simbox, center, scheme, sqdeg, test_exact_output=test_exact_output)

def test_field(simbox, center, scheme, sqdeg, cartesian_distortion=True, test_exact_output=False):

    field = simbox.field(center=center, scheme=scheme, sqdeg=sqdeg, cartesian_distortion=cartesian_distortion)
    while True:
        print("Show plots for center=%s, sqdeg=%s, cartesian_distortion=%s?"%(center, sqdeg, cartesian_distortion), end="")
        user_input = input("(y/n) [q to quit] ==> ").lower()
        make_plots = end_all = False
        if user_input.startswith("y"):
            make_plots = True
            break
        elif user_input.startswith("n"):
            break
        elif user_input.startswith("q"):
            end_all = True
            break

    if end_all:
        exit()

    if make_plots:
        mass = tp.plot_halo_mass(field)
        plt.show()
        data_xyz, rand_xyz = tp.plot_pos_scatter(field)
        plt.show()
        data_rdz, rand_rdz = tp.plot_sky_scatter(field)
        plt.show()
        xi_real = tp.plot_xi_rp_pi(field, realspace=True)
        plt.show()
        xi_red = tp.plot_xi_rp_pi(field)
        plt.show()
        wp = tp.plot_wp_rp(field)
        plt.show()
    else:
        mass = tp.plot_halo_mass(field, plot=False)
        data_xyz, rand_xyz = tp.plot_pos_scatter(field, plot=False)
        data_rdz, rand_rdz = tp.plot_sky_scatter(field, plot=False)
        xi_real = tp.plot_xi_rp_pi(field, realspace=True, plot=False)
        xi_red = tp.plot_xi_rp_pi(field, plot=False)
        wp = tp.plot_wp_rp(field, plot=False)
    
    if test_exact_output and False:
        np.save("test-suite/data_xyz", data_xyz)
        np.save("test-suite/data_rdz", data_rdz)
        np.save("test-suite/rand_xyz", rand_xyz)
        np.save("test-suite/rand_rdz", rand_rdz)
        np.save("test-suite/mass", mass)
        np.save("test-suite/xi_real", xi_real)
        np.save("test-suite/xi_red", xi_red)
        np.save("test-suite/wp", wp)
