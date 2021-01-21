class BOSSCMASSSurvey:
    sqdeg = 10_000.0  # effectively full-sky for our mock
    zrange = [0.0, 100.0]  # no explicit limits

    @staticmethod
    def custom_selector(data):
        import numpy as np

        # Info from http://www.sdss3.org/dr9/algorithms/boss_galaxy_ts.php
        g, r, i = data["m_g"], data["m_r"], data["m_i"]
        dperp = (r - i) - (g - r) / 8.0

        cut1 = dperp > 0.55
        cut2 = i < 19.86 + 1.6 * (dperp - 0.8)
        cut3 = (17.5 < i) & (i < 19.9)
        cut4 = r - i < 2.0
        # Cuts we cannot perform
        # ======================
        # cut5 = i_fib2 < 21.5
        # cut6 = (i_psf - i_mod) > 0.2 + 0.2 * (20.0 - i_mod)
        # cut7 = (z_psf - z_mod) > 9.125 - 0.46 * z_mod
        return np.all([cut1, cut2, cut3, cut4], axis=0)


boss_cmass = BOSSCMASSSurvey()
