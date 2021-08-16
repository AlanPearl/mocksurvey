class BOSSLOWZSurvey:
    sqdeg = 10_000.0  # effectively full-sky for our mock
    zrange = [0.0, 100.0]  # no explicit limits

    @staticmethod
    def custom_selector(data):
        import numpy as np

        # Info from http://www.sdss3.org/dr9/algorithms/boss_galaxy_ts.php
        g, r, i, cr = data["m_g"], data["m_r"], data["m_i"], data["m_cmod_r"]
        cpar = 0.7 * (g - r) + 1.2 * ((r - i) - 0.18)
        cperp = (r - i) - (g - r) / 4.0 - 0.18

        cut1 = np.abs(cperp) < 0.2
        cut2 = cr < 13.5 + cpar / 0.3
        cut3 = (16.0 < cr) & (cr < 19.6)
        # Cuts we cannot perform
        # ======================
        # cut4 = r_psf - r_cmod > 0.3
        return np.all([cut1, cut2, cut3], axis=0)


boss_lowz = BOSSLOWZSurvey()
