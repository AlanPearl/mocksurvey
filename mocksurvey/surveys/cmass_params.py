class CMASSSurvey:
    sqdeg = None  # leave it as full-sky for now
    zrange = [0.0, 100.0]  # no explicit limits

    @staticmethod
    def custom_selector(data):
        import numpy as np

        # Info from http://www.sdss3.org/dr9/algorithms/boss_galaxy_ts.php
        dperp = (data["m_r"] - data["m_i"]) - (data["m_g"] - data["m_r"]) / 8.0

        cut1 = dperp > 0.55
        cut2 = data["m_i"] < 19.86 + 1.6 * (dperp - 0.8)
        cut3 = (17.5 < data["m_i"]) & (data["m_i"] < 19.9)
        cut4 = data["m_r"] - data["m_i"] < 2.0
        return np.all([cut1, cut2, cut3, cut4], axis=0)


cmass = CMASSSurvey()
