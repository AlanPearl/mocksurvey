import halotools as ht


def make_predictor_UMmags(UMhalos, z_avg, photbands=None, dz=0.2, nwin=501):
    """
    Generate a function to predict the absolute magnitude in a number of photometric bands of UniverseMachine galaxies. This is done by fitting to UltraVISTA data.

    First, sSFR_uv is calculated for UniverseMachine galaxies by conditional abundance matching (CAM) sSFR to UltraVISTA sSFR_uv values. Then, a random forest is trained to predict MJ,MY from sSFR_uv and redshift. Errors are ~0.1 dex or ~0.25 mag.

    Parameters
    ----------
    UMhalos : DataFrame
        Must contain columns "obs_sm" and "obs_sfr", storing stellar mass and SFR of all UniverseMachine

    z_avg : float
        Redshift slice of the UniverseMachine data

    photbands : list of length-1 strings
        Which bands to allow prediction for

    dz : float
        UVISTA redshifts will be selected to be between z_avg - dz/2 and z_avg + dz/2

    nwin : int (must be odd number)
        Number of windows to divide stellar mass into for CAM

    Returns
    -------
    Predictor : Function with signature f(s,z) -> tuple(MK,MH,...)
        Once the redshift is measured for each galaxy, you can use this function to predict its absolute magnitudes. z, MK, MH, ... are all arrays of shape (s.sum(),), where s is the boolean survey selection mask of shape (len(UMhalos),). Order is assumed to be preserved in UMhalos and z.
    """
    if photbands is None:
        photbands = ["k", "i", "j", "y", "g", "r"]
    else:
        photbands = [s.lower() for s in photbands]
        if not ("i" in photbands):
            photbands.append("i")
        if not ("k" in photbands):
            photbands.append("k")

    UVISTA = UVISTACache()
    UVISTA.PHOTBANDS = {k:UVISTA.PHOTBANDS[k] for k in photbands}
    UVISTAcat = UVISTA.load()

    logm = np.log10(UMhalos["obs_sm"])
    logssfr = np.log10(UMhalos["obs_sfr"]) - logm


    uvista_z = UVISTAcat["z"]
    uvista_logm = UVISTAcat["logm"]
    uvista_logssfr_uv = np.log10(UVISTAcat["sfr_uv"]) - uvista_logm

    names = ["M_" + key.upper() for key in UVISTACache.PHOTBANDS.keys()]
    uvista_m2l = [uvista_logm + UVISTAcat[name]/2.5 for name in names]

    s = np.isfinite(uvista_logssfr_uv)
    x = np.array([uvista_logssfr_uv, uvista_z]).T[s]
    y = np.array(uvista_m2l).T[s]

    from sklearn import ensemble
    reg = ensemble.RandomForestRegressor(n_estimators=10)
    reg.fit(x, y)

    s = (z_avg-dz/2. <= uvista_z) & (uvista_z <= z_avg+dz/2.)
    logssfr_uv = empirical_models.conditional_abunmatch(logm, logssfr,
                        uvista_logm[s], uvista_logssfr_uv[s], nwin=nwin)

    predictor = _make_predictor_UMmags(reg, logm, logssfr_uv, photbands)
    return predictor

def _make_predictor_UMmags(regressor, logm, logssfr_uv, photbands):
    """Define predictor in another function to
    reduce the memory required for closure"""
    def predictor(selection, redshifts):
        """
        predict absolute magnitudes for UniverseMachine galaxies, given individual observed redshifts for each galaxy
        """
        x = np.array([logssfr_uv[selection], redshifts]).T
        y = regressor.predict(x)
        Mag = -2.5 * (np.asarray(logm)[selection,None] - y)
        return pd.DataFrame(Mag, columns=photbands)
    return predictor
