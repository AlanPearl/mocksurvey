class WAVESSurvey:
    # Parameters for WAVES-Deep, taken from
    # https://wavesurvey.org/project/survey-design/
    sqdeg = 50.0 + 4.0 * 4  # WD23 + WD01 + WD02 + WD03 + WD10
    zrange = [0.2, 0.8]
    max_dict = dict(m_z=21.25)
    completeness = 0.95  # or greater (unprecedented)


waves = WAVESSurvey()
