class S82MGCSurvey:
    # Stripe 82 Massive Galaxy Catalog (has SDSS photometry)
    # Parameters taken from Bundy+ 2015
    sqdeg = 139.4
    zrange = [0.0, 100.0]  # no explicit limits

    # Not sure on the following -- taken from intro paragraph 4...
    max_dict = dict(m_r=22.5)
    completeness = 0.9


s82mgc = S82MGCSurvey()
