class PFSSurvey:
    z_samples = {
        0: {
            "zrange": [0.7, 1.0],
            "max_dict": dict(m_y=22.5, m_j=22.8)
        },
        1: {
            "zrange": [1.0, 1.7],
            "max_dict": dict(m_j=22.8)
        }
    }
    sqdeg = 12.0
    completeness = 0.7

    def __init__(self, z_sample):
        self.zrange = self.z_samples[z_sample]["zrange"]
        self.max_dict = self.z_samples[z_sample]["max_dict"]


pfs_low = PFSSurvey(0)
pfs_high = PFSSurvey(1)
