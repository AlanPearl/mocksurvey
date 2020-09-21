import numpy as np


class MOONRISESurvey:
    # Parameters taken from Maiolino+ 2020
    z_samples = {
        0: {
            "zrange": [0.9, 1.1],
            "max_dict": dict(m_h=23)
        },

        1: {
            "zrange": [1.2, 1.7],
            "max_dict": dict(m_h=23.5)
        },
        2: {
            "zrange": [2.0, 2.6],
            "max_dict": dict(m_h=24)
        },
        3: {
            "zrange": [5.0, float("inf")],
            "max_dict": dict(m_h=26)
        }
    }
    fields = {
        # Cosmic Evolution Survey field
        "COSMOS": {
            "completeness": 0.8,
            "schemes": {
                "Xswitch": {"sqdeg": 1.0},
                "Stare": {"sqdeg": 1.0}
            }
        },
        # Two fields from the VIDEO survey
        "XMM-LSS": {
            "completeness": 0.7,
            "schemes": {
                "Xswitch": {"sqdeg": 3.0 / 2},
                "Stare": {"sqdeg": 6.0 / 2}
            }
        },
        "ECDFS": {
            "completeness": 0.7,
            "schemes": {
                "Xswitch": {"sqdeg": 3.0 / 2},
                "Stare": {"sqdeg": 6.0 / 2}
            }
        }
    }

    def __init__(self, z_sample, field, scheme):
        """
        This survey has several redshift samples, fields, and
        targeting schemes. Therefore, you must provide:

        Parameters
        ----------
        z_sample : int
            0, 1, 2, or 3. Larger integers = higher redshift
        field : str
            "COSMOS", "XMM-LSS", "ECDFS", or "combine" which
            adds the sky areas together and takes a weighted
            average of the completeness in each field
        scheme : str
            "Xswitch" or "Stare". Targeting scheme used in the
            VIDEO fields. Stare is more efficient and doubles
            the sky area that can be covered.
        """
        if field == "combine":
            self.sqdeg = sum(x["schemes"][scheme]["sqdeg"]
                             for x in self.fields.values())
        else:
            self.sqdeg = self.fields[field]["schemes"][scheme]["sqdeg"]

        self.zrange = self.z_samples[z_sample]["zrange"]
        self.max_dict = self.z_samples[z_sample]["max_dict"]

        if field == "combine":
            a = [x["schemes"][scheme]["sqdeg"] for x in self.fields.values()]
            w = [x["completeness"] for x in self.fields.values()]
            self.completeness = np.average(a, weights=w)
        else:
            self.completeness = self.fields[field]["completeness"]


# For now let's combine all fields together and use
# the optimistic "Stare" scheme
moons_low = MOONRISESurvey(0, "combine", "Stare")
moons_mid = MOONRISESurvey(1, "combine", "Stare")
moons_high = MOONRISESurvey(2, "combine", "Stare")

# Same as before, using the "Xswitch" scheme (more likely)
moons_x_low = MOONRISESurvey(0, "combine", "Xswitch")
moons_x_mid = MOONRISESurvey(1, "combine", "Xswitch")
moons_x_high = MOONRISESurvey(2, "combine", "Xswitch")
