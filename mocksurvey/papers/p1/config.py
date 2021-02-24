import numpy as np
import mocksurvey as ms


def lnprior(sigma, alpha, fsat=0.5):
    if (1e-5 <= sigma <= 5.0) & (0.1 <= alpha <= 3.0) & (0 <= fsat <= 1):
        return 0
    else:
        return -np.inf


class ModelConfig:
    rp_edges = np.geomspace(1, 27, 7)
    boxsize = 250.0

    params = {
        "w": {
            "redshift": 0.647
        },
        "p": {
            "redshift": 0.979
        },
        "m": {
            "redshift": 1.367
        }
    }

    def __init__(self, name):
        """
        This class configures the model in which we predict wp(rp)
        from an HOD populated over a periodic cube at a given
        snapshot in redshift.

        Parameters
        ----------
        name : str
            First character specifies survey w = WAVES, p = PFS, m = MOONS.
            This only affects the which redshift will be used.
        """
        self.redshift = self.params[name[0]]["redshift"]
        self.threshold = SurveyParamGrid.params[name[0]]["threshold"]


class SurveyParamGrid:
    rp_edges = ModelConfig.rp_edges

    params = {
        "w": {
            "threshold": 10 ** 11,
            "zlim": [0.5, 0.8],
            "sqdeg": ms.surveys.waves.sqdeg,
            "completeness": ms.surveys.waves.completeness,
            "completeness_grid": np.array([0.6, 0.75, 0.875, 0.95, 1.0])
        },
        "p": {
            "threshold": 10 ** 10.5,
            "zlim": [0.8, 1.2],
            "sqdeg": ms.surveys.pfs_low.sqdeg,
            "completeness": ms.surveys.pfs_low.completeness,
            "completeness_grid": np.linspace(0.4, 1.0, 5)
        },
        "m": {
            "threshold": 10 ** 10,
            "zlim": [1.2, 1.6],
            "sqdeg": ms.surveys.moons_x_mid.sqdeg,
            "completeness": ms.surveys.moons_x_mid.completeness,
            "completeness_grid": np.linspace(0.475, 0.975, 5)
        },
    }

    def __init__(self, name):
        """
        This class controls the survey parameters according
        to the given name of the grid

        Parameters
        ----------
        name : str
            First character specifies survey (w = WAVES, p = PFS, m = MOONS)
            Last character specifies grid number (1 = const area, 2 = const N)
        """
        self.zlim = self.params[name[0]]["zlim"]
        self.threshold = self.params[name[0]]["threshold"]
        self.completeness_grid = self.params[name[0]]["completeness_grid"]

        self.true_sqdeg = self.params[name[0]]["sqdeg"]
        self.true_completeness = self.params[name[0]]["completeness"]
        if int(name[-1]) == 1:
            grid = np.full_like(self.completeness_grid, self.true_sqdeg)
        elif int(name[-1]) == 2:
            grid = self.true_sqdeg * (self.true_completeness / self.completeness_grid)
        else:
            raise ValueError(f"gridnum={name[-1]} but must be either 1 or 2.")
        self.sqdeg_grid = grid
