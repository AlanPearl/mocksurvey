import numpy as np
import mocksurvey as ms


def lnprior(sigma, alpha, fsat=0.5):
    if (1e-5 <= sigma <= 5.0) & (0.1 <= alpha <= 3.0) & (0 <= fsat <= 1):
        return 0
    else:
        return -np.inf


class ModelConfig:
    rp_edges = np.geomspace(1, 27, 7)

    params = {
        "w": {
            "redshift": 0.65
        },
        "p": {
            "redshift": 1.0
        },
        "m": {
            "redshift": 1.4
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


class SurveyParamGrid:
    rp_edges = ModelConfig.rp_edges

    params = {
        "w": {
            "threshold": 10 ** 11,
            "zlim": [0.5, 0.8],
            "sqdeg": ms.surveys.waves.sqdeg,
            "completeness_grid": np.linspace(0.8, 1.0, 5)
        },
        "p": {
            "threshold": 10 ** 10.55,
            "zlim": [0.8, 1.2],
            "sqdeg": ms.surveys.pfs_low.sqdeg,
            "completeness_grid": np.linspace(0.4, 1.0, 5)
        },
        "m": {
            "threshold": 10 ** 10.1,
            "zlim": [1.2, 1.6],
            "sqdeg": ms.surveys.moons_x_mid.sqdeg,
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
            First character specifies survey w = WAVES, p = PFS, m = MOONS
            Last character specifies grid number (1 = const area or 2 = const N)
        """
        self.zlim = self.params[name[0]]["zlim"]
        self.threshold = self.params[name[0]]["threshold"]
        self.completeness_grid = self.params[name[0]]["completeness_grid"]

        sqdeg = self.params[name[0]]["sqdeg"]
        if int(name[-1]) == 1:
            grid = np.full_like(self.completeness_grid, sqdeg)
        elif int(name[-1]) == 2:
            grid = sqdeg * (self.completeness_grid[2] / self.completeness_grid)
        else:
            raise ValueError(f"gridnum={name[-1]} but must be either 1 or 2.")
        self.sqdeg_grid = grid
