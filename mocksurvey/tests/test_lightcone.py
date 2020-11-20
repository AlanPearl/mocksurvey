import unittest
import os
import json
import numpy as np

import mocksurvey as ms


class TestLightCone(unittest.TestCase):

    # TODO: Test the distmod column of a lightcone (something seems off)

    def test_lightcone(self):
        path = os.path.dirname(os.path.realpath(__file__))
        dat_fn = os.path.join(path, "test_lightcone_tmp_0.dat")
        test_dat_fn = os.path.join(path, "test_lightcone_0.dat")
        json_fn = os.path.join(path, "test_lightcone_tmp_0.json")
        test_json_fn = os.path.join(path, "test_lightcone_0.json")
        npy_fn = os.path.join(path, "test_lightcone_tmp_0.npy")
        test_npy_fn = os.path.join(path, "test_lightcone_0.npy")

        # Might want to suppress stdout here as well
        ms.ummags.lightcone(0.8, 1.0, 5, 5, rseed=1234567,
                            outfilepath=path,
                            outfilebase="test_lightcone_tmp",
                            keep_ascii_files=True)

        with ms.util.suppress_stdout():
            data = ms.ummags.util.load_ascii_data(dat_fn, 0, 0)
            test = ms.ummags.util.load_ascii_data(test_dat_fn, 0, 0)
        j, j_test = json.load(open(json_fn)), json.load(open(test_json_fn))
        n, n_test = np.load(npy_fn), np.load(test_npy_fn)

        os.remove(os.path.join(path, "test_lightcone_tmp_0.dat"))
        os.remove(os.path.join(path, "test_lightcone_tmp_0.npy"))
        os.remove(os.path.join(path, "test_lightcone_tmp_0.json"))

        assert np.all(data == test)
        assert j == j_test
        for key in n_test.dtype.names:
            if key.startswith("m_"):
                assert n_test[key].min() < n[key].mean() < n_test[key].max()


if __name__ == "__main__":
    unittest.main()
