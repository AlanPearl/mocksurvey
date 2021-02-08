"""
Instructions:
=============
Run directly with `python test_lightcone.py`.
Running with nosetests command does not work.

Prerequisites:
==============
- python -m mocksurvey download-uvista
- python -m mocksurvey download-um 1
- python -m mocksurvey config UM set-lightcone-executable FILEPATH
"""

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
        ms.umcool.lightcone(0.8, 1.0, 5, 5, rseed=1234567,
                            outfilepath=path, calibration="uvista",
                            outfilebase="test_lightcone_tmp",
                            keep_ascii_files=True)

        with ms.util.suppress_stdout():
            data = ms.umcool.util.load_ascii_data(dat_fn, 0, 0)
            test = ms.umcool.util.load_ascii_data(test_dat_fn, 0, 0)
        with open(json_fn) as f:
            j = json.load(f)
        with open(test_json_fn) as f:
            j_test = json.load(f)
        n, n_test = np.load(npy_fn), np.load(test_npy_fn)

        os.remove(os.path.join(path, "test_lightcone_tmp_0.dat"))
        os.remove(os.path.join(path, "test_lightcone_tmp_0.npy"))
        os.remove(os.path.join(path, "test_lightcone_tmp_0.json"))

        assert np.all(data == test), "generated lightcone is not as expected"
        assert j == j_test, "json not as expected"
        for key in n_test.dtype.names:
            if key.startswith("m_"):
                assert n_test[key].min() < n[key].mean() < n_test[key].max(), \
                    f"{key} column not as expected"


class TestSelector(unittest.TestCase):
    def test_fullsky_and_square(self):
        testdata = ms.util.make_struc_array(["redshift", "ra", "dec"],
                                            [[0.1, 0.2, 0.3, 0.4],
                                             [-170.0, 1.0, -1.0, 170.0],
                                             [1.0, -1.0, 80.0, -80.0]])

        s1 = ms.LightConeSelector(0, 1)
        s2 = ms.LightConeSelector(0, 1, sqdeg=15.0, fieldshape="square")
        s3 = s1 & s2

        assert np.all(s1(testdata))
        assert np.all(s2(testdata) == [0, 1, 0, 0])
        assert np.all(s3(testdata) == [0, 1, 0, 0])

        assert np.all(np.isclose(s1.field_selector.get_fieldshape(rdz=True),
                                 [6.2831826, 6.2831826, 1.]))
        assert np.all(np.isclose(s2.field_selector.get_fieldshape(rdz=True),
                                 [0.06760275, 0.06760275, 1.]))
        assert np.all(np.isclose(s3.field_selector.get_fieldshape(rdz=True),
                                 [0.06760275, 0.06760275, 1.]))


if __name__ == "__main__":
    unittest.main()
