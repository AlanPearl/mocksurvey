from sys import argv
import argparse
from . import scripts

available_scripts = {
    "lightcone": scripts.LightCone,
    "lightcone-selection": scripts.LightConeSelection,
    "lightcone-spectra": scripts.LightConeSpectra,
    "recalibrate-lightcone": scripts.RecalibrateLightCone,
    "set-data-path": scripts.SetDataPath,
    "download-um": scripts.DownloadUM,
    "download-uvista": scripts.DownloadUVISTA,
    "download-uvista-mock-spectra": scripts.DownloadUVISTAMockSpectra,
    # TODO: "download-sdss": scripts.DownloadSDSS,
    "config": scripts.Config,
}

if __name__ == "__main__":
    argv[0] = "python -m mocksurvey"
    desc = "You can execute several mocksurvey functions from the command line"

    parser = argparse.ArgumentParser(prog="mocksurvey", description=desc)
    sub = parser.add_subparsers(dest="PROGRAM", required=True)
    funcs = {name: script(sub.add_parser(name, help=script.desc))
             for name, script in available_scripts.items()}

    a = parser.parse_args()
    if a.PROGRAM not in available_scripts:
        raise IOError(f"Unknown command: \"{a.PROGRAM}\"")

    # Run the specified command
    argv.pop(0)
    funcs[a.PROGRAM]()
