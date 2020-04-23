from sys import argv
import argparse
from .scripts import scripts

available_scripts = {
    "lightcone": scripts.LightCone,
    "set-data-path": scripts.SetDataPath,
    "download-um": scripts.DownloadUM,
    "download-uvista": scripts.DownloadUVISTA,
    "config": scripts.Config,
}

if __name__ == "__main__":
    argv[0] = "python -m mocksurvey"
    desc = "You can execute several mocksurvey functions from the command line"

    parser = argparse.ArgumentParser(prog="mocksurvey", description=desc)
    sub = parser.add_subparsers(dest="COMMAND", required=True)
    funcs = {name: script(sub.add_parser(name, help=script.desc))
             for name, script in available_scripts.items()}

    # # Mandatory positional arguments
    # parser.add_argument("COMMAND", type=str, help="Options: "
    #                                               f"{set(available_scripts.keys())}")
    #
    # # Optional positional arguments
    # parser.add_argument("ARGS", type=str, default=[], nargs="*",
    #                     help="Positional arguments for COMMAND")

    a = parser.parse_args()
    if a.COMMAND not in available_scripts:
        raise IOError(f"Unknown command: \"{a.COMMAND}\"")

    # Run the specified command
    # argv.pop(0)
    funcs[a.COMMAND]()
