from sys import argv
import argparse
from .scripts import scripts

available_scripts = {
    "lightcone": scripts.lightcone,
    "set-data-path": scripts.set_data_path,
    "download-um": scripts.download_um,
    "download-uvista": scripts.download_uvista,
    "config": scripts.config,
}

if __name__ == "__main__":
    argv[0] = "python -m mocksurvey"
    desc = "You can execute several mocksurvey functions from the command line"

    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser(description=desc)

    # Mandatory positional arguments
    parser.add_argument("COMMAND", type=str, help="Options: "
                                                  f"{set(available_scripts.keys())}")

    # Optional positional arguments
    parser.add_argument("ARGS", type=str, default=[], nargs="*",
                        help="Positional arguments for COMMAND")

    a = parser.parse_args()

    if not argv[1] in available_scripts:
        raise IOError(f"Unknown command \"{argv[1]}\".\nOptions: "
                      f"{set(available_scripts.keys())}")

    # Run the specified command
    argv.pop(0)
    available_scripts[argv[0]]()
