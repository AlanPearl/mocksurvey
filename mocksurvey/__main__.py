from sys import argv
from .scripts import scripts

available_scripts = {
    "lightcone": scripts.lightcone,
    "set-data-path": scripts.set_data_path,
    "download-um": scripts.download_um,
    "download-uvista": scripts.download_uvista,
    "config": scripts.config,
}

if __name__ == "__main__":
    # Error message if first argument is not a valid command
    msg = f"Usage: python -m mocksurvey {set(available_scripts.keys())}"
    if len(argv) < 2:
        raise IOError(msg)
    if not argv[1] in available_scripts:
        raise IOError(f"Unknown command \"{argv[1]}\".\n{msg}")

    # Run the specified command
    argv.pop(0)
    available_scripts[argv[0]]()
