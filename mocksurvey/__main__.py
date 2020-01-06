from sys import argv
from .scripts import scripts

available_scripts = {"lightcone": scripts.lightcone,
                     }

if __name__ == "__main__":
    # Error message if first argument is not a valid command
    argv.pop(0)
    msg = f"Usage: python -m mocksurvey {set(available_scripts.keys())}"
    if len(argv) == 0:
        raise IOError(msg)
    if not argv[0] in available_scripts:
        raise IOError(f"Unknown command \"{argv[0]}\". {msg}")

    available_scripts[argv[0]]()
