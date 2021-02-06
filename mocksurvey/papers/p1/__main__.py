from . import hod_vs_survey

if __name__ == "__main__":
    import argparse

    avail_cmds = {
        "runmcmc": hod_vs_survey.runmcmc,
        "wpreals": hod_vs_survey.wpreals,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("COMMAND", help=f"Options: {list(avail_cmds.keys())}")
    parser.add_argument("ARG", nargs="*",
                        help="Positional argument(s) passed to COMMAND")
    parser.add_argument("--kwarg", action="append", nargs=2,
                        metavar=("KEY", "VALUE"),
                        help="Keyword argument(s) passed to COMMAND")
    a = parser.parse_args()

    args = [eval(arg) for arg in a.ARG]
    if a.kwarg is None:
        kwargs = {}
    else:
        kwargs = {key.replace("-", "_"): eval(value) for key, value in a.kwarg}

    cmd = avail_cmds[a.COMMAND.lower()]
    cmd(*args, **kwargs)
