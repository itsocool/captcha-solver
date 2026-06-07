import sys

from fastapi_cli.cli import main as fastapi_cli_main


DEV_DEFAULT_ARGS = {
    "--host": "0.0.0.0",
    "--port": "8000",
}


def _has_option(args: list[str], option: str) -> bool:
    return option in args or any(arg.startswith(f"{option}=") for arg in args)


def _inject_dev_defaults() -> None:
    args = sys.argv[1:]

    if not args or args[0] != "dev" or "-h" in args or "--help" in args:
        return

    for option, value in DEV_DEFAULT_ARGS.items():
        if not _has_option(args, option):
            sys.argv.extend([option, value])


def main() -> None:
    _inject_dev_defaults()
    fastapi_cli_main()
