"""Launch the Streamlit dashboard from any working directory."""
from pathlib import Path
import sys


def main():
    from streamlit.web import cli
    import tomllib
    directory = Path(__file__).parent
    with (directory / "theme.toml").open("rb") as handle:
        config = tomllib.load(handle)

    def flags(data, prefix=""):
        for key, value in data.items():
            option = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                yield from flags(value, option)
            else:
                yield f"--{option}={str(value).lower() if isinstance(value, bool) else value}"
    sys.argv = ["streamlit", "run", str(directory / "dashboard.py"), *flags(config), *sys.argv[1:]]
    raise SystemExit(cli.main())


if __name__ == "__main__":
    main()
