"""Enable `python -m muck …` — a PATH-independent way to run the CLI."""

from .cli import app

if __name__ == "__main__":
    app()
