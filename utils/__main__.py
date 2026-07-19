"""
CLI entry point for `python -m utils`.

Subcommands:
    run   — Run merge pipeline from config
    clean — Run cleaning pipeline
    full  — Run merge + clean in one shot
"""

from utils.pipeline import main

if __name__ == "__main__":
    main()
