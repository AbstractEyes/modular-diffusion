"""modular-diffusion: the entry point for the editor's run button.

The work is done by the package's modules (see README.md). This file prints the version and how to run the tests:

    pip install -r requirements.txt
    pip install --no-deps -e .
    pytest
"""
from geolip.modular_diffusion import __version__

if __name__ == "__main__":
    print(f"modular-diffusion {__version__}")
    print(__doc__)
