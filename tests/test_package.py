"""The package imports and reports its version."""
import re

import geolip.modular_diffusion as md


def test_version_is_a_release_string():
    assert re.fullmatch(r"\d+\.\d+\.\d+", md.__version__)


def test_namespace_package_has_no_init_at_the_top():
    # geolip is a namespace shared with sibling packages: it must not carry an __init__.py of its own.
    import geolip

    assert getattr(geolip, "__file__", None) is None
