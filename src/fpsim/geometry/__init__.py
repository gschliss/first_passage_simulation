"""Domains in which agents diffuse, with their boundaries and targets."""

from fpsim.geometry.base import Geometry
from fpsim.geometry.sphere import SphericalShell

GEOMETRIES = {
    "spherical_shell": SphericalShell,
}

__all__ = ["Geometry", "SphericalShell", "GEOMETRIES"]
