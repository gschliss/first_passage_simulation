"""Cell clusters on close-packed lattices."""

from fpsim.lattice.base import Cluster, build_cluster
from fpsim.lattice.hex2d import hex_cluster
from fpsim.lattice.layer import layer_factor


def make_cluster(geometry) -> Cluster:
    """Build the cluster described by a ``config.Geometry``."""
    if geometry.dim == 2:
        return hex_cluster(geometry.cluster_rings, geometry.cell_radius, geometry.spacing)
    raise NotImplementedError("3D HCP lattice comes in build step 5")


__all__ = ["Cluster", "build_cluster", "hex_cluster", "layer_factor", "make_cluster"]
