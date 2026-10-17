"""Cell clusters on close-packed lattices."""

from fpsim.lattice.base import Cluster, build_cluster
from fpsim.lattice.hcp3d import hcp_cluster
from fpsim.lattice.hex2d import hex_cluster
from fpsim.lattice.layer import layer_factor


def make_cluster(geometry) -> Cluster:
    """Build the cluster described by a ``config.Geometry``."""
    if geometry.dim == 2:
        return hex_cluster(geometry.cluster_rings, geometry.cell_radius, geometry.spacing)
    return hcp_cluster(geometry.cluster_rings, geometry.cell_radius, geometry.spacing)


__all__ = ["Cluster", "build_cluster", "hcp_cluster", "hex_cluster", "layer_factor", "make_cluster"]
