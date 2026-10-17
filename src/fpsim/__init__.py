"""fpsim: agent-based first-passage simulations across diffusion dimensions."""

from fpsim.agents import BrownianWalkers
from fpsim.geometry import Geometry, SphericalShell
from fpsim.simulation import SimulationResult, run_simulation

__all__ = [
    "BrownianWalkers",
    "Geometry",
    "SphericalShell",
    "SimulationResult",
    "run_simulation",
]
__version__ = "0.1.0"
