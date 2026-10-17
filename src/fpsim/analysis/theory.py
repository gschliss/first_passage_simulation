"""Closed-form results used to validate the simulator."""

from __future__ import annotations

import numpy as np


def mfpt_spherical_shell(dim: int, D: float, R: float, a: float, r0: float) -> float:
    """MFPT from radius r0 to an absorbing ball (radius a) inside a reflecting ball (radius R).

    Solves D * Laplacian(T) = -1 radially with T(a) = 0, T'(R) = 0:

        T(r0) = 1/(d D) * [ R^d * I_d(a, r0) - (r0^2 - a^2) / 2 ]

    where I_d = integral_a^r0 s^(1-d) ds.
    """
    if dim == 2:
        integral = np.log(r0 / a)
    else:
        integral = (r0 ** (2 - dim) - a ** (2 - dim)) / (2 - dim)
    return (R**dim * integral - (r0**2 - a**2) / 2) / (dim * D)
