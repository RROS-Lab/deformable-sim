from abc import ABC
from typing import Tuple

import warp as wp


class SuctionCup(ABC):
    """
    Abstract base class for a suction cup model.
    """

    def __init__(
        self,
        xform: wp.transform,
        mass: float,
        size: Tuple[float, float, float] = (0.015, 0.04, 0.05),
        grid_dim_sim: Tuple[int, int] = (8, 3),
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._xform = xform
        self._mass = mass

        self._size = size
        self._grid_dim_sim = grid_dim_sim
        self._particle_mass = mass / (grid_dim_sim[0] * grid_dim_sim[1])

    @property
    def xform(self) -> wp.transform:
        return self._xform

    @property
    def mass(self) -> float:
        return self._mass

    @property
    # [0]: inner radius, [1]: outer radius, [2]: zone radius
    def size(self) -> Tuple[float, float, float]:
        return self._size

    @property
    # [0]: circumferential, [1]: radial
    def grid_dim_sim(self) -> Tuple[int, int]:
        return self._grid_dim_sim

    @property
    def particle_mass(self) -> float:
        return self._particle_mass


def create_suction_cup(xform: wp.transform, mass: float) -> SuctionCup:
    """
    Create a suction cup model.
    """
    return SuctionCup(xform, mass)
