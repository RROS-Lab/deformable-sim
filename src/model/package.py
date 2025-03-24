from abc import ABC
from typing import Tuple

import warp as wp


class Package(ABC):
    """
    Abstract base class for a package model.
    """

    def __init__(
        self,
        xform: wp.transform,
        mass: float,
        particle_num_in_between: int,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._xform = xform
        self._mass = mass
        self._particle_num_in_between = particle_num_in_between

        self._size = None
        self._grid_dim_sim = None
        self._grid_dim_real = None
        self._particle_radius = None
        self._particle_mass = None

    @property
    def xform(self) -> wp.transform:
        return self._xform

    @property
    def mass(self) -> float:
        return self._mass

    @property
    def size(self) -> Tuple[float, float, float]:
        return self._size

    @property
    def grid_dim_sim(self) -> Tuple[int, int]:
        return self._grid_dim_sim

    @property
    def grid_dim_real(self) -> Tuple[int, int]:
        return self._grid_dim_real

    @property
    def particle_radius(self) -> float:
        return self._particle_radius

    @property
    def particle_mass(self) -> float:
        return self._particle_mass


class SmallPackage(Package):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._size = (0.2032, 0.254, 0.0004)

        self._grid_dim_real = (5, 6)
        self._grid_dim_sim = (
            self._grid_dim_real[0]
            + self._particle_num_in_between * (self._grid_dim_real[0] - 1),
            self._grid_dim_real[1]
            + self._particle_num_in_between * (self._grid_dim_real[1] - 1),
        )
        self._particle_radius = max(
            self._size[0] / (self._grid_dim_sim[0] - 1),
            self._size[1] / (self._grid_dim_sim[1] - 1),
            self._size[2],
        )

        self._particle_mass = self._mass / (
            2 * self._grid_dim_real[0] * self._grid_dim_real[1]
        )


class MediumPackage(Package):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._size = (0.2286, 0.3048, 0.0003)

        self._grid_dim_real = (5, 6)
        self._grid_dim_sim = (
            self._grid_dim_real[0]
            + self._particle_num_in_between * (self._grid_dim_real[0] - 1),
            self._grid_dim_real[1]
            + self._particle_num_in_between * (self._grid_dim_real[1] - 1),
        )
        self._particle_radius = max(
            self._size[0] / (self._grid_dim_sim[0] - 1),
            self._size[1] / (self._grid_dim_sim[1] - 1),
            self._size[2],
        )

        self._particle_mass = self._mass / (
            2 * self._grid_dim_real[0] * self._grid_dim_real[1]
        )


class LargePackage(Package):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._size = (0.254, 0.3048, 0.0004)

        self._grid_dim_real = (6, 7)
        self._grid_dim_sim = (
            self._grid_dim_real[0]
            + self._particle_num_in_between * (self._grid_dim_real[0] - 1),
            self._grid_dim_real[1]
            + self._particle_num_in_between * (self._grid_dim_real[1] - 1),
        )
        self._particle_radius = max(
            self._size[0] / (self._grid_dim_sim[0] - 1),
            self._size[1] / (self._grid_dim_sim[1] - 1),
            self._size[2],
        )

        self._particle_mass = self._mass / (
            2 * self._grid_dim_real[0] * self._grid_dim_real[1]
        )


def create_package(
    package_type: str, xform: wp.transform, mass: float, particle_in_between: int
) -> Package:
    if package_type == "L":
        return SmallPackage(
            xform=xform, mass=mass, particle_num_in_between=particle_in_between
        )
    elif package_type == "M":
        return MediumPackage(
            xform=xform, mass=mass, particle_num_in_between=particle_in_between
        )
    elif package_type == "H":
        return LargePackage(
            xform=xform, mass=mass, particle_num_in_between=particle_in_between
        )
    else:
        raise ValueError(f"Unknown package type: {package_type}")
