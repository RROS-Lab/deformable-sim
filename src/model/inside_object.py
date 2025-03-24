import os

from abc import ABC
from typing import Tuple

import numpy as np
import warp as wp


class InternalObject(ABC):
    """
    Abstract base class for an inside object model.
    """

    def __init__(
        self,
        xform: wp.transform,
        mass: float,
        object_type: str,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._xform = xform
        self._mass = mass

        self._size = None
        self._inertia = None
        self._mesh_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "assets",
            "internal_objects",
            object_type,
            "object.obj",
        )

        assert os.path.exists(self._mesh_path), f"Mesh file not found: {self._mesh_path}"

    @property
    def xform(self) -> wp.transform:
        return self._xform

    @property
    def mass(self) -> float:
        return self._mass

    @property
    # [0]: radius, [1]: height
    def size(self) -> Tuple[float, float]:
        return self._size

    @property
    def inertia(self) -> wp.mat33:
        return self._inertia
    
    @property
    def mesh(self) -> str:
        return self._mesh_path


class InternalSmallObject(InternalObject):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._size = (0.105, 0.02)
        self._inertia = wp.mat33(
            1.0
            / 12
            * self._mass
            * np.array(
                [
                    [self._size[1] ** 2 + 3 * self._size[0] ** 2, 0, 0],
                    [0, 6 * self._size[0] ** 2, 0],
                    [0, 0, self._size[1] ** 2 + 3 * self._size[0] ** 2],
                ]
            )
        )


class InternalMediumObject(InternalObject):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._size = (0.13, 0.015)
        self._inertia = wp.mat33(
            1.0
            / 12
            * self._mass
            * np.array(
                [
                    [self._size[1] ** 2 + 3 * self._size[0] ** 2, 0, 0],
                    [0, 6 * self._size[0] ** 2, 0],
                    [0, 0, self._size[1] ** 2 + 3 * self._size[0] ** 2],
                ]
            )
        )


class InternalLargeObject(InternalObject):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._size = (0.16, 0.015)
        self._inertia = wp.mat33(
            1.0
            / 12
            * self._mass
            * np.array(
                [
                    [self._size[1] ** 2 + 3 * self._size[0] ** 2, 0, 0],
                    [0, 6 * self._size[0] ** 2, 0],
                    [0, 0, self._size[1] ** 2 + 3 * self._size[0] ** 2],
                ]
            )
        )


def create_internal_object(
    object_type: str, xform: wp.transform, mass: float
) -> InternalObject:
    if object_type == "L":
        return InternalSmallObject(xform=xform, mass=mass, object_type=object_type)
    elif object_type == "M":
        return InternalMediumObject(xform=xform, mass=mass, object_typ=object_type)
    elif object_type == "H":
        return InternalLargeObject(xform=xform, mass=mass, object_type=object_type)
    else:
        raise ValueError(f"Unknown package type: {object_type}")
