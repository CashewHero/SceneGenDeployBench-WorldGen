"""WorldGen package with a lazy high-level API import.

The panorama runner uses the depth and splat modules directly and should not
load the optional FLUX and Nunchaku stack merely by importing the package.
"""

from typing import Any

__all__ = ["WorldGen"]


def __getattr__(name: str) -> Any:
    if name == "WorldGen":
        from .worldgen import WorldGen

        return WorldGen
    raise AttributeError(name)
