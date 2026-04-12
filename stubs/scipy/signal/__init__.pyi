import numpy as np

def resample_poly(
    x: np.ndarray,
    up: int,
    down: int,
    axis: int = 0,
    window: object = ("kaiser", 5),
    padtype: str = "constant",
    cval: float | None = None,
) -> np.ndarray: ...
