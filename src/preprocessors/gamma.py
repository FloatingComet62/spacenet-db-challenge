import numpy as np

GAMMA = 0.8

def process(image: np.ndarray) -> np.ndarray:
    lookup = np.array([
        ((i / 255.0) ** GAMMA) * 255
        for i in range(256)
    ]).clip(0, 255).astype(np.uint8)

    return lookup[image]
