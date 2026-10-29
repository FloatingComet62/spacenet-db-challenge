import cv2
import numpy as np


DIAMETER = 5
SIGMA_COLOR = 50
SIGMA_SPACE = 50


def process(image: np.ndarray) -> np.ndarray:
    return cv2.bilateralFilter(
        image,
        d=DIAMETER,
        sigmaColor=SIGMA_COLOR,
        sigmaSpace=SIGMA_SPACE,
    )
