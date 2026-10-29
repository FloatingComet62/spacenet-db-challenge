import cv2
import numpy as np

SIGMA = 1.0
STRENGTH = 1.0

def process(image: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(
        image,
        ksize=(0, 0),
        sigmaX=SIGMA,
    )

    sharpened = cv2.addWeighted(
        image,
        1.0 + STRENGTH,
        blurred,
        -STRENGTH,
        0,
    )

    return sharpened
