import cv2
import numpy as np

def process(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    luminance, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )
    luminance = clahe.apply(luminance)

    enhanced = cv2.merge((luminance, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
