import cv2
import rasterio
from rasterio.windows import Window
from pathlib import Path
import numpy as np

tif = "AOI_8_Mumbai_PS-RGB_COG.tif"
tile_size = 2048

Path("tiles").mkdir(exist_ok=True)

with rasterio.open(tif) as src:
    for y in range(0, src.height, tile_size):
        for x in range(0, src.width, tile_size):
            width = min(tile_size, src.width - x)
            height = min(tile_size, src.height - y)
            window = Window(x, y, width, height)
            tile = src.read(window=window)
            tile = np.transpose(tile, (1, 2, 0))
            tile = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
            filename = f"tiles/tile_{x}_{y}.png"
            cv2.imwrite(filename, tile)
