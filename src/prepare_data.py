import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from dirs import *

from rasterio.features import rasterize
from shapely import wkt

NPZ_DIR.mkdir(exist_ok=True)

df = pd.read_csv(REFERENCE)
buildings = df[df["Object"] == "Building"].copy()

# Group polygons by their corresponding image tile.
grouped = buildings.groupby("ImageId")

for image_id, group in grouped:
    image_path = IMAGE_DIR / f"{image_id}.tif"

    if not image_path.exists():
        print(f"Missing image: {image_path}")
        continue

    with rasterio.open(image_path) as src:
        image = src.read([1, 2, 3])
        image = np.moveaxis(image, 0, -1)

        height, width = src.height, src.width

    polygons = []

    for geometry in group["Wkt_Pix"].dropna():
        polygon = wkt.loads(geometry)

        if polygon.is_empty or not polygon.is_valid:
            continue

    polygons.append(polygon)

    mask = rasterize(
        [(polygon, 1) for polygon in polygons],
        out_shape=(height, width),
        fill=0,
        dtype=np.uint8,
    )

    np.savez_compressed(
        NPZ_DIR / f"{image_id}.npz",
        image=image,
        mask=mask,
    )

    print(
        f"{image_id}: {len(polygons)} buildings, "
        f"{mask.sum()} building pixels"
    )

# Visualize one generated image-mask pair.
sample = next(NPZ_DIR.glob("*.npz"))

with np.load(sample) as data:
    image = data["image"]
    mask = data["mask"]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(image)
axes[0].set_title("Satellite image")

axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
axes[1].set_title("Building mask")

axes[2].imshow(image)
axes[2].imshow(mask, cmap="autumn", alpha=0.5)
axes[2].set_title("Overlay")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()
