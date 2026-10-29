#!/usr/bin/env python3

import rasterio
import pandas as pd

df = pd.read_csv("data/Germany_Training_Public_reference.csv")
buildings = df[df["Object"] == "Building"]

image_id = buildings.iloc[0]["ImageId"]
image_path = f"data/PRE-event/{image_id}.tif"

with rasterio.open(image_path) as src:
    print("Image:", image_id)
    print("Dimensions:", src.width, src.height)
    print("Bands:", src.count)
    print("CRS:", src.crs)

print("\nSample polygon:")
print(buildings.iloc[0]["Wkt_Pix"])
