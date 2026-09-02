import rasterio

tif = "AOI_8_Mumbai_PS-RGB_COG.tif"

with rasterio.open(tif) as src:
    print(src.height, src.width)
    print(src.count)
