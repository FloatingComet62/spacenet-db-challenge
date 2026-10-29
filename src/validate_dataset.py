from pathlib import Path
import numpy as np

PATCH_SIZE = 512
files = sorted(Path("data/building_dataset").glob("*.npz"))

total_patches = 0
positive_patches = 0
positive_pixels = 0

total_pixels = 0
total_positive = 0
empty_masks = 0
positive_tiles = 0

for path in files:
    with np.load(path) as data:
        mask = data["mask"]

    positive = np.count_nonzero(mask)

    total_pixels += mask.size
    total_positive += positive

    if positive == 0:
        empty_masks += 1
    else:
        positive_tiles += 1

    h, w = mask.shape

    ys = sorted(set(list(range(0, h - PATCH_SIZE + 1, PATCH_SIZE))
                    + [h - PATCH_SIZE]))
    xs = sorted(set(list(range(0, w - PATCH_SIZE + 1, PATCH_SIZE))
                    + [w - PATCH_SIZE]))

    for y in ys:
        for x in xs:
            patch = mask[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
            count = np.count_nonzero(patch)

            total_patches += 1
            positive_pixels += count

            if count:
                positive_patches += 1

print(f"Total patches: {total_patches}")
print(f"Patches containing buildings: {positive_patches}")
print(f"Empty patches: {total_patches - positive_patches}")
print(f"Positive patch percentage: {100 * positive_patches / total_patches:.2f}%")
print(f"Total building pixels in patches: {positive_pixels}")
print(f"Total tiles: {len(files)}")
print(f"Tiles with buildings: {positive_tiles}")
print(f"Empty masks: {empty_masks}")
print(f"Total building pixels: {total_positive:,}")
print(f"Total pixels: {total_pixels:,}")
print(f"Building pixel percentage: {100 * total_positive / total_pixels:.4f}%")
