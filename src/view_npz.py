import numpy as np
import matplotlib.pyplot as plt
from dirs import *

samples = sorted(NPZ_DIR.glob("*.npz"))
index = 0

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
plt.subplots_adjust(bottom=0.12)


def show_sample():
    sample = samples[index]

    with np.load(sample) as data:
        image = data["image"]
        mask = data["mask"]

    for ax in axes:
        ax.clear()
        ax.axis("off")

    axes[0].imshow(image)
    axes[0].set_title("Satellite image")

    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Building mask")

    axes[2].imshow(image)
    axes[2].imshow(mask, cmap="autumn", alpha=0.5)
    axes[2].set_title("Overlay")

    fig.suptitle(
        f"{index + 1}/{len(samples)} — {sample.stem}",
        fontsize=12,
    )
    fig.canvas.draw_idle()


def on_key(event):
    global index

    if event.key == " ":
        index = (index + 1) % len(samples)
        show_sample()
    elif event.key == "backspace":
        index = (index - 1) % len(samples)
        show_sample()


fig.canvas.mpl_connect("key_press_event", on_key)

show_sample()
plt.show()
