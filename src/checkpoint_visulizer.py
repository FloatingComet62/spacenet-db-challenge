from pathlib import Path
import subprocess
import sys
import questionary

checkpoint_dir = Path("checkpoints")
visualizer = Path("src/visualize.py")

checkpoints = sorted(
    checkpoint_dir.glob("*.pt"),
    key=lambda p: (
        p.name != "best_unet_resnet34.pt",
        p.name.lower(),
    ),
)

if not checkpoints:
    print(f"No checkpoints found in {checkpoint_dir}/")
    sys.exit(1)

selected = questionary.select(
    "Choose checkpoint",
    choices=[
        questionary.Choice(
            title=path.name,
            value=path,
        )
        for path in checkpoints
    ],
).ask()

if selected is not None:
    subprocess.run(
        [
            sys.executable,
            str(visualizer),
            "--checkpoint",
            str(selected.resolve()),
        ],
        check=False,
    )
