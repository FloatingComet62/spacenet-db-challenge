#!/usr/bin/env python3

from datetime import datetime
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import questionary


ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data" / "building_dataset"
CHECKPOINT_DIR = ROOT / "checkpoints"
LOG_DIR = ROOT / "logs"

PREPROCESSORS = [
    ("noop", 0),
    ("bilateral", 1),
    ("clahe", 2),
    ("gamma", 3),
    ("sharpen", 4),
    ("chainA", 5),
    ("chainB", 6),
    ("chainC", 7),
]


class Tee:
    """Write output to multiple streams simultaneously."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(
            getattr(stream, "isatty", lambda: False)()
            for stream in self.streams
        )

    def fileno(self):
        for stream in self.streams:
            if hasattr(stream, "fileno"):
                try:
                    return stream.fileno()
                except (OSError, ValueError):
                    pass
        raise OSError("No valid file descriptor available")

    def writable(self):
        return True

    def readable(self):
        return False

    def seekable(self):
        return False

    @property
    def encoding(self):
        return getattr(self.streams[0], "encoding", "utf-8")

    @property
    def errors(self):
        return getattr(self.streams[0], "errors", "strict")

    
def run_command(args):
    """Stream subprocess output to both terminal and log."""
    print(f"\n$ {' '.join(map(str, args))}", flush=True)

    process = subprocess.Popen(
        args,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    try:
        assert process.stdout is not None

        for line in process.stdout:
            print(line, end="", flush=True)

        return_code = process.wait()

    except BaseException:
        process.kill()
        process.wait()
        raise

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, args)

def restore_dataset(original_dataset):
    """Restore the dataset to its original, unprocessed state."""
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)

    shutil.copytree(original_dataset, DATA_DIR)
    print("Restored original dataset.")


def run_experiments(selected):
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATA_DIR}"
        )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Keep an independent pristine copy for reliable restoration.
    with tempfile.TemporaryDirectory(prefix="building_dataset_") as temp:
        original_dataset = Path(temp) / "building_dataset"
        shutil.copytree(DATA_DIR, original_dataset)

        try:
            for name, index in selected:
                print(f"\n{'=' * 60}")
                print(f"Experiment: {name}")
                print(f"{'=' * 60}")

                try:
                    # Every experiment starts from the same original data.
                    restore_dataset(original_dataset)

                    run_command([
                        sys.executable,
                        "src/preprocessing.py",
                        str(index),
                    ])

                    # Remove any old checkpoint to avoid copying stale results.
                    best_checkpoint = (
                        CHECKPOINT_DIR / "best_unet_resnet34.pt"
                    )
                    if best_checkpoint.exists():
                        best_checkpoint.unlink()

                    run_command([
                        sys.executable,
                        "src/train.py",
                    ])

                    if not best_checkpoint.is_file():
                        raise FileNotFoundError(
                            "Training did not produce the expected checkpoint: "
                            f"{best_checkpoint}"
                        )

                    destination = CHECKPOINT_DIR / f"{name}.pt"
                    shutil.copy2(best_checkpoint, destination)

                    print(f"Saved checkpoint: {destination.relative_to(ROOT)}")

                except subprocess.CalledProcessError as exc:
                    print(
                        f"Experiment '{name}' failed with exit code "
                        f"{exc.returncode}."
                    )
                    print("Stopping remaining experiments.")
                    break

                except Exception as exc:
                    print(
                        f"Experiment '{name}' failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    print("Stopping remaining experiments.")
                    break

                finally:
                    # Restore even when preprocessing or training fails.
                    try:
                        restore_dataset(original_dataset)
                    except Exception as exc:
                        print(
                            "CRITICAL: Could not restore the original dataset: "
                            f"{type(exc).__name__}: {exc}"
                        )
                        raise
        finally:
                pass

    print("\nExperiment runner finished.")


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = LOG_DIR / f"{timestamp}.txt"

    with log_path.open(
        "w",
        encoding="utf-8",
        buffering=1,
    ) as log:
        try:
            print(f"Experiment log: {log_path.relative_to(ROOT)}")
            choices = [
                questionary.Choice(title=name, value=(name, index))
                for name, index in PREPROCESSORS
            ]

            selected = questionary.checkbox(
                "Choose preprocessing methods "
                "(Space to select, Enter to confirm):",
                choices=choices,
                validate=lambda selection: (
                    True if selection else "Select at least one method."
                ),
            ).ask()

            if selected is None:
                print("Cancelled.")
                return

            original_stdout = sys.stdout
            original_stderr = sys.stderr

            sys.stdout = Tee(original_stdout, log)
            sys.stderr = Tee(original_stderr, log)
            run_experiments(selected)

        except KeyboardInterrupt:
            print("\nInterrupted by user.")

        except Exception as exc:
            print(
                f"\nFatal error: {type(exc).__name__}: {exc}"
            )
            raise

        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    print(f"\nLog saved to: {log_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
