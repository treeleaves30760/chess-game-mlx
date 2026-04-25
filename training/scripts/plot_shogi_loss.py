"""
Plot training loss / LR curves from a pretrain.py run log.

Usage::

    uv run python training/scripts/plot_shogi_loss.py \\
        --log /tmp/shogi_40m_train.log \\
        --output /tmp/shogi_40m_loss.png
"""

from __future__ import annotations

import re

import click


_LINE_RE = re.compile(
    r"step\s+(\d+)\s+\|\s+loss\s+([0-9.]+)\s+\|\s+lr\s+([0-9.eE+-]+)"
)


@click.command()
@click.option("--log", "log_path", type=str, required=True)
@click.option("--output", type=str, required=True)
@click.option("--smooth", type=int, default=20, show_default=True,
              help="Rolling-window size for the smoothed loss curve.")
def main(log_path: str, output: str, smooth: int) -> None:
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    steps: list[int] = []
    losses: list[float] = []
    lrs: list[float] = []

    with open(log_path) as f:
        for line in f:
            m = _LINE_RE.search(line)
            if m:
                steps.append(int(m.group(1)))
                losses.append(float(m.group(2)))
                lrs.append(float(m.group(3)))

    if not steps:
        click.echo("No training-step lines found in log.", err=True)
        raise SystemExit(1)

    steps_a = np.asarray(steps)
    losses_a = np.asarray(losses)
    lrs_a = np.asarray(lrs)

    if smooth > 1 and len(losses_a) >= smooth:
        kernel = np.ones(smooth) / smooth
        smoothed = np.convolve(losses_a, kernel, mode="valid")
        smooth_steps = steps_a[smooth - 1 :]
    else:
        smoothed = losses_a
        smooth_steps = steps_a

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1.plot(steps_a, losses_a, alpha=0.3, label="raw")
    ax1.plot(smooth_steps, smoothed, color="C1", label=f"rolling {smooth}")
    ax1.axhline(7.69, color="grey", linestyle=":", label="uniform-2187 baseline")
    ax1.set_ylabel("Total loss")
    ax1.set_title(f"Training loss — {log_path}")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(steps_a, lrs_a, color="C2")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Learning rate")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output, dpi=120)
    click.echo(f"Wrote {output}  (n={len(steps_a)} log entries)")


if __name__ == "__main__":
    main()
