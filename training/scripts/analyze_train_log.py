"""
Analyze a pretrain.py training log to make a continue-vs-restart decision.

Reports:
    * Per-window median / 25th-percentile loss
    * Trend (slope of last N steps)
    * Verdict: HEALTHY / BORDERLINE / RESTART

Usage::

    uv run python training/scripts/analyze_train_log.py --log /tmp/shogi_40m_train.log
"""

from __future__ import annotations

import re

import click


_LINE_RE = re.compile(
    r"step\s+(\d+)\s+\|\s+loss\s+([0-9.]+)\s+\|\s+lr\s+([0-9.eE+-]+)"
)


@click.command()
@click.option("--log", "log_path", type=str, required=True)
@click.option("--window", type=int, default=500, show_default=True,
              help="Window size for trailing statistics.")
@click.option("--healthy-median", type=float, default=5.5, show_default=True,
              help="Median loss below this = HEALTHY.")
@click.option("--restart-median", type=float, default=6.5, show_default=True,
              help="Median loss above this = RESTART.")
def main(log_path: str, window: int, healthy_median: float, restart_median: float) -> None:
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

    if len(steps) < 10:
        click.echo(f"INSUFFICIENT_DATA: only {len(steps)} step lines.")
        raise SystemExit(2)

    steps_a = np.asarray(steps)
    losses_a = np.asarray(losses)
    lrs_a = np.asarray(lrs)

    # Overall stats
    click.echo(f"Log: {log_path}")
    click.echo(f"Total log entries  : {len(steps_a)}")
    click.echo(f"Step range         : {steps_a[0]} → {steps_a[-1]}")
    click.echo(f"Current LR         : {lrs_a[-1]:.2e}")
    click.echo("")

    # Print rolling window stats every 1000 steps
    click.echo(f"{'window_end':>10}  {'lr':>9}  {'min':>6}  {'p25':>6}  {'median':>6}  {'p75':>6}  {'max':>6}")
    for end in range(1000, steps_a[-1] + 1, 1000):
        mask = (steps_a > end - window) & (steps_a <= end)
        if mask.sum() == 0:
            continue
        ls = losses_a[mask]
        lr_at_end = lrs_a[mask][-1] if mask.sum() > 0 else 0
        click.echo(
            f"{end:>10}  {lr_at_end:.2e}  "
            f"{ls.min():.3f}  {np.percentile(ls,25):.3f}  "
            f"{np.median(ls):.3f}  {np.percentile(ls,75):.3f}  {ls.max():.3f}"
        )
    click.echo("")

    # Verdict on last `window` steps
    last = losses_a[-window:] if len(losses_a) >= window else losses_a
    last_n = len(last)
    median = float(np.median(last))
    p25 = float(np.percentile(last, 25))
    minv = float(last.min())

    # Trend = slope of linear fit on last_n
    x = np.arange(last_n)
    slope = float(np.polyfit(x, last, 1)[0])

    click.echo(f"Trailing window ({last_n} entries):")
    click.echo(f"  median loss : {median:.3f}")
    click.echo(f"  p25 loss    : {p25:.3f}")
    click.echo(f"  min loss    : {minv:.3f}")
    click.echo(f"  slope       : {slope:+.5f} per step  ({'descending' if slope<0 else 'ascending'})")
    click.echo("")

    # Verdict
    if median < healthy_median and slope < 0:
        verdict = "HEALTHY"
    elif median > restart_median:
        verdict = "RESTART"
    elif slope < -1e-4:
        verdict = "HEALTHY"  # still descending fast
    elif slope > 1e-4:
        verdict = "RESTART"  # ascending substantially
    else:
        verdict = "BORDERLINE"

    click.echo(f"VERDICT: {verdict}")

    # Suggested action
    if verdict == "HEALTHY":
        click.echo("  → continue training")
    elif verdict == "RESTART":
        click.echo("  → kill and restart with WSD schedule + lower peak LR")
    else:
        click.echo("  → wait another 1-2 hours then re-evaluate")

    raise SystemExit(0 if verdict == "HEALTHY" else (1 if verdict == "BORDERLINE" else 2))


if __name__ == "__main__":
    main()
