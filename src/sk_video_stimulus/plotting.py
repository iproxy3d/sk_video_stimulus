#!/usr/bin/env python3
"""Plot video-feature CSV files and inspect whether signals oscillate.

Examples:
    python plot_video_features.py video_features.csv --output-dir plots
    python plot_video_features.py normalized_features.csv --output-dir plots
    python plot_video_features.py video_features.csv normalized_features.csv \
        --output-dir plots --start 5 --end 30

For each CSV the script writes:

* ``<stem>_raw_timeseries.png`` -- x_t and raw window features;
* ``<stem>_normalized_timeseries.png`` -- calibrated features, when present;
* ``<stem>_oscillation_diagnostics.png`` -- comparable z-scored signals,
  autocorrelation and normalized power spectra.

The diagnostics do not decide that a signal is an oscillation merely because
it moves. A repeated waveform should normally produce secondary peaks in the
autocorrelation and a non-zero-frequency peak in the spectrum.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


RAW_COLUMNS = ("x_t", "activity", "anxiety", "tempo", "deviation")
NORMALIZED_COLUMNS = (
    "s_activity",
    "s_anxiety",
    "s_tempo",
    "normalized_deviation",
)

LABELS = {
    "x_t": "x_t (video stimulus)",
    "activity": "activity",
    "anxiety": "anxiety",
    "tempo": "tempo, events/s",
    "deviation": "deviation",
    "s_activity": "s_activity",
    "s_anxiety": "s_anxiety",
    "s_tempo": "s_tempo",
    "normalized_deviation": "normalized_deviation",
}


@dataclass(frozen=True)
class FeatureTable:
    path: Path
    timestamp: np.ndarray
    ready: np.ndarray
    columns: dict[str, np.ndarray]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, nargs="+", help="feature CSV file(s)")
    parser.add_argument("--output-dir", type=Path, default=Path("video_feature_plots"))
    parser.add_argument("--start", type=float, default=None, help="first second to plot")
    parser.add_argument("--end", type=float, default=None, help="last second to plot")
    parser.add_argument(
        "--acf-seconds",
        type=float,
        default=10.0,
        help="maximum lag shown in autocorrelation",
    )
    parser.add_argument(
        "--max-frequency",
        type=float,
        default=5.0,
        help="maximum frequency shown in the spectrum; 0 means Nyquist",
    )
    parser.add_argument(
        "--max-plot-points",
        type=int,
        default=30000,
        help="maximum displayed points per line; analysis still uses all rows",
    )
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--show", action="store_true", help="also open plot windows")
    return parser.parse_args()


def _parse_float(value: str | None) -> float:
    if value is None or not value.strip():
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def _parse_bool(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _detect_dialect(path: Path) -> csv.Dialect:
    sample = path.read_text(encoding="utf-8-sig", errors="replace")[:8192]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t")
    except csv.Error:
        return csv.excel


def load_feature_csv(path: Path) -> FeatureTable:
    if not path.is_file():
        raise FileNotFoundError(path)

    dialect = _detect_dialect(path)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, dialect=dialect)
        fieldnames = tuple(reader.fieldnames or ())
        if "timestamp" not in fieldnames:
            raise ValueError(f"{path}: missing required column 'timestamp'")

        available = [name for name in (*RAW_COLUMNS, *NORMALIZED_COLUMNS) if name in fieldnames]
        if not available:
            raise ValueError(f"{path}: no supported video-feature columns found")

        times: list[float] = []
        ready: list[bool] = []
        values: dict[str, list[float]] = {name: [] for name in available}
        for row in reader:
            timestamp = _parse_float(row.get("timestamp"))
            if not math.isfinite(timestamp):
                continue
            times.append(timestamp)
            ready.append(_parse_bool(row.get("ready", "true")))
            for name in available:
                values[name].append(_parse_float(row.get(name)))

    if not times:
        raise ValueError(f"{path}: no rows with a numeric timestamp")

    timestamp_array = np.asarray(times, dtype=np.float64)
    if np.any(np.diff(timestamp_array) <= 0):
        raise ValueError(f"{path}: timestamps must be strictly increasing")

    return FeatureTable(
        path=path,
        timestamp=timestamp_array,
        ready=np.asarray(ready, dtype=bool),
        columns={name: np.asarray(items, dtype=np.float64) for name, items in values.items()},
    )


def select_time_range(
    table: FeatureTable,
    start: float | None,
    end: float | None,
) -> FeatureTable:
    if start is not None and end is not None and end <= start:
        raise ValueError("--end must be greater than --start")
    mask = np.ones(len(table.timestamp), dtype=bool)
    if start is not None:
        mask &= table.timestamp >= start
    if end is not None:
        mask &= table.timestamp <= end
    if not np.any(mask):
        raise ValueError(f"{table.path}: selected time range contains no rows")
    return FeatureTable(
        path=table.path,
        timestamp=table.timestamp[mask],
        ready=table.ready[mask],
        columns={name: values[mask] for name, values in table.columns.items()},
    )


def _plot_indices(length: int, maximum: int) -> np.ndarray:
    if maximum <= 0 or length <= maximum:
        return np.arange(length)
    return np.linspace(0, length - 1, maximum, dtype=np.int64)


def _first_ready_time(table: FeatureTable) -> float | None:
    positions = np.flatnonzero(table.ready)
    if len(positions) == 0 or positions[0] == 0:
        return None
    return float(table.timestamp[positions[0]])


def _plot_timeseries(
    table: FeatureTable,
    columns: Sequence[str],
    output: Path,
    title: str,
    dpi: int,
    max_plot_points: int,
) -> bool:
    import matplotlib.pyplot as plt

    present = [name for name in columns if name in table.columns]
    if not present:
        return False

    indices = _plot_indices(len(table.timestamp), max_plot_points)
    figure, axes = plt.subplots(
        len(present),
        1,
        figsize=(14, max(3.2, 2.45 * len(present))),
        sharex=True,
        constrained_layout=True,
    )
    if len(present) == 1:
        axes = [axes]

    first_ready = _first_ready_time(table)
    for axis, name in zip(axes, present):
        axis.plot(
            table.timestamp[indices],
            table.columns[name][indices],
            linewidth=1.0,
            color="#2458A6",
        )
        axis.set_ylabel(LABELS.get(name, name))
        axis.grid(True, alpha=0.25)
        if name.startswith("s_") or name == "normalized_deviation" or name == "activity":
            axis.axhline(0.0, color="#333333", linewidth=0.7, alpha=0.55)
        if first_ready is not None:
            axis.axvspan(
                float(table.timestamp[0]),
                first_ready,
                color="#D9D9D9",
                alpha=0.35,
                label="warm-up",
            )

    axes[0].set_title(title)
    axes[-1].set_xlabel("time, s")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi)
    return True


def _finite_interpolated(values: np.ndarray) -> np.ndarray | None:
    finite = np.isfinite(values)
    if int(finite.sum()) < 3:
        return None
    if finite.all():
        return values.astype(np.float64, copy=True)
    positions = np.arange(len(values), dtype=np.float64)
    return np.interp(positions, positions[finite], values[finite])


def _zscore(values: np.ndarray) -> np.ndarray | None:
    clean = _finite_interpolated(values)
    if clean is None:
        return None
    standard_deviation = float(clean.std())
    if standard_deviation <= 1e-12:
        return None
    return (clean - float(clean.mean())) / standard_deviation


def _sampling_frequency(timestamp: np.ndarray) -> float:
    intervals = np.diff(timestamp)
    median_interval = float(np.median(intervals))
    if not math.isfinite(median_interval) or median_interval <= 0:
        raise ValueError("cannot estimate sampling frequency from timestamps")
    return 1.0 / median_interval


def _autocorrelation(values: np.ndarray, maximum_lag: int) -> np.ndarray:
    centered = values - float(values.mean())
    size = len(centered)
    fft_size = 1 << (2 * size - 1).bit_length()
    spectrum = np.fft.rfft(centered, n=fft_size)
    correlation = np.fft.irfft(spectrum * np.conjugate(spectrum), n=fft_size)[:size]
    if correlation[0] <= 1e-15:
        return np.ones(1, dtype=np.float64)
    correlation /= correlation[0]
    return correlation[: min(maximum_lag + 1, size)]


def _power_spectrum(values: np.ndarray, sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
    centered = values - float(values.mean())
    window = np.hanning(len(centered))
    spectrum = np.fft.rfft(centered * window)
    power = np.abs(spectrum) ** 2
    frequencies = np.fft.rfftfreq(len(centered), d=1.0 / sample_rate)
    if len(power):
        power[0] = 0.0
    maximum = float(power.max()) if len(power) else 0.0
    if maximum > 0:
        power /= maximum
    return frequencies, power


def _dominant_frequency(
    frequencies: np.ndarray,
    power: np.ndarray,
    duration: float,
) -> float | None:
    minimum = max(0.05, 1.0 / max(duration, 1e-9))
    valid = (frequencies >= minimum) & np.isfinite(power)
    if not np.any(valid) or float(power[valid].max()) <= 0:
        return None
    positions = np.flatnonzero(valid)
    return float(frequencies[positions[int(np.argmax(power[valid]))]])


def _diagnostic_columns(table: FeatureTable) -> list[str]:
    normalized = [name for name in NORMALIZED_COLUMNS if name in table.columns]
    if normalized:
        return (["x_t"] if "x_t" in table.columns else []) + normalized
    return [name for name in RAW_COLUMNS if name in table.columns]


def _plot_oscillation_diagnostics(
    table: FeatureTable,
    output: Path,
    dpi: int,
    acf_seconds: float,
    max_frequency: float,
    max_plot_points: int,
) -> dict[str, float | None]:
    import matplotlib.pyplot as plt

    if acf_seconds <= 0:
        raise ValueError("--acf-seconds must be positive")

    # Warm-up values are placeholders for the window features and would add a
    # false step/flat segment to their spectra. Keep them in the time-series
    # plots for transparency, but exclude them from oscillation diagnostics.
    analysis_mask = table.ready if int(table.ready.sum()) >= 3 else np.ones(len(table.ready), dtype=bool)
    analysis_time = table.timestamp[analysis_mask]
    sample_rate = _sampling_frequency(analysis_time)
    duration = float(analysis_time[-1] - analysis_time[0])
    columns = _diagnostic_columns(table)
    signals: dict[str, np.ndarray] = {}
    for name in columns:
        normalized = _zscore(table.columns[name][analysis_mask])
        if normalized is not None:
            signals[name] = normalized

    if not signals:
        raise ValueError(f"{table.path}: all diagnostic signals are constant or empty")

    figure, axes = plt.subplots(3, 1, figsize=(14, 11), constrained_layout=True)
    colors = plt.get_cmap("tab10")
    indices = _plot_indices(len(analysis_time), max_plot_points)
    dominant: dict[str, float | None] = {}

    for color_index, (name, signal) in enumerate(signals.items()):
        color = colors(color_index % 10)
        axes[0].plot(
            analysis_time[indices],
            signal[indices],
            linewidth=0.9,
            alpha=0.82,
            label=LABELS.get(name, name),
            color=color,
        )

        maximum_lag = max(1, int(round(acf_seconds * sample_rate)))
        correlation = _autocorrelation(signal, maximum_lag)
        lags = np.arange(len(correlation), dtype=np.float64) / sample_rate
        axes[1].plot(lags, correlation, linewidth=1.0, label=LABELS.get(name, name), color=color)

        frequencies, power = _power_spectrum(signal, sample_rate)
        frequency_limit = sample_rate / 2.0 if max_frequency <= 0 else min(max_frequency, sample_rate / 2.0)
        frequency_mask = frequencies <= frequency_limit
        axes[2].plot(
            frequencies[frequency_mask],
            power[frequency_mask],
            linewidth=1.0,
            label=LABELS.get(name, name),
            color=color,
        )
        dominant[name] = _dominant_frequency(frequencies[frequency_mask], power[frequency_mask], duration)

    axes[0].set_title(f"Oscillation diagnostics: {table.path.name}")
    axes[0].set_ylabel("standardized value, z")
    axes[0].set_xlabel("time, s")
    axes[1].set_ylabel("autocorrelation")
    axes[1].set_xlabel("lag, s")
    axes[1].axhline(0.0, color="#333333", linewidth=0.7, alpha=0.55)
    axes[1].set_ylim(-1.05, 1.05)
    axes[2].set_ylabel("relative power")
    axes[2].set_xlabel("frequency, Hz")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(loc="upper right", ncol=2, fontsize=8)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi)
    return dominant


def _unique_output_stem(path: Path, used: set[str]) -> str:
    candidate = path.stem
    if candidate not in used:
        used.add(candidate)
        return candidate
    counter = 2
    while f"{candidate}_{counter}" in used:
        counter += 1
    result = f"{candidate}_{counter}"
    used.add(result)
    return result


def process_csv(path: Path, args: argparse.Namespace, stem: str) -> list[Path]:
    import matplotlib.pyplot as plt

    table = select_time_range(load_feature_csv(path), args.start, args.end)
    outputs: list[Path] = []

    raw_output = args.output_dir / f"{stem}_raw_timeseries.png"
    if _plot_timeseries(
        table,
        RAW_COLUMNS,
        raw_output,
        f"Raw video features: {path.name}",
        args.dpi,
        args.max_plot_points,
    ):
        outputs.append(raw_output)

    normalized_output = args.output_dir / f"{stem}_normalized_timeseries.png"
    if _plot_timeseries(
        table,
        NORMALIZED_COLUMNS,
        normalized_output,
        f"Normalized video features: {path.name}",
        args.dpi,
        args.max_plot_points,
    ):
        outputs.append(normalized_output)

    diagnostics_output = args.output_dir / f"{stem}_oscillation_diagnostics.png"
    dominant = _plot_oscillation_diagnostics(
        table,
        diagnostics_output,
        args.dpi,
        args.acf_seconds,
        args.max_frequency,
        args.max_plot_points,
    )
    outputs.append(diagnostics_output)

    print(f"{path}:")
    print(f"  rows: {len(table.timestamp)}")
    print(f"  duration: {table.timestamp[-1] - table.timestamp[0]:.3f} s")
    for name, frequency in dominant.items():
        if frequency is None:
            print(f"  {name}: no non-zero dominant frequency")
        else:
            print(
                f"  {name}: strongest spectral frequency {frequency:.4g} Hz "
                f"(period {1.0 / frequency:.4g} s)"
            )
    for output in outputs:
        print(f"  plot: {output.resolve()}")

    if not args.show:
        plt.close("all")
    return outputs


def main() -> int:
    args = _parse_args()
    if args.dpi <= 0:
        raise SystemExit("--dpi must be positive")
    if args.max_plot_points < 0:
        raise SystemExit("--max-plot-points must be non-negative")

    # Select a non-interactive backend before importing pyplot in helpers.
    if not args.show:
        import matplotlib

        matplotlib.use("Agg")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    used_stems: set[str] = set()
    for path in args.csv:
        stem = _unique_output_stem(path, used_stems)
        process_csv(path, args, stem)

    if args.show:
        import matplotlib.pyplot as plt

        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
