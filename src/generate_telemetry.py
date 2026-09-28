#!/usr/bin/env python3
"""
Synthetic Flight Strain Gauge Telemetry Generator.

Produces multi-channel time-series sensor data representing three flight
regimes for airframe structural testing:

    1. normal_flight   : Steady cruise within structural design envelope
    2. high_turbulence : Elevated strain and acceleration from gust loading
    3. over_g_maneuver : Aggressive pull up exceeding positive load factor

Outputs:
    output/flight_telemetry.json
    output/flight_telemetry.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Default output directory relative to repository root
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

# Scenario durations (seconds) and sample rate (Hz)
SAMPLE_RATE_HZ = 10.0
SCENARIO_DURATION_S = {
    "normal_flight": 30.0,
    "high_turbulence": 20.0,
    "over_g_maneuver": 15.0,
}


def _generate_normal_flight(t: np.ndarray, rng: np.random.Generator) -> dict:
    """
    Steady cruise telemetry within structural design envelope.

    Strain ~ 800 ue, acceleration ~ 1.0 g, skin temp ~ 25 C.
    """
    n = len(t)
    return {
        "wing_root_strain_ue": 800.0 + 50.0 * np.sin(0.3 * t) + rng.normal(0, 15, n),
        "fuselage_accel_g": 1.0 + 0.05 * np.sin(0.5 * t) + rng.normal(0, 0.02, n),
        "skin_temp_c": 25.0 + 2.0 * np.sin(0.1 * t) + rng.normal(0, 0.5, n),
        "scenario": ["normal_flight"] * n,
    }


def _generate_high_turbulence(t: np.ndarray, rng: np.random.Generator) -> dict:
    """
    Turbulent gust loading with elevated strain and acceleration spikes.

    Strain peaks near 2800 ue (approaching yield), acceleration up to ~4 g,
    skin temp rises modestly from kinetic heating.
    """
    n = len(t)
    # Gust envelope: amplitude grows then decays mid scenario
    gust_envelope = np.sin(np.pi * np.linspace(0, 1, n))
    return {
        "wing_root_strain_ue": (
            1500.0
            + 1200.0 * gust_envelope * np.sin(2.0 * t)
            + rng.normal(0, 40, n)
        ),
        "fuselage_accel_g": (
            1.5
            + 2.5 * gust_envelope * np.abs(np.sin(3.0 * t))
            + rng.normal(0, 0.1, n)
        ),
        "skin_temp_c": 35.0 + 5.0 * gust_envelope + rng.normal(0, 1.0, n),
        "scenario": ["high_turbulence"] * n,
    }


def _generate_over_g_maneuver(t: np.ndarray, rng: np.random.Generator) -> dict:
    """
    Aggressive pull up maneuver exceeding positive load factor limits.

    Strain exceeds yield (~3800+ ue), acceleration peaks above 6 g,
    skin temperature elevated. Intentionally produces negative margins
    of safety for verifier validation.
    """
    n = len(t)
    # Maneuver profile: ramp up, hold peak, ramp down
    phase = np.linspace(0, 1, n)
    pull_up = np.where(
        phase < 0.3,
        phase / 0.3,
        np.where(phase < 0.7, 1.0, (1.0 - phase) / 0.3),
    )
    return {
        "wing_root_strain_ue": (
            2000.0 + 2200.0 * pull_up + rng.normal(0, 50, n)
        ),
        "fuselage_accel_g": (
            1.0 + 6.5 * pull_up + rng.normal(0, 0.15, n)
        ),
        "skin_temp_c": 45.0 + 30.0 * pull_up + rng.normal(0, 2.0, n),
        "scenario": ["over_g_maneuver"] * n,
    }


SCENARIO_GENERATORS = {
    "normal_flight": _generate_normal_flight,
    "high_turbulence": _generate_high_turbulence,
    "over_g_maneuver": _generate_over_g_maneuver,
}


def generate_scenario(
    scenario: str,
    duration_s: float,
    sample_rate_hz: float,
    time_offset_s: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Generate telemetry samples for a single flight scenario.

    Args:
        scenario: Scenario key (normal_flight, high_turbulence, over_g_maneuver).
        duration_s: Scenario duration in seconds.
        sample_rate_hz: Sampling frequency in Hz.
        time_offset_s: Absolute start time for this scenario segment.
        rng: NumPy random generator for reproducible noise.

    Returns:
        DataFrame of telemetry samples for the scenario.
    """
    if scenario not in SCENARIO_GENERATORS:
        raise ValueError(
            f"Unknown scenario '{scenario}'. "
            f"Choose from: {list(SCENARIO_GENERATORS)}"
        )

    n_samples = int(duration_s * sample_rate_hz)
    t_local = np.arange(n_samples) / sample_rate_hz
    t_abs = t_local + time_offset_s

    channels = SCENARIO_GENERATORS[scenario](t_local, rng)
    data = {"timestamp_s": t_abs, **channels}
    return pd.DataFrame(data)


def generate_telemetry(
    scenarios: list[str] | None = None,
    sample_rate_hz: float = SAMPLE_RATE_HZ,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate concatenated multi scenario flight telemetry.

    Args:
        scenarios: Ordered list of scenario names. Defaults to all three.
        sample_rate_hz: Sampling frequency in Hz.
        seed: Random seed for deterministic synthetic data.

    Returns:
        Combined DataFrame sorted by timestamp_s.
    """
    if scenarios is None:
        scenarios = list(SCENARIO_DURATION_S.keys())

    rng = np.random.default_rng(seed)
    frames: list[pd.DataFrame] = []
    time_cursor = 0.0

    for scenario in scenarios:
        duration = SCENARIO_DURATION_S.get(scenario, 20.0)
        frame = generate_scenario(
            scenario=scenario,
            duration_s=duration,
            sample_rate_hz=sample_rate_hz,
            time_offset_s=time_cursor,
            rng=rng,
        )
        frames.append(frame)
        time_cursor += duration

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("timestamp_s").reset_index(drop=True)

    # Round numeric columns for clean serialization
    for col in ("timestamp_s", "wing_root_strain_ue", "fuselage_accel_g", "skin_temp_c"):
        combined[col] = combined[col].round(4)

    return combined


def write_outputs(df: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    """
    Write telemetry DataFrame to JSON and CSV under output_dir.

    Args:
        df: Telemetry DataFrame.
        output_dir: Destination directory (created if absent).

    Returns:
        Tuple of (json_path, csv_path).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "flight_telemetry.json"
    csv_path = output_dir / "flight_telemetry.csv"

    records = df.to_dict(orient="records")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2)

    df.to_csv(csv_path, index=False)

    return json_path, csv_path


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for synthetic telemetry generation."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic flight strain gauge telemetry for "
            "aircraft structural stress verification."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for flight_telemetry.json and .csv (default: output/)",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=list(SCENARIO_DURATION_S.keys()),
        default=None,
        help="Scenarios to generate (default: all)",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=SAMPLE_RATE_HZ,
        help=f"Sample rate in Hz (default: {SAMPLE_RATE_HZ})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible output (default: 42)",
    )
    args = parser.parse_args(argv)

    df = generate_telemetry(
        scenarios=args.scenarios,
        sample_rate_hz=args.sample_rate,
        seed=args.seed,
    )
    json_path, csv_path = write_outputs(df, args.output_dir)

    print(f"Generated {len(df)} telemetry samples")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")
    scenario_counts = df["scenario"].value_counts().to_dict()
    for name, count in scenario_counts.items():
        print(f"  Scenario '{name}': {count} samples")

    return 0


if __name__ == "__main__":
    sys.exit(main())
