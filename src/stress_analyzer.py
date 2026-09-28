"""
Aircraft Structural Stress Analyzer.

Parses multi-channel time-series sensor telemetry (strain, acceleration,
temperature) and evaluates readings against aeronautical structural limits.
Computes yield margins of safety and flags exceedance events for airframe
structural testing workflows.
"""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Aeronautical specification limits (defense aerospace structural testing)
# ---------------------------------------------------------------------------

class ChannelType(str, Enum):
    """Supported telemetry channel categories."""

    STRAIN = "strain"
    ACCELERATION = "acceleration"
    TEMPERATURE = "temperature"


class AeronauticalLimits(BaseModel):
    """
    Structural limit envelope derived from airframe design specifications.

    Limits map directly to the Specification Matrix documented in README.md.
    Units:
        strain       : microstrain (ue)
        acceleration : g (multiples of standard gravity)
        temperature  : degrees Celsius
        stress       : MPa (derived from strain via modulus)
    """

    # Wing root strain gauge limits
    wing_root_yield_strain_ue: float = Field(
        default=3500.0,
        description="Wing root yield strain limit in microstrain",
        gt=0,
    )
    wing_root_ultimate_strain_ue: float = Field(
        default=4500.0,
        description="Wing root ultimate strain limit in microstrain",
        gt=0,
    )

    # Fuselage acceleration bounds
    fuselage_accel_positive_g: float = Field(
        default=6.0,
        description="Maximum positive fuselage normal acceleration (g)",
        gt=0,
    )
    fuselage_accel_negative_g: float = Field(
        default=-3.0,
        description="Maximum negative fuselage normal acceleration (g)",
        lt=0,
    )

    # Skin thermal thresholds
    skin_thermal_max_c: float = Field(
        default=120.0,
        description="Maximum allowable skin temperature in Celsius",
    )
    skin_thermal_min_c: float = Field(
        default=-55.0,
        description="Minimum allowable skin temperature in Celsius",
    )
    thermal_expansion_limit_ue: float = Field(
        default=800.0,
        description="Thermal expansion strain contribution limit in microstrain",
        gt=0,
    )

    # Material properties for stress conversion
    youngs_modulus_mpa: float = Field(
        default=72000.0,
        description="Young's modulus for aluminum airframe alloy (MPa)",
        gt=0,
    )
    yield_stress_mpa: float = Field(
        default=252.0,
        description="Material yield stress in MPa (derived from yield strain)",
        gt=0,
    )

    @field_validator("wing_root_ultimate_strain_ue")
    @classmethod
    def ultimate_exceeds_yield(cls, value: float, info: Any) -> float:
        """Ensure ultimate strain is strictly above yield strain."""
        yield_strain = info.data.get("wing_root_yield_strain_ue")
        if yield_strain is not None and value <= yield_strain:
            raise ValueError(
                "Ultimate strain must exceed yield strain "
                f"(got ultimate={value}, yield={yield_strain})"
            )
        return value


class TelemetryRecord(BaseModel):
    """Single multi-channel telemetry sample at one timestamp."""

    timestamp_s: float = Field(..., description="Elapsed flight time in seconds", ge=0)
    wing_root_strain_ue: Optional[float] = Field(
        None, description="Wing root strain gauge reading in microstrain"
    )
    fuselage_accel_g: Optional[float] = Field(
        None, description="Fuselage normal acceleration in g"
    )
    skin_temp_c: Optional[float] = Field(
        None, description="Skin surface temperature in Celsius"
    )
    scenario: Optional[str] = Field(
        None, description="Flight scenario label (normal, turbulence, over_g)"
    )


class ExceedanceEvent(BaseModel):
    """Record of a structural limit exceedance detected during analysis."""

    timestamp_s: float
    channel: ChannelType
    measured_value: float
    limit_value: float
    margin_of_safety: Optional[float] = None
    severity: str = "WARNING"
    message: str = ""


class AnalysisResult(BaseModel):
    """Aggregated structural analysis output for a telemetry stream."""

    total_samples: int = 0
    exceedance_count: int = 0
    min_margin_of_safety: Optional[float] = None
    mean_margin_of_safety: Optional[float] = None
    events: list[ExceedanceEvent] = Field(default_factory=list)
    missing_channels: list[str] = Field(default_factory=list)
    passed: bool = True


# ---------------------------------------------------------------------------
# Core engineering calculations
# ---------------------------------------------------------------------------

def format_signed(value: float, precision: int = 2) -> str:
    """
    Format a numeric value for log output without ASCII hyphen characters.

    Negative values are rendered as ``minus X`` so warning strings remain
    dash free while preserving sign information.
    """
    if value < 0:
        return f"minus {abs(value):.{precision}f}"
    return f"{value:.{precision}f}"


def calculate_margin_of_safety(yield_limit: float, applied_stress: float) -> float:
    """
    Compute the yield margin of safety.

    Formula:
        Margin of Safety = (Yield Limit / Applied Stress) - 1

    A positive margin indicates the structure is operating below yield.
    A zero margin indicates the structure is at the yield limit.
    A negative margin indicates yield has been exceeded.

    Args:
        yield_limit: Allowable yield stress or strain (same units as applied).
        applied_stress: Measured applied stress or strain.

    Returns:
        Margin of safety as a dimensionless float.

    Raises:
        ValueError: If applied_stress is zero (division undefined) or
                    yield_limit is not positive.
    """
    if yield_limit <= 0:
        raise ValueError(
            f"Yield limit must be positive, received {yield_limit}"
        )
    if applied_stress == 0:
        # Zero applied load implies infinite margin; return a large sentinel
        return float("inf")
    # Use absolute applied value so compressive loads are handled correctly
    return (yield_limit / abs(applied_stress)) - 1.0


def strain_to_stress_mpa(strain_ue: float, youngs_modulus_mpa: float) -> float:
    """
    Convert microstrain to stress using Hooke's law.

    stress (MPa) = E (MPa) * strain (unitless)
                 = E * (strain_ue * 1e-6)

    Args:
        strain_ue: Strain reading in microstrain.
        youngs_modulus_mpa: Young's modulus in MPa.

    Returns:
        Equivalent stress in MPa.
    """
    return youngs_modulus_mpa * (strain_ue * 1.0e-6)


def thermal_expansion_strain_ue(
    delta_temp_c: float,
    coefficient: float = 23.0e-6,
) -> float:
    """
    Estimate thermal expansion strain contribution.

    Default coefficient is for aluminum alloy (23e-6 / C).

    Args:
        delta_temp_c: Temperature change from reference (typically 20 C).
        coefficient: Linear thermal expansion coefficient (1/C).

    Returns:
        Thermal strain contribution in microstrain.
    """
    return abs(delta_temp_c) * coefficient * 1.0e6


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

REQUIRED_CHANNELS = (
    "wing_root_strain_ue",
    "fuselage_accel_g",
    "skin_temp_c",
)


class StressAnalyzer:
    """
    Multi-channel structural stress and telemetry verification engine.

    Loads time-series sensor data, evaluates each sample against the
    aeronautical limit envelope, computes yield margins of safety, and
    emits structured exceedance events with aeronautical specification
    references.
    """

    REFERENCE_TEMP_C: float = 20.0

    def __init__(self, limits: Optional[AeronauticalLimits] = None) -> None:
        """
        Initialize the analyzer with structural limit specifications.

        Args:
            limits: Aeronautical limit envelope. Uses defaults if omitted.
        """
        self.limits = limits or AeronauticalLimits()
        self._configure_logging()

    @staticmethod
    def _configure_logging() -> None:
        """Attach a stream handler if the module logger has none."""
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

    # -- Data loading -------------------------------------------------------

    def load_telemetry(self, path: str | Path) -> pd.DataFrame:
        """
        Load multi-channel telemetry from JSON or CSV.

        Args:
            path: Path to flight_telemetry.json or flight_telemetry.csv.

        Returns:
            DataFrame with columns matching TelemetryRecord fields.

        Raises:
            FileNotFoundError: If the path does not exist.
            ValueError: If the file format is unsupported or empty.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Telemetry file not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".json":
            df = pd.read_json(path)
        elif suffix == ".csv":
            df = pd.read_csv(path)
        else:
            raise ValueError(
                f"Unsupported telemetry format '{suffix}'. "
                "Use .json or .csv."
            )

        if df.empty:
            raise ValueError(f"Telemetry file is empty: {path}")

        return df

    def detect_missing_channels(self, df: pd.DataFrame) -> list[str]:
        """
        Identify required telemetry channels absent from the DataFrame.

        A channel is considered missing if the column is absent or if
        every value in the column is null.

        Args:
            df: Telemetry DataFrame.

        Returns:
            List of missing channel names.
        """
        missing: list[str] = []
        for channel in REQUIRED_CHANNELS:
            if channel not in df.columns:
                missing.append(channel)
            elif df[channel].isna().all():
                missing.append(channel)
        return missing

    # -- Per-sample evaluation ----------------------------------------------

    def evaluate_sample(
        self,
        record: TelemetryRecord,
    ) -> list[ExceedanceEvent]:
        """
        Evaluate a single telemetry sample against all structural limits.

        Args:
            record: Validated telemetry record.

        Returns:
            List of ExceedanceEvent objects (empty if all limits satisfied).
        """
        events: list[ExceedanceEvent] = []
        limits = self.limits
        ts = record.timestamp_s

        # --- Wing root strain / yield margin -------------------------------
        if record.wing_root_strain_ue is not None:
            strain = record.wing_root_strain_ue
            applied_stress = strain_to_stress_mpa(
                abs(strain), limits.youngs_modulus_mpa
            )
            mos = calculate_margin_of_safety(
                limits.yield_stress_mpa, applied_stress
            )

            if abs(strain) > limits.wing_root_yield_strain_ue:
                severity = "CRITICAL"
                if abs(strain) > limits.wing_root_ultimate_strain_ue:
                    severity = "ULTIMATE"
                msg = (
                    f"Wing root strain {format_signed(strain, 1)} ue exceeds "
                    f"yield limit {format_signed(limits.wing_root_yield_strain_ue, 1)} ue "
                    f"at t={format_signed(ts, 3)} s. "
                    f"Margin of safety={format_signed(mos, 4)}. "
                    "Reference: MIL STD airframe structural yield criterion."
                )
                logger.warning(msg)
                events.append(
                    ExceedanceEvent(
                        timestamp_s=ts,
                        channel=ChannelType.STRAIN,
                        measured_value=strain,
                        limit_value=limits.wing_root_yield_strain_ue,
                        margin_of_safety=mos,
                        severity=severity,
                        message=msg,
                    )
                )
            elif mos < 0.15:
                msg = (
                    f"Wing root strain {format_signed(strain, 1)} ue approaching "
                    f"yield limit. Margin of safety={format_signed(mos, 4)} "
                    f"at t={format_signed(ts, 3)} s. "
                    "Reference: FAR 25.303 factor of safety guidance."
                )
                logger.warning(msg)
                events.append(
                    ExceedanceEvent(
                        timestamp_s=ts,
                        channel=ChannelType.STRAIN,
                        measured_value=strain,
                        limit_value=limits.wing_root_yield_strain_ue,
                        margin_of_safety=mos,
                        severity="CAUTION",
                        message=msg,
                    )
                )

        # --- Fuselage acceleration -----------------------------------------
        if record.fuselage_accel_g is not None:
            accel = record.fuselage_accel_g
            if accel > limits.fuselage_accel_positive_g:
                msg = (
                    f"Fuselage acceleration {format_signed(accel)} g exceeds "
                    f"positive limit "
                    f"{format_signed(limits.fuselage_accel_positive_g)} g "
                    f"at t={format_signed(ts, 3)} s. "
                    "Reference: design envelope positive load factor."
                )
                logger.warning(msg)
                events.append(
                    ExceedanceEvent(
                        timestamp_s=ts,
                        channel=ChannelType.ACCELERATION,
                        measured_value=accel,
                        limit_value=limits.fuselage_accel_positive_g,
                        severity="CRITICAL",
                        message=msg,
                    )
                )
            elif accel < limits.fuselage_accel_negative_g:
                msg = (
                    f"Fuselage acceleration {format_signed(accel)} g exceeds "
                    f"negative limit "
                    f"{format_signed(limits.fuselage_accel_negative_g)} g "
                    f"at t={format_signed(ts, 3)} s. "
                    "Reference: design envelope negative load factor."
                )
                logger.warning(msg)
                events.append(
                    ExceedanceEvent(
                        timestamp_s=ts,
                        channel=ChannelType.ACCELERATION,
                        measured_value=accel,
                        limit_value=limits.fuselage_accel_negative_g,
                        severity="CRITICAL",
                        message=msg,
                    )
                )

        # --- Skin thermal thresholds ---------------------------------------
        if record.skin_temp_c is not None:
            temp = record.skin_temp_c
            if temp > limits.skin_thermal_max_c:
                msg = (
                    f"Skin temperature {format_signed(temp, 1)} C exceeds "
                    f"maximum threshold "
                    f"{format_signed(limits.skin_thermal_max_c, 1)} C "
                    f"at t={format_signed(ts, 3)} s. "
                    "Reference: airframe skin thermal operating envelope."
                )
                logger.warning(msg)
                events.append(
                    ExceedanceEvent(
                        timestamp_s=ts,
                        channel=ChannelType.TEMPERATURE,
                        measured_value=temp,
                        limit_value=limits.skin_thermal_max_c,
                        severity="CRITICAL",
                        message=msg,
                    )
                )
            elif temp < limits.skin_thermal_min_c:
                msg = (
                    f"Skin temperature {format_signed(temp, 1)} C below "
                    f"minimum threshold "
                    f"{format_signed(limits.skin_thermal_min_c, 1)} C "
                    f"at t={format_signed(ts, 3)} s. "
                    "Reference: airframe skin thermal operating envelope."
                )
                logger.warning(msg)
                events.append(
                    ExceedanceEvent(
                        timestamp_s=ts,
                        channel=ChannelType.TEMPERATURE,
                        measured_value=temp,
                        limit_value=limits.skin_thermal_min_c,
                        severity="CRITICAL",
                        message=msg,
                    )
                )

            # Thermal expansion contribution check
            delta_t = temp - self.REFERENCE_TEMP_C
            thermal_strain = thermal_expansion_strain_ue(delta_t)
            if thermal_strain > limits.thermal_expansion_limit_ue:
                msg = (
                    f"Thermal expansion strain "
                    f"{format_signed(thermal_strain, 1)} ue "
                    f"exceeds limit "
                    f"{format_signed(limits.thermal_expansion_limit_ue, 1)} ue "
                    f"at t={format_signed(ts, 3)} s "
                    f"(delta T={format_signed(delta_t, 1)} C). "
                    "Reference: thermal structural coupling criterion."
                )
                logger.warning(msg)
                events.append(
                    ExceedanceEvent(
                        timestamp_s=ts,
                        channel=ChannelType.TEMPERATURE,
                        measured_value=thermal_strain,
                        limit_value=limits.thermal_expansion_limit_ue,
                        severity="WARNING",
                        message=msg,
                    )
                )

        return events

    # -- Full stream analysis -----------------------------------------------

    def analyze(self, df: pd.DataFrame) -> AnalysisResult:
        """
        Run structural verification across an entire telemetry DataFrame.

        Args:
            df: Multi-channel telemetry DataFrame.

        Returns:
            AnalysisResult summarizing margins, exceedances, and status.
        """
        missing = self.detect_missing_channels(df)
        if missing:
            channels_str = ", ".join(missing)
            msg = (
                f"Missing telemetry channels detected: {channels_str}. "
                "Structural analysis will skip unavailable sensors."
            )
            logger.warning(msg)

        events: list[ExceedanceEvent] = []
        margins: list[float] = []

        for _, row in df.iterrows():
            data: dict[str, Any] = {
                "timestamp_s": float(row.get("timestamp_s", 0.0)),
            }
            for ch in REQUIRED_CHANNELS:
                if ch in df.columns and pd.notna(row.get(ch)):
                    data[ch] = float(row[ch])
            if "scenario" in df.columns and pd.notna(row.get("scenario")):
                data["scenario"] = str(row["scenario"])

            record = TelemetryRecord(**data)
            sample_events = self.evaluate_sample(record)
            events.extend(sample_events)

            # Collect margin for statistical summary
            if record.wing_root_strain_ue is not None:
                stress = strain_to_stress_mpa(
                    abs(record.wing_root_strain_ue),
                    self.limits.youngs_modulus_mpa,
                )
                mos = calculate_margin_of_safety(
                    self.limits.yield_stress_mpa, stress
                )
                if np.isfinite(mos):
                    margins.append(mos)

        min_mos = float(np.min(margins)) if margins else None
        mean_mos = float(np.mean(margins)) if margins else None

        # Fail if any CRITICAL or ULTIMATE events, or negative MoS
        critical = any(
            e.severity in ("CRITICAL", "ULTIMATE") for e in events
        )
        negative_mos = min_mos is not None and min_mos < 0

        result = AnalysisResult(
            total_samples=len(df),
            exceedance_count=len(events),
            min_margin_of_safety=min_mos,
            mean_margin_of_safety=mean_mos,
            events=events,
            missing_channels=missing,
            passed=not (critical or negative_mos),
        )
        return result

    def analyze_file(self, path: str | Path) -> AnalysisResult:
        """
        Convenience method: load telemetry from disk and analyze.

        Args:
            path: Path to JSON or CSV telemetry file.

        Returns:
            AnalysisResult for the loaded stream.
        """
        df = self.load_telemetry(path)
        return self.analyze(df)

    def summarize(self, result: AnalysisResult) -> str:
        """
        Produce a human readable analysis summary without dash characters.

        Args:
            result: Completed AnalysisResult.

        Returns:
            Multi line summary string suitable for log or console output.
        """
        status = "PASS" if result.passed else "FAIL"
        lines = [
            f"Structural Analysis Status: {status}",
            f"Total Samples: {result.total_samples}",
            f"Exceedance Events: {result.exceedance_count}",
        ]
        if result.min_margin_of_safety is not None:
            lines.append(
                "Minimum Margin of Safety: "
                f"{format_signed(result.min_margin_of_safety, 4)}"
            )
        if result.mean_margin_of_safety is not None:
            lines.append(
                "Mean Margin of Safety: "
                f"{format_signed(result.mean_margin_of_safety, 4)}"
            )
        if result.missing_channels:
            channels = ", ".join(result.missing_channels)
            lines.append(f"Missing Channels: {channels}")
        for event in result.events:
            lines.append(
                f"  [{event.severity}] t={format_signed(event.timestamp_s, 3)}s "
                f"channel={event.channel.value} "
                f"measured={format_signed(event.measured_value)} "
                f"limit={format_signed(event.limit_value)}"
            )
        return "\n".join(lines)


def run_analysis(telemetry_path: str | Path) -> AnalysisResult:
    """
    Entry point for command line structural analysis.

    Args:
        telemetry_path: Path to telemetry JSON or CSV.

    Returns:
        AnalysisResult from the verification run.
    """
    analyzer = StressAnalyzer()
    result = analyzer.analyze_file(telemetry_path)
    print(analyzer.summarize(result))
    return result


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description=(
            "Aircraft Structural Stress & Telemetry Data Verifier. "
            "Evaluates multi channel sensor data against aeronautical limits."
        )
    )
    parser.add_argument(
        "telemetry",
        type=str,
        help="Path to flight telemetry JSON or CSV file",
    )
    args = parser.parse_args()
    analysis = run_analysis(args.telemetry)
    sys.exit(0 if analysis.passed else 1)
