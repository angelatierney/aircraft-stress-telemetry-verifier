"""
Pytest suite for Aircraft Structural Stress & Telemetry Data Verifier.

Covers edge cases required for defense aerospace structural testing:
    - Thermal expansion limit exceedance
    - Structural yield strain exceedance
    - Negative margin of safety detection
    - Missing telemetry channel handling
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.stress_analyzer import (
    AeronauticalLimits,
    ChannelType,
    StressAnalyzer,
    TelemetryRecord,
    calculate_margin_of_safety,
    strain_to_stress_mpa,
    thermal_expansion_strain_ue,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_limits() -> AeronauticalLimits:
    """Standard aeronautical limit envelope."""
    return AeronauticalLimits()


@pytest.fixture
def analyzer(default_limits: AeronauticalLimits) -> StressAnalyzer:
    """StressAnalyzer configured with default limits."""
    return StressAnalyzer(limits=default_limits)


@pytest.fixture
def nominal_record() -> TelemetryRecord:
    """In envelope telemetry sample (cruise conditions)."""
    return TelemetryRecord(
        timestamp_s=1.0,
        wing_root_strain_ue=800.0,
        fuselage_accel_g=1.0,
        skin_temp_c=25.0,
        scenario="normal_flight",
    )


# ---------------------------------------------------------------------------
# Margin of Safety unit tests
# ---------------------------------------------------------------------------

class TestMarginOfSafety:
    """Validate MoS = (Yield Limit / Applied Stress) - 1."""

    def test_positive_margin(self) -> None:
        """Applied stress below yield produces positive MoS."""
        mos = calculate_margin_of_safety(yield_limit=252.0, applied_stress=126.0)
        assert mos == pytest.approx(1.0)

    def test_zero_margin_at_yield(self) -> None:
        """Applied stress equal to yield produces zero MoS."""
        mos = calculate_margin_of_safety(yield_limit=252.0, applied_stress=252.0)
        assert mos == pytest.approx(0.0)

    def test_negative_margin_of_safety_detection(self) -> None:
        """Applied stress above yield produces negative MoS."""
        mos = calculate_margin_of_safety(yield_limit=252.0, applied_stress=300.0)
        assert mos < 0
        assert mos == pytest.approx((252.0 / 300.0) - 1.0)

    def test_zero_applied_returns_infinity(self) -> None:
        """Zero applied stress is treated as infinite margin."""
        mos = calculate_margin_of_safety(yield_limit=252.0, applied_stress=0.0)
        assert math.isinf(mos)

    def test_non_positive_yield_raises(self) -> None:
        """Non positive yield limit raises ValueError."""
        with pytest.raises(ValueError, match="Yield limit must be positive"):
            calculate_margin_of_safety(yield_limit=0.0, applied_stress=100.0)

    def test_compressive_load_uses_absolute(self) -> None:
        """Negative (compressive) applied stress uses absolute value."""
        mos = calculate_margin_of_safety(yield_limit=252.0, applied_stress=-126.0)
        assert mos == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Thermal expansion limits
# ---------------------------------------------------------------------------

class TestThermalExpansionLimits:
    """Validate thermal expansion strain checks against specification limits."""

    def test_thermal_expansion_within_limit(
        self, analyzer: StressAnalyzer
    ) -> None:
        """Modest temperature rise stays within thermal expansion limit."""
        record = TelemetryRecord(
            timestamp_s=5.0,
            wing_root_strain_ue=800.0,
            fuselage_accel_g=1.0,
            skin_temp_c=40.0,  # delta T = 20 C -> ~460 ue
            scenario="normal_flight",
        )
        events = analyzer.evaluate_sample(record)
        thermal_events = [
            e for e in events
            if e.channel == ChannelType.TEMPERATURE and e.severity == "WARNING"
        ]
        assert len(thermal_events) == 0

    def test_thermal_expansion_limit_exceedance(
        self, analyzer: StressAnalyzer, default_limits: AeronauticalLimits
    ) -> None:
        """Large temperature excursion exceeds thermal expansion limit."""
        # delta T needed: limit_ue / (coeff * 1e6) = 800 / 23 ≈ 34.8 C
        # Use skin_temp well above reference + that delta
        hot_temp = StressAnalyzer.REFERENCE_TEMP_C + 50.0  # delta=50 -> 1150 ue
        record = TelemetryRecord(
            timestamp_s=10.0,
            wing_root_strain_ue=800.0,
            fuselage_accel_g=1.0,
            skin_temp_c=hot_temp,
            scenario="over_g_maneuver",
        )
        events = analyzer.evaluate_sample(record)
        thermal_events = [
            e for e in events
            if e.channel == ChannelType.TEMPERATURE
            and e.limit_value == default_limits.thermal_expansion_limit_ue
        ]
        assert len(thermal_events) >= 1
        assert thermal_events[0].measured_value > default_limits.thermal_expansion_limit_ue

    def test_thermal_expansion_formula(self) -> None:
        """Verify thermal strain calculation for aluminum coefficient."""
        strain = thermal_expansion_strain_ue(delta_temp_c=100.0)
        # 100 C * 23e-6 * 1e6 = 2300 ue
        assert strain == pytest.approx(2300.0)

    def test_skin_temp_above_max_threshold(
        self, analyzer: StressAnalyzer, default_limits: AeronauticalLimits
    ) -> None:
        """Skin temperature above max C triggers CRITICAL exceedance."""
        record = TelemetryRecord(
            timestamp_s=12.0,
            wing_root_strain_ue=800.0,
            fuselage_accel_g=1.0,
            skin_temp_c=default_limits.skin_thermal_max_c + 10.0,
            scenario="over_g_maneuver",
        )
        events = analyzer.evaluate_sample(record)
        temp_critical = [
            e for e in events
            if e.channel == ChannelType.TEMPERATURE and e.severity == "CRITICAL"
        ]
        assert len(temp_critical) >= 1

    def test_warning_messages_contain_no_dashes(
        self, analyzer: StressAnalyzer, default_limits: AeronauticalLimits
    ) -> None:
        """Exceedance warning strings must not contain dash characters."""
        record = TelemetryRecord(
            timestamp_s=12.0,
            wing_root_strain_ue=default_limits.wing_root_yield_strain_ue + 500.0,
            fuselage_accel_g=7.5,
            skin_temp_c=default_limits.skin_thermal_max_c + 20.0,
            scenario="over_g_maneuver",
        )
        events = analyzer.evaluate_sample(record)
        assert len(events) > 0
        for event in events:
            assert "-" not in event.message, (
                f"Dash found in message: {event.message}"
            )


# ---------------------------------------------------------------------------
# Structural yield strain exceedance
# ---------------------------------------------------------------------------

class TestYieldStrainExceedance:
    """Validate detection of wing root yield and ultimate strain exceedance."""

    def test_strain_within_yield(
        self, analyzer: StressAnalyzer, nominal_record: TelemetryRecord
    ) -> None:
        """Nominal cruise strain produces no strain exceedance events."""
        events = analyzer.evaluate_sample(nominal_record)
        strain_events = [e for e in events if e.channel == ChannelType.STRAIN]
        assert len(strain_events) == 0

    def test_structural_yield_strain_exceedance(
        self, analyzer: StressAnalyzer, default_limits: AeronauticalLimits
    ) -> None:
        """Strain above yield limit triggers CRITICAL exceedance with MoS."""
        over_yield = default_limits.wing_root_yield_strain_ue + 200.0
        record = TelemetryRecord(
            timestamp_s=20.0,
            wing_root_strain_ue=over_yield,
            fuselage_accel_g=3.0,
            skin_temp_c=30.0,
            scenario="over_g_maneuver",
        )
        events = analyzer.evaluate_sample(record)
        strain_events = [e for e in events if e.channel == ChannelType.STRAIN]
        assert len(strain_events) >= 1
        assert strain_events[0].severity == "CRITICAL"
        assert strain_events[0].margin_of_safety is not None
        assert strain_events[0].margin_of_safety < 0.15

    def test_ultimate_strain_exceedance(
        self, analyzer: StressAnalyzer, default_limits: AeronauticalLimits
    ) -> None:
        """Strain above ultimate limit is flagged ULTIMATE severity."""
        over_ultimate = default_limits.wing_root_ultimate_strain_ue + 100.0
        record = TelemetryRecord(
            timestamp_s=25.0,
            wing_root_strain_ue=over_ultimate,
            fuselage_accel_g=5.0,
            skin_temp_c=40.0,
            scenario="over_g_maneuver",
        )
        events = analyzer.evaluate_sample(record)
        strain_events = [e for e in events if e.channel == ChannelType.STRAIN]
        assert len(strain_events) >= 1
        assert strain_events[0].severity == "ULTIMATE"

    def test_strain_to_stress_conversion(
        self, default_limits: AeronauticalLimits
    ) -> None:
        """Hooke's law conversion: stress = E * strain."""
        stress = strain_to_stress_mpa(
            3500.0, default_limits.youngs_modulus_mpa
        )
        expected = 72000.0 * 3500.0e-6  # 252 MPa
        assert stress == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Negative margin of safety end to end
# ---------------------------------------------------------------------------

class TestNegativeMarginDetection:
    """End to end detection of negative MoS across a telemetry stream."""

    def test_analyze_detects_negative_mos(
        self, analyzer: StressAnalyzer, default_limits: AeronauticalLimits
    ) -> None:
        """Stream with over yield strain yields negative min MoS and FAIL."""
        over_yield_stress = strain_to_stress_mpa(
            default_limits.wing_root_yield_strain_ue + 500.0,
            default_limits.youngs_modulus_mpa,
        )
        expected_mos = calculate_margin_of_safety(
            default_limits.yield_stress_mpa, over_yield_stress
        )
        assert expected_mos < 0

        df = pd.DataFrame(
            [
                {
                    "timestamp_s": 0.0,
                    "wing_root_strain_ue": 800.0,
                    "fuselage_accel_g": 1.0,
                    "skin_temp_c": 25.0,
                    "scenario": "normal_flight",
                },
                {
                    "timestamp_s": 1.0,
                    "wing_root_strain_ue": (
                        default_limits.wing_root_yield_strain_ue + 500.0
                    ),
                    "fuselage_accel_g": 7.0,
                    "skin_temp_c": 50.0,
                    "scenario": "over_g_maneuver",
                },
            ]
        )
        result = analyzer.analyze(df)
        assert result.min_margin_of_safety is not None
        assert result.min_margin_of_safety < 0
        assert result.passed is False
        assert result.exceedance_count > 0

    def test_nominal_stream_passes(self, analyzer: StressAnalyzer) -> None:
        """All in envelope samples produce a PASS result."""
        df = pd.DataFrame(
            [
                {
                    "timestamp_s": float(i),
                    "wing_root_strain_ue": 800.0 + i * 10.0,
                    "fuselage_accel_g": 1.0,
                    "skin_temp_c": 25.0,
                    "scenario": "normal_flight",
                }
                for i in range(5)
            ]
        )
        result = analyzer.analyze(df)
        assert result.passed is True
        assert result.min_margin_of_safety is not None
        assert result.min_margin_of_safety > 0


# ---------------------------------------------------------------------------
# Missing telemetry channel handling
# ---------------------------------------------------------------------------

class TestMissingTelemetryChannels:
    """Validate graceful handling of absent or null sensor channels."""

    def test_detect_fully_missing_column(
        self, analyzer: StressAnalyzer
    ) -> None:
        """Absent column is reported as a missing channel."""
        df = pd.DataFrame(
            {
                "timestamp_s": [0.0, 1.0],
                "wing_root_strain_ue": [800.0, 900.0],
                # fuselage_accel_g intentionally omitted
                "skin_temp_c": [25.0, 26.0],
            }
        )
        missing = analyzer.detect_missing_channels(df)
        assert "fuselage_accel_g" in missing
        assert "wing_root_strain_ue" not in missing
        assert "skin_temp_c" not in missing

    def test_detect_all_null_column(self, analyzer: StressAnalyzer) -> None:
        """Column present but entirely null is treated as missing."""
        df = pd.DataFrame(
            {
                "timestamp_s": [0.0, 1.0],
                "wing_root_strain_ue": [800.0, 900.0],
                "fuselage_accel_g": [None, None],
                "skin_temp_c": [25.0, 26.0],
            }
        )
        missing = analyzer.detect_missing_channels(df)
        assert "fuselage_accel_g" in missing

    def test_analyze_with_missing_channels_continues(
        self, analyzer: StressAnalyzer
    ) -> None:
        """Analysis proceeds on available channels when some are missing."""
        df = pd.DataFrame(
            {
                "timestamp_s": [0.0, 1.0],
                "wing_root_strain_ue": [800.0, 4000.0],
                # No acceleration or temperature channels
            }
        )
        result = analyzer.analyze(df)
        assert "fuselage_accel_g" in result.missing_channels
        assert "skin_temp_c" in result.missing_channels
        # Strain channel still evaluated; over yield sample should flag
        assert result.exceedance_count >= 1
        assert result.passed is False

    def test_partial_null_sample_skipped_gracefully(
        self, analyzer: StressAnalyzer
    ) -> None:
        """Individual null channel values are skipped without raising."""
        record = TelemetryRecord(
            timestamp_s=3.0,
            wing_root_strain_ue=800.0,
            fuselage_accel_g=None,
            skin_temp_c=None,
            scenario="normal_flight",
        )
        events = analyzer.evaluate_sample(record)
        # Only strain evaluated; nominal value => no events
        assert events == []

    def test_all_channels_missing(self, analyzer: StressAnalyzer) -> None:
        """DataFrame with only timestamps reports all channels missing."""
        df = pd.DataFrame({"timestamp_s": [0.0, 1.0, 2.0]})
        missing = analyzer.detect_missing_channels(df)
        assert set(missing) == {
            "wing_root_strain_ue",
            "fuselage_accel_g",
            "skin_temp_c",
        }
        result = analyzer.analyze(df)
        assert result.passed is True  # no data to fail on
        assert result.exceedance_count == 0
        assert len(result.missing_channels) == 3


# ---------------------------------------------------------------------------
# Acceleration envelope
# ---------------------------------------------------------------------------

class TestAccelerationBounds:
    """Fuselage acceleration positive and negative limit checks."""

    def test_positive_g_exceedance(
        self, analyzer: StressAnalyzer, default_limits: AeronauticalLimits
    ) -> None:
        """Acceleration above positive g limit is CRITICAL."""
        record = TelemetryRecord(
            timestamp_s=8.0,
            wing_root_strain_ue=1000.0,
            fuselage_accel_g=default_limits.fuselage_accel_positive_g + 1.0,
            skin_temp_c=30.0,
        )
        events = analyzer.evaluate_sample(record)
        accel_events = [
            e for e in events if e.channel == ChannelType.ACCELERATION
        ]
        assert len(accel_events) == 1
        assert accel_events[0].severity == "CRITICAL"

    def test_negative_g_exceedance(
        self, analyzer: StressAnalyzer, default_limits: AeronauticalLimits
    ) -> None:
        """Acceleration below negative g limit is CRITICAL."""
        record = TelemetryRecord(
            timestamp_s=9.0,
            wing_root_strain_ue=1000.0,
            fuselage_accel_g=default_limits.fuselage_accel_negative_g - 1.0,
            skin_temp_c=30.0,
        )
        events = analyzer.evaluate_sample(record)
        accel_events = [
            e for e in events if e.channel == ChannelType.ACCELERATION
        ]
        assert len(accel_events) == 1
        assert accel_events[0].severity == "CRITICAL"


# ---------------------------------------------------------------------------
# Limits validation
# ---------------------------------------------------------------------------

class TestAeronauticalLimits:
    """Pydantic model validation for the specification envelope."""

    def test_default_limits_valid(self) -> None:
        """Default AeronauticalLimits constructs without error."""
        limits = AeronauticalLimits()
        assert limits.wing_root_ultimate_strain_ue > limits.wing_root_yield_strain_ue

    def test_ultimate_below_yield_rejected(self) -> None:
        """Ultimate strain <= yield strain raises ValidationError."""
        with pytest.raises(Exception):
            AeronauticalLimits(
                wing_root_yield_strain_ue=4000.0,
                wing_root_ultimate_strain_ue=3000.0,
            )


# ---------------------------------------------------------------------------
# Summary formatting (no dashes)
# ---------------------------------------------------------------------------

class TestSummaryFormatting:
    """Ensure analyzer summary output contains no dash characters."""

    def test_summarize_has_no_dashes(self, analyzer: StressAnalyzer) -> None:
        """summarize() output is free of ASCII hyphen/dash characters."""
        df = pd.DataFrame(
            [
                {
                    "timestamp_s": 0.0,
                    "wing_root_strain_ue": 4000.0,
                    "fuselage_accel_g": 7.0,
                    "skin_temp_c": 130.0,
                    "scenario": "over_g_maneuver",
                }
            ]
        )
        result = analyzer.analyze(df)
        summary = analyzer.summarize(result)
        assert "-" not in summary
