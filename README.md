# Aircraft Structural Stress & Telemetry Data Verifier

Automated stress analysis and sensor telemetry processing framework for
airframe structural testing. Evaluates multi-channel sensor data (strain
gauge, accelerometer, and thermal) against aeronautical limits to calculate
yield margins of safety and flag structural exceedances under simulated
maneuver loads.

## System Overview

This repository provides a defense aerospace testing workflow that:

1. **Generates** synthetic multi-channel flight telemetry spanning normal
   cruise, high turbulence, and over-G maneuver regimes.
2. **Parses** time-series strain, acceleration, and temperature channels.
3. **Computes** yield margin of safety for each sample:

   ```
   Margin of Safety = (Yield Limit / Applied Stress) - 1
   ```

4. **Flags** structural limit exceedance events and logs warnings against
   aeronautical specifications (design envelope load factors, thermal
   operating limits, and yield/ultimate strain criteria).

```
aircraft-stress-telemetry-verifier/
├── README.md
├── requirements.txt
├── output/                     # Generated telemetry artifacts
├── src/
│   ├── __init__.py
│   ├── generate_telemetry.py   # Synthetic flight data CLI
│   └── stress_analyzer.py      # Structural verification engine
└── tests/
    └── test_stress.py          # Pytest edge case suite
```

## Specification Matrix

Structural limits map directly to automated validation rules enforced by
`StressAnalyzer`. Defaults reflect a typical aluminum alloy fighter/trainer
airframe structural test envelope.

| Structural Domain | Parameter | Limit | Unit | Automated Validation Rule |
|---|---|---|---|---|
| Wing root strain gauge | Yield strain | 3500 | µε | Flag CRITICAL when \|ε\| > yield; compute MoS |
| Wing root strain gauge | Ultimate strain | 4500 | µε | Escalate to ULTIMATE severity when \|ε\| > ultimate |
| Wing root strain gauge | Yield stress | 252 | MPa | MoS = (σ_yield / σ_applied) − 1; FAIL if MoS < 0 |
| Wing root strain gauge | Young's modulus | 72000 | MPa | σ = E · ε (Hooke's law conversion) |
| Fuselage acceleration | Positive load factor | +6.0 | g | Flag CRITICAL when a_z > +N_z,max |
| Fuselage acceleration | Negative load factor | −3.0 | g | Flag CRITICAL when a_z < −N_z,min |
| Skin thermal threshold | Maximum temperature | 120 | °C | Flag CRITICAL when T_skin > T_max |
| Skin thermal threshold | Minimum temperature | −55 | °C | Flag CRITICAL when T_skin < T_min |
| Skin thermal threshold | Thermal expansion | 800 | µε | Flag WARNING when α·\|ΔT\|·10⁶ > limit |
| Margin caution band | Approaching yield | MoS < 0.15 | — | Flag CAUTION per FAR 25.303 guidance |

Positive MoS indicates operation below yield. Zero MoS indicates the yield
limit. Negative MoS indicates yield exceedance and forces an overall FAIL.

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Generate synthetic flight telemetry

Produces `output/flight_telemetry.json` and `output/flight_telemetry.csv`
covering normal flight, high turbulence, and over-G maneuver scenarios:

```bash
python -m src.generate_telemetry
```

Optional flags:

```bash
python -m src.generate_telemetry --seed 42 --sample-rate 10
python -m src.generate_telemetry --scenarios normal_flight high_turbulence
python -m src.generate_telemetry --output-dir output
```

### 3. Run structural analysis

Evaluate the generated telemetry against the Specification Matrix:

```bash
python -m src.stress_analyzer output/flight_telemetry.json
```

CSV input is also supported:

```bash
python -m src.stress_analyzer output/flight_telemetry.csv
```

Exit code `0` indicates PASS (no critical exceedances, MoS ≥ 0).
Exit code `1` indicates FAIL.

### 4. Run the test suite

```bash
pytest tests/ -v
```

## Flight Scenarios

| Scenario | Duration | Strain Regime | Purpose |
|---|---|---|---|
| `normal_flight` | 30 s | ~800 µε, ~1 g | Baseline in-envelope cruise |
| `high_turbulence` | 20 s | up to ~2800 µε, ~4 g | Gust loading near yield caution band |
| `over_g_maneuver` | 15 s | >3500 µε, >6 g | Intentional exceedance for verifier validation |

## API Usage

```python
from src.stress_analyzer import StressAnalyzer, AeronauticalLimits

analyzer = StressAnalyzer()                          # default limits
# analyzer = StressAnalyzer(limits=AeronauticalLimits(
#     wing_root_yield_strain_ue=3000.0,
# ))

result = analyzer.analyze_file("output/flight_telemetry.json")
print(analyzer.summarize(result))

print(result.passed)
print(result.min_margin_of_safety)
print(result.exceedance_count)
for event in result.events:
    print(event.severity, event.channel, event.message)
```

## License

Released for defense aerospace structural testing and educational use.
