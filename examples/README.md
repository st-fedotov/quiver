# Quiver Analysis Pipeline Examples

This directory contains example scripts and configurations for running quiver analysis pipelines.

## Requirements

- Python 3.8+
- `quiver-representations` library: `pip install quiver-representations`
- [GNU parallel](https://www.gnu.org/software/parallel/)
- [Macaulay2](https://www.macaulay2.com/) (M2)

## Quick Start

1. **Install the library:**
   ```bash
   pip install quiver-representations
   ```

2. **Download the run script:**
   ```bash
   curl -O https://raw.githubusercontent.com/st-fedotov/quiver/main/examples/run_analysis.py
   ```

3. **Create your config file** (or download an example):
   ```bash
   curl -O https://raw.githubusercontent.com/st-fedotov/quiver/main/examples/configs/a3_sink.yaml
   ```

4. **Run the analysis:**
   ```bash
   python run_analysis.py a3_sink.yaml
   ```

## Configuration Format

Configs are YAML files with three main sections:

```yaml
# Optional: name for this analysis run
name: my_analysis

quiver:
  name: MyQuiver           # Name for the quiver
  vertices: [0, 1, 2]      # List of vertex labels
  arrows:                  # List of [source, target, label] triples
    - [0, 1, a01]
    - [1, 2, a12]

coverage:
  projective: 1            # Multiplicity for projective modules
  injective: 1             # Multiplicity for injective modules

  # Or per-vertex:
  # projective: {0: 1, 1: 2, 2: 1}
  # injective: {0: 1, 1: 1, 2: 2}

runtime:
  output_dir: ./results    # Where to write output
  workers: 32              # Parallel workers for RAD computation
  hilbert_workers: 32      # Parallel workers for Hilbert computation
  r_max: 3                 # Maximum degree for Hilbert functions
  gc_heap_size: 16G        # Macaulay2 GC heap size
  hom_prime: 107           # Prime for Hom computation (D_n quivers only)
```

## Example Configurations

- `configs/a3_sink.yaml` - Type A quiver: `0 -> 1 <- 2`
- `configs/d4_star.yaml` - Type D quiver: `0 -> 1, 2 -> 1, 1 -> 3`

## Quiver Types

The script automatically detects whether your quiver is type A_n or D_n based on its structure:

- **Type A_n**: Linear or tree-like quivers where each vertex has in-degree at most 1
- **Type D_n**: Quivers with a "fork" vertex that has in-degree >= 2

Different analysis pipelines are used for each type.

## Output

Results are written to the `output_dir` specified in your config:

```
results/
├── config_resolved.yaml     # Full config with defaults filled in
├── <coverage_name>/
│   ├── parsed.csv           # Parsed RAD computation results
│   ├── rank_poset/          # Rank poset data
│   ├── reports/             # Conjecture checking reports
│   ├── hilbert_results.csv  # Hilbert function results
│   ├── batch_rad/           # RAD computation jobs (intermediate)
│   └── batch_hilbert/       # Hilbert computation jobs (intermediate)
└── results.zip              # Archive of results (excluding batch dirs)
```
