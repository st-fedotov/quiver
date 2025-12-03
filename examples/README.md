# Quiver Analysis Pipeline Examples

This directory contains example scripts and configurations for running quiver analysis pipelines.

## Requirements

- Python 3.8+
- `quiver-representations` library
- [GNU parallel](https://www.gnu.org/software/parallel/)
- [Macaulay2](https://www.macaulay2.com/) (M2)

## Quick Start

1. **Install the library:**
   ```bash
   pip install git+https://github.com/st-fedotov/quiver.git@pre-launch-fix
   ```

2. **Download the run script:**
   ```bash
   curl -O https://raw.githubusercontent.com/st-fedotov/quiver/pre-launch-fix/examples/run_analysis.py
   ```

3. **Create your config file** (or download an example):
   ```bash
   curl -O https://raw.githubusercontent.com/st-fedotov/quiver/pre-launch-fix/examples/configs/a3_sink.yaml
   ```

4. **Run the analysis:**
   ```bash
   python run_analysis.py a3_sink.yaml
   ```

## Configuration Format

Configs are YAML files with these sections:

```yaml
name: my_analysis          # Optional: name for this analysis run
type: An                   # REQUIRED: 'An' or 'Dn'

quiver:
  name: MyQuiver           # Name for the quiver
  vertices: [0, 1, 2]      # Vertices MUST be 0, 1, 2, ..., n-1
  arrows:                  # List of [source, target, label] triples
    - [0, 1, a01]
    - [1, 2, a12]

coverage:
  projective: 1            # Uniform multiplicity, or per-vertex dict
  injective: 1

  # Per-vertex example:
  # projective: {0: 1, 1: 2, 2: 1}
  # injective: {0: 1, 1: 1, 2: 2}

runtime:
  output_dir: ./results    # Where to write output
  workers: 32              # Parallel workers for RAD computation
  hilbert_workers: 32      # Parallel workers for Hilbert computation
  r_max: 3                 # Maximum degree for Hilbert functions
  gc_heap_size: 16G        # Macaulay2 GC heap size
  hom_prime: 107           # Prime for Hom computation (D_n only)
```

## Quiver Types and Vertex Numbering

You must specify the quiver type explicitly via the `type` field.

**Vertices must be numbered 0, 1, 2, ..., n-1** with specific edge structures:

### Type A_n
Edges must form a chain: `0-1, 1-2, 2-3, ..., (n-2)-(n-1)`

```
0 --- 1 --- 2 --- ... --- (n-1)
```

Arrow orientations can be arbitrary.

### Type D_n (n >= 4)
Edges must be: `0-1, 1-2, ..., (n-4)-(n-3), (n-3)-(n-2), (n-3)-(n-1)`

Branching at vertex `n-3`:

```
                    (n-2)
                   /
0 --- 1 --- ... --- (n-3)
                   \
                    (n-1)
```

For D4: `0 - 1, 1 - 2, 1 - 3` (branch at vertex 1)
For D5: `0 - 1, 1 - 2, 2 - 3, 2 - 4` (branch at vertex 2)

Arrow orientations can be arbitrary.

## Example Configurations

- `configs/a3_sink.yaml` - Type A_n quiver with 3 vertices
- `configs/d4_star.yaml` - Type D_n quiver with 4 vertices

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
