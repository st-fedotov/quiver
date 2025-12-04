# Quiver Analysis Pipeline Examples

This directory contains example scripts and configurations for running quiver analysis pipelines.

## Requirements

- Python 3.8+
- `quiver-representations` library
- [GNU parallel](https://www.gnu.org/software/parallel/)
- [Macaulay2](https://www.macaulay2.com/) (M2)
- [Graphviz](https://graphviz.org/) (for visualization)

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

## What the Pipeline Does

The `run_analysis.py` script requires as an input a `yaml` file with the following information:

* Quiver type (currently supports **An** or **Dn**) and orientation
* Multiplicities of indecomposables in a projective representation **P** and an injective representation **I**, given explicitly as:

  ```
  coverage:
    projective: {0: 1, 1: 2, 2: 1}
    injective: {0: 1, 1: 1, 2: 2}
  ```

  or generally as

  ```
  coverage:
    projective: 1
    injective: 1
  ```

The script performs a series of computations and checks for quiver Grassmannians **Gr_d(M)** with **d = dim(P)** over all representations **M** with dimension vector **dim(P⊕I)**.

The pipeline consists of the following steps:

### Step 1: Enumerate Indecomposable Bags

Enumerates all possible multisets (bags) of indecomposable modules that sum to the given coverage dimension vector. 

### Step 2: Write "RAD" jobs

For each bag, writes all Plücker and incidence relations and generates Macaulay2 scripts to compute 

* Gröbner basis for reduced and saturated ideal cutting **Gr_d(M)**
* Dimensions of irreducible components of **Gr_d(M)**

### Step 3: Run parallel computations of "RAD" jobs with Macaulay 2

Executes the Macaulay2 radical computations in parallel using GNU parallel. This is one of two most computationally intensive steps.

### Step 4: Parse Results

Parses the output of RAD computations into a structured CSV format (`parsed.csv`), extracting:
- Irreducible component dimensions (`irred_dims`)
- Equidimensionality information
- Some other geometric properties

### Step 5: Build Rank Poset

Constructs the degeneration poset (partial order) on the set of modules:
- **For A_n:** Uses Hom-order via interval module combinatorics
- **For D_n:** Uses Hom-matrix computations between indecomposables over a finite field **F_107**

The poset is saved to `edges.csv`.

### Step 5b: Generate Visualization

Creates a DAG visualization (`degeneracy_dag.svg`) of the Hasse diagram of the degeneration poset. Edges are color-coded:
- **Blue:** Both endpoints have exactly one irreducible component of generic dimension
- **Magenta:** Both endpoints have multiple components, all of generic dimension

### Step 6: Check Conjectures

Analyzes the poset structure to verify mathematical conjectures:

- **For A_n:** whether each representation from the *irreducible min dimensional locus* degenerate to **P⊕I** (`reports/conj1_check.csv`); reports sinks of the *min dimensional locus* (`reports/conj2_minimals.csv`).
- **For D_n:** Reports sinks of the *irreducible min dimensional locus* (`dn_minima_single.csv`); sinks of the *min dimensional locus* (`reports/dn_minima_k_multi.csv`).

### Step 7: Compute Hilbert function values

Computes Hilbert functions for each stratum using Macaulay2, up to degree `r_max`. Results are collected into `hilbert_results.csv`.

### Step 8: Check Hilbert function hypothesis

Verifies whether all modules with irreducible components of generic dimension have identical Hilbert function values (`reports/conj3_hilbert.csv`).

### Step 9: Create archive

Packages all relevant results (excluding intermediate batch files) into `results.zip` in the current working directory. The structure of results.zip is:

For **An**:

```
  ├── parsed.csv
  ├── poset_input.json
  ├── rank_poset/
  │   ├── edges.csv
  │   └── ranks.csv
  ├── degeneracy_dag.dot
  ├── degeneracy_dag.svg
  ├── reports/
  │   ├── conj1_minima.csv
  │   ├── conj2_equidim.csv
  │   └── conj3_hilbert.csv
  └── hilbert_results.csv
```

For **Dn**:

```
  ├── parsed.csv
  ├── rank_poset/
  │   └── edges.csv
  ├── degeneracy_dag.dot
  ├── degeneracy_dag.svg
  ├── reports/
  │   ├── dn_minima_single.csv
  │   ├── dn_minima_k_multi.csv
  │   └── conj3_hilbert.csv
  └── hilbert_results.csv
```

## Configuration Format

Configs are YAML files with these sections:

```yaml
name: my_quiver             # Name for this analysis (used for output folder)
type: An                    # REQUIRED: 'An' or 'Dn'

quiver:
  name: MyQuiver            # Display name for the quiver
  vertices: [0, 1, 2]       # Vertices MUST be 0, 1, 2, ..., n-1
  arrows:                   # List of [source, target, label] triples
    - [0, 1, a01]
    - [1, 2, a12]

coverage:
  projective: 1             # Uniform multiplicity, or per-vertex dict
  injective: 1

  # Per-vertex example:
  # projective: {0: 1, 1: 2, 2: 1}
  # injective: {0: 1, 1: 1, 2: 2}

runtime:
  output_dir: ./results     # Base output directory
  workers: 32               # Parallel workers for RAD computation
  hilbert_workers: 32       # Parallel workers for Hilbert computation
  r_max: 3                  # Maximum degree for Hilbert functions
  gc_heap_size: 16G         # Macaulay2 GC heap size
  hom_prime: 107            # Prime for Hom computation (D_n only)
```

The bulk of our experiments were done on a [Nebius](https://nebius.com/)' virtual machine with 128 CPU and 512Gb RAM. Depending on the memory requirements, we used between 30 and 120 parallel computational streams. Though the default `r_max` is 3, we recommend decreasing it for compute-intensive tasks.

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

- `configs/a3_sink.yaml` - A3 sink quiver: `0 -> 1 <- 2` with uniform coverage (1 of each P and I)
- `configs/a3_custom_coverage.yaml` - A3 quiver with per-vertex multiplicities (e.g., `P_0 + 2*P_1 + P_2`)
- `configs/d4.yaml` - D4 quiver: `0 -> 1, 1 -> 2, 1 -> 3` with uniform coverage

## Output Structure

Results are written to `output_dir/<name>/` and archived to `results.zip` in the current directory.

### For A_n Quivers

```
results/<name>/
├── parsed.csv              # Parsed RAD results with irred_dims, equidimensionality
├── poset_input.json        # Input data for rank poset construction
├── rank_poset/
│   ├── edges.csv           # Hasse edges of the degeneration poset
│   └── ranks.csv           # Rank arrays for each module
├── degeneracy_dag.dot      # DOT source for visualization
├── degeneracy_dag.svg      # Rendered DAG visualization
├── reports/
│   ├── conj1_minima.csv    # Local minima analysis
│   ├── conj2_equidim.csv   # Equidimensionality check
│   └── conj3_hilbert.csv   # Hilbert sequence identity check
├── hilbert_results.csv     # Consolidated Hilbert functions
├── batch_rad/              # [excluded from archive] RAD job files
└── batch_hilbert/          # [excluded from archive] Hilbert job files
```

### For D_n Quivers

```
results/<name>/
├── parsed.csv              # Parsed RAD results with irred_dims, equidimensionality
├── rank_poset/
│   └── edges.csv           # Hasse edges of the degeneration poset (Hom-based)
├── degeneracy_dag.dot      # DOT source for visualization
├── degeneracy_dag.svg      # Rendered DAG visualization
├── reports/
│   ├── dn_minima_single.csv   # Local minima with single generic component
│   ├── dn_minima_k_multi.csv  # Local minima with k generic components
│   └── conj3_hilbert.csv      # Hilbert sequence identity check
├── hilbert_results.csv     # Consolidated Hilbert functions
├── batch_rad/              # [excluded from archive] RAD job files
└── batch_hilbert/          # [excluded from archive] Hilbert job files
```

### Output File Descriptions

| File | Description |
|------|-------------|
| `parsed.csv` | Main results table with task ID, irreducible component dimensions, equidimensionality flags |
| `rank_poset/edges.csv` | Hasse diagram edges: `(src, dst)` means src degenerates to dst |
| `rank_poset/ranks.csv` | (A_n only) Rank arrays and interval multiplicities for each module |
| `degeneracy_dag.svg` | Visual representation of the degeneration poset |
| `hilbert_results.csv` | Hilbert function values for each stratum at degrees 0 to r_max |
| `conj1_minima.csv` | (A_n) Analysis of local minima in the generic stratum |
| `conj2_equidim.csv` | (A_n) Equidimensionality verification |
| `conj3_hilbert.csv` | Result of Hilbert sequence identity hypothesis check |
| `dn_minima_single.csv` | (D_n) Modules that are local minima with exactly one generic component |
| `dn_minima_k_multi.csv` | (D_n) Modules that are local minima with k generic components |

## Archive Contents

The `results.zip` file is created in the current working directory and contains all output files **except** the intermediate batch computation directories (`batch_rad/` and `batch_hilbert/`). This keeps the archive size manageable while preserving all derived results.
