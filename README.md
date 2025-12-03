# quiver-representations

A Python library for computational experiments with quiver representations and quiver Grassmannians.

## Features

- **Quivers and modules**: Create custom quivers and representations over the field of complex numbers or over finite fields (based on the `galois` library); infer simple, projective, and injective modules. For now, the library only supports quivers without oriented cycles.
- **Module and morphism operations**: Compute direct sums, kernels, images, radicals and socles, projective covers and injective hulls. Find a basis of **Hom_Q(M, N)** for given **M** and **N**. We recommed to use this functionality only over finite fields - over **C** every matrix is virtually full-rank due to errors in floating-point computations, so correctness isn't guaranteed.
- **Quiver Grassmannians**: Create quiver Grassmannians. Infer equations cutting a quiver Grassmannian in a multi-projective space, as described in [this paper](https://arxiv.org/pdf/1607.01058).
- **Degeneration posets**: Construct and visualize partial orders on modules with the same dimension vector given by the relation "**M** has **N** in the closure of its orbit".
- **Hilbert functions**: Compute Hilbert functions for a quiver Grassmannian
- **Support for A_n and D_n quivers**: Specialized algorithms for these Dynkin types

## Installation

### Python Package

```bash
pip install git+https://github.com/st-fedotov/quiver.git
```

### External Dependencies

The library requires several external tools for full functionality:

#### Macaulay2 (required for computing geometric properties of quiver Grassmannians)

**Ubuntu/Debian:**
```bash
sudo apt install -y macaulay2
```

**macOS (Homebrew):**
```bash
brew install macaulay2
```

**Other systems:** See [Macaulay2 installation guide](https://www.macaulay2.com/doc/Macaulay2/share/doc/Macaulay2/Macaulay2Doc/html/_installing_sp__Macaulay2.html)

Tested with Macaulay2 version 1.22.

#### GNU Parallel (required for batch computations)

**Ubuntu/Debian:**
```bash
sudo apt install -y parallel
```

**macOS (Homebrew):**
```bash
brew install parallel
```

#### Graphviz (required for visualization)

**Ubuntu/Debian:**
```bash
sudo apt install -y graphviz
```

**macOS (Homebrew):**
```bash
brew install graphviz
```

### Full Setup (Ubuntu/Debian)

```bash
# Install system dependencies
sudo apt install -y macaulay2 parallel graphviz

# Install Python package
pip install git+https://github.com/st-fedotov/quiver.git
```

## Quick Start

### Basic Usage

```python
from quiver_representations import Quiver, FiniteField
from quiver_representations.module import Module

# Create an A3 quiver: 0 -> 1 -> 2
Q = Quiver("A3")
v0 = Q.add_vertex("0")
v1 = Q.add_vertex("1")
v2 = Q.add_vertex("2")
Q.add_arrow(v0, v1, "a01")
Q.add_arrow(v1, v2, "a12")

# Work over finite field F_5
F = FiniteField(5)

# Create projective module at vertex 0
P0, S0, _ = Module.projective(Q, F, 0)
print(f"P(0) dimension vector: {P0.get_dimension_vector()}")
# Output: {0: 1, 1: 1, 2: 1}

# Find its radical
rad_P0, inclusion = P0.radical()
print(f"rad P(0) dimension vector: {rad_P0.get_dimension_vector()}")
# Output: {0: 0, 1: 1, 2: 1}

# Find quotient P/rad P via cokernel of the inclusion
top_P0, _ = inclusion.cokernel()
print(f"P(0)/rad P(0) dimension vector: {top_P0.get_dimension_vector()}")
# Output: {0: 1, 1: 0, 2: 0}

# Direct sum P(0) + P(0)/rad P(0)
M, i1, i2, p1, p2 = Module.direct_sum(P0, top_P0)
print(f"P(0) + P(0)/rad P(0) dimension vector: {M.get_dimension_vector()}")
# Output: {0: 2, 1: 1, 2: 1}
```

### Grassmannian hypotheses checks

For examples of comprehensive analysis of quiver Grassmannians, see the [examples](examples/) directory.

## Project Structure

```
quiver_representations/
├── quiver.py              # Quiver class
├── module.py              # Module class and operations
├── morphism.py            # Morphism computations
├── field.py               # Field implementations (Complex, Finite)
├── analysis/              # Grassmannian analysis tools
│   ├── coverage_pipeline.py     # A_n analysis pipeline
│   ├── coverage_pipeline_dn.py  # D_n analysis pipeline
│   ├── visualization.py         # DAG visualization
│   └── conjectures.py           # Conjecture checking
├── interval_modules/      # Interval module enumeration
├── grassmannians/         # Quiver Grassmannian computations
├── batch/                 # Batch job management
└── scripts/               # Shell scripts for parallel execution
```

## License

MIT License. See [LICENSE](LICENSE) for details.

Paper forthcoming.
