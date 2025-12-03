# quiver-representations

A Python library for computational experiments with quiver representations and quiver Grassmannians.

## Features

- **Quiver construction**: Build quivers with vertices and arrows
- **Module operations**: Create projective/injective modules, compute direct sums, dimension vectors
- **Morphism computations**: Find homomorphism bases between modules
- **Quiver Grassmannians**: Analyze Grassmannian strata via radical computations
- **Degeneration posets**: Construct and visualize partial orders on modules
- **Hilbert functions**: Compute Hilbert functions of Grassmannian strata
- **Support for A_n and D_n quivers**: Specialized algorithms for these Dynkin types

## Installation

### Python Package

```bash
pip install git+https://github.com/st-fedotov/quiver.git@pre-launch-fix
```

### External Dependencies

The library requires several external tools for full functionality:

#### Macaulay2 (required for Grassmannian computations)

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
pip install git+https://github.com/st-fedotov/quiver.git@pre-launch-fix
```

## Quick Start

### Basic Usage

```python
from quiver_representations import Quiver, ComplexNumbers
from quiver_representations.module import Module

# Create an A3 quiver: 0 -> 1 -> 2
Q = Quiver("A3")
v0 = Q.add_vertex("0")
v1 = Q.add_vertex("1")
v2 = Q.add_vertex("2")
Q.add_arrow(v0, v1, "a01")
Q.add_arrow(v1, v2, "a12")

# Work over complex numbers
F = ComplexNumbers()

# Create projective module at vertex 0
P0, _ = Module.projective(Q, F, 0)
print(f"P(0) dimension vector: {P0.get_dimension_vector()}")

# Create injective module at vertex 2
I2, _ = Module.injective(Q, F, 2)
print(f"I(2) dimension vector: {I2.get_dimension_vector()}")

# Direct sum
M, _ = Module.direct_sum(P0, I2)
print(f"P(0) + I(2) dimension vector: {M.get_dimension_vector()}")
```

### Running the Analysis Pipeline

For comprehensive analysis of quiver Grassmannians, see the [examples](examples/) directory:

```bash
# Download the analysis script
curl -O https://raw.githubusercontent.com/st-fedotov/quiver/pre-launch-fix/examples/run_analysis.py

# Download an example config
curl -O https://raw.githubusercontent.com/st-fedotov/quiver/pre-launch-fix/examples/configs/a3_sink.yaml

# Run the analysis
python run_analysis.py a3_sink.yaml
```

This runs the full pipeline: enumeration, radical computations, poset construction, visualization, conjecture checking, and Hilbert function computation.

See [examples/README.md](examples/README.md) for detailed documentation.

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

## Citation

Paper forthcoming.
