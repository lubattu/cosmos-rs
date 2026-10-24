# Roadmap

This file states what the maintainer intends to work on after `0.1.0` and in
which order. It is a statement of direction, not a schedule. Nothing listed
here is part of the crate until it has been measured and published; see the
[scope rule](docs/book/src/introduction.md#scope).

## 0.1.0 — the article

Version `0.1.0` implements exactly the model family reported in the
Neural Computing and Applications article: flat and patch-wise
classifiers, the geodesic class-manifold-region readout, gauge dynamics on
the Grassmannian and four interior topologies, on the CPU in `f64`.

## Planned extensions

Each item below corresponds to a direction named in the article's outlook or
to infrastructure the maintainer considers necessary for adoption. Each one
will enter the crate only after its own written study, with numbers
reproducible from public artifacts.

### SAOR-consistent nonlinear arrangements

The arrangement layer is currently linear on the subspaces. The plan is to
define nonlinear arrangements that keep the SAOR (state, arrangement,
objective, rule) separation intact, measure each in isolation against the
standard nonlinearities used in conventional networks, and then measure
them coupled with the capacity-aware diagnostics below.

### Capacity-aware diagnostics via intrinsic plasticity

Biological neurons regulate their own excitability. The plan is a
per-block plasticity mechanism driven by geometric sensors (commutator
magnitude, principal-angle motion, spectral shape of the update, temporal
coherence), studied first as a diagnostic of where capacity is used, then
as a controller. This is the first extension scheduled for its own study.

### Objectives beyond classification

Regression and generative objectives on the same state and arrangement.
The generative direction would model sequences with the same geometric
state and readout; it is the conceptually deepest item on this list and
will be written up on its own.

### SAOR applied to conventional architectures

The decomposition is not tied to Grassmannian geometry. A companion study
will apply the same state/arrangement/objective/rule discipline to
standard Transformer and MLP models, to isolate what the decomposition
changes in training and what the geometry adds on top. It will be
published separately and linked from here.

### Python bindings

A `cosmos-py` package built with PyO3 and maturin, versioned against
tagged releases of this crate, with a CPU wheel and, later, a CUDA wheel.
Hardware acceleration will land in this crate as an optional `cuda` feature
implemented with custom kernels; the crate's dependency rules (no external
machine-learning frameworks) will not change.

### Calibrate to your data

Choosing the interior dimension and the readout initialization is the step
most likely to need guidance: an unlucky initialization can sit in a
minimum from which accuracy never improves. The plan is a documented
calibration procedure first, and an automated `calibrate` step before
training if the procedure turns out to be mechanical.

## Order

1. Capacity-aware diagnostics via intrinsic plasticity.
2. SAOR-consistent nonlinear arrangements (they will build on the
   plasticity work).
3. Python bindings.
4. Calibrate to your data.
5. Objectives beyond classification.

The companion study on conventional architectures will be planned
separately from this list.

## How to influence this

Open a discussion issue that names the item above it relates to. Issues
about this code go to the tracker; the wider open-problem catalogue does
not live here (see [Contributing](docs/book/src/contributing.md)).
