# Getting started

## Install

```toml
[dependencies]
cosmos-rs = "0.1"
```

The crate needs no system library. The default feature set is pure Rust and
computes in `f64` on the CPU.

If you already have a tuned BLAS/LAPACK and want the dense decompositions
to go through it, enable the optional feature and pick a backend in your
own manifest:

```toml
[dependencies]
cosmos-rs = { version = "0.1", features = ["lapack"] }
ndarray-linalg = { version = "0.18", features = ["openblas-system"] }
```

The crate deliberately selects no backend for you, because the right one
depends on your machine. With the feature enabled you may also choose the
driver the decomposition uses; the default is the more conservative one.

## The shortest complete program

The `quickstart_synthetic` example builds a model, trains it for a few
epochs on separable Gaussian clusters, and prints the accuracy alongside
the orthonormality residual of the State:

```text
cargo run --release --example quickstart_synthetic
```

Read it before anything else: in about forty lines it shows the whole
lifecycle — configure, build, seed, train, evaluate, measure the invariant.

## The four examples

| Example | What it shows |
|---|---|
| `quickstart_synthetic` | the full lifecycle on synthetic clusters |
| `flat_classifier` | a grayscale image classifier with the default configuration, and a checkpoint round trip |
| `patchwise_classifier` | the patch-wise front end on colour images |
| `custom_data` | your own `Array2<f64>` and labels, with nothing else implemented |

Run any of them with:

```text
cargo run --release --example <name>
```

The image examples fetch their dataset on first use and cache it under
`data/`, verifying it against a pinned digest.

## Reading the numbers a run prints

Two quantities deserve attention from the first run.

**Orthonormality residual.** `‖UᵀU − I‖_F` per block, and the aggregate
across blocks. On a healthy run this sits at the numerical floor — on the
order of `1e-14` — and stays there whether or not periodic retraction is
enabled. A value that grows with the epoch count means something is moving
the State outside the gauge rule.

**Commutator norm.** The rule reports the mean commutator between the
connection and the current basis. It is the compatibility diagnostic that
contract 4 requires be exposed: it says how much of the proposed update is
a rotation the frame can actually absorb.

## Seeding

State blocks are seeded with deterministic orthonormal frames:

```rust,ignore
model.get_state_mut().seed_orthonormal(319_635_789)?;
```

This line is excerpted from
`crates/cosmos-rs/examples/quickstart_synthetic.rs`, right after `model` is
built from a `CosmosModelConfig`.

Seeding matters more here than in a conventional network. An identity-like
initialization leaves every block spanning the same subspace, the
objective's angles start degenerate, and the run plateaus at chance. The
seeded orthonormal draw is the documented starting point, and a run is
reproducible from its seed.

## Checkpoints

The checkpoint container is self-describing and `f64`-exact: a magic
header, a format version, an ordered metadata section, named matrices and a
trailing checksum. A reader accepts the format version this crate writes
and rejects any other, rather than guessing.

```rust,ignore
let mut checkpoint = Checkpoint::new();
store_state(model.get_state(), &mut checkpoint);
checkpoint.save(&path)?;

let reloaded = Checkpoint::load(&path)?;
let mut restored = CosmosModel::new(config)?;
restore_state(restored.get_state_mut(), &reloaded)?;
```

This is excerpted from `crates/cosmos-rs/examples/flat_classifier.rs`:
`Checkpoint` is the container, `store_state`/`restore_state` are the free
functions that copy a `CosmosState` to and from it, and `restored` is a
fresh model built from the same `config` as `model`.
