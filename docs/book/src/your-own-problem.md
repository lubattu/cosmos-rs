# Your own problem

The library asks very little of your data: a dense `Array2<f64>` of shape
`(samples, features)` and a `Vec<usize>` of labels. There is no trait to
implement and no data-loading framework to adopt. The `custom_data` example
is the whole story in one file; this chapter is its commentary.

## Shape your data

```rust,ignore
use ndarray::Array2;

let inputs: Array2<f64> = load_your_features();   // (n_samples, n_features)
let labels: Vec<usize> = load_your_labels();      // values in 0..n_classes
```

Two habits pay off immediately.

**Scale your features.** The objective measures angles. Features whose
scales differ by orders of magnitude make one coordinate dominate every
angle, and the geometry stops being informative. Standardize, or normalize
each sample to unit norm.

**Do not one-hot your labels.** Classes are subspaces here, not coordinates
of an output vector. The readout owns one anchor frame per class and scores
a sample against all of them.

## Choose the shape of the State

Five numbers define a flat model.

| Field | Meaning | How to choose it |
|---|---|---|
| `input_dim` | ambient width of your features | fixed by your data |
| `input_subspace_dim` | width of the input projector | start near `min(32, input_dim / 8)` |
| `interior_subspace_dim` | width of each interior block | start at half the input subspace |
| `num_interior_blocks` | depth | start at 3; the topology study found depth cheap and flat |
| `cmr_k` | width of each class anchor | start at 8 |

The parameter count is a function of these alone, and the library reports
it. There is no hidden capacity elsewhere.

## Build and train

```rust,ignore
let mut model = CosmosModel::new(
    CosmosModelConfig::new(dim, 2)
        .with_input_subspace_dim(8)
        .with_interior_subspace_dim(4)
        .with_num_interior_blocks(2)
        .with_cmr_k(4)
        .with_interior_topology(InteriorTopology::Series),
)?;
model.get_state_mut().seed_orthonormal(seed)?;

let mut adapter = FlatClassificationAdapter::new(FlatClassificationAdapterConfig {
    loss: ClassificationLoss::GeodesicHinge {
        margin: std::f64::consts::PI / 5.0,
    },
    ..Default::default()
})?;
let mut backend = DeviceBackend::default();
let mut rules = GaugeDynamics::new(
    model.get_state().num_blocks(),
    GaugeDynamicsConfig {
        lr: 0.1,
        momentum: 0.9,
        ..Default::default()
    },
)?;
```

This is excerpted from `crates/cosmos-rs/examples/custom_data.rs`.
`CosmosModelConfig::new` fixes project-wide defaults for the remaining
fields; the `with_*` methods override only the ones a flat model needs.
`GaugeDynamics::new` takes the block count off the model's own state, so
the rule and the model can never disagree about the number of blocks.

Feed batches through the training loop the examples use, and read back the
accuracy and the diagnostics after each epoch.

## Tuning, in the order that pays

1. **`GaugeDynamicsConfig::lr`.** This is the only knob with a large,
   monotone effect on early progress. The article's regime map sweeps it
   over three orders of magnitude; the interesting region is wide, not a
   knife edge.
2. **`momentum`.** Raising it from zero lifts the achievable accuracy
   substantially at small learning rates. The shipped default is `0.9`.
3. **The loss margin**, `ClassificationLoss::GeodesicHinge { margin }` on
   the adapter's config. The angular margin of the hinge, in radians,
   default `π/5`. Smaller margins make the objective easier to satisfy and
   can under-separate the classes; the article's margin ablation walks that
   trade-off.
4. **`momentum_max_norm`.** A ceiling on the momentum norm. It is inert
   across most of the regime map and engages only at the aggressive corner,
   so treat it as a safety rail rather than a tuning knob. `0.0` means
   unconstrained.
5. **`retraction_interval`.** Numerical hygiene. `10` is a reasonable
   default; `0` disables periodic retraction and, on the article's
   protocol, changes neither the invariant nor the accuracy materially.

Depth and topology come last. The article's topology study found the
framework insensitive to both across the range it swept, which is a useful
thing to know before spending a week on architecture search.

## When it does not work

**Accuracy pinned at chance.** Almost always the seeding. Call
`seed_orthonormal` with a seed; an identity-like start is degenerate.

**The orthonormality residual grows.** Something is writing to the State
outside the rule. Every mutation must go through the rule; an objective
that mutates has broken contract 3.

**The loss falls and accuracy does not.** Look at the angle diagnostics.
If the mean angle to the correct anchor falls while the smallest wrong
angle falls with it, the anchors are collapsing together; raise the margin.

**Everything is slow.** Check whether you are building in release mode, and
whether your batch is large enough for the matrix work to amortize. The
cost of an orthogonal update grows with the cube of the block width, and
the per-block form is the cheap path the benchmark measures.
