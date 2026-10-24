# Concepts

The architecture has four parts. Their names are the vocabulary of the
whole codebase, and the boundaries between them are enforced by tests, not
by convention.

## State

The State is everything that learns. Concretely it is a `CosmosState`: a
collection of named blocks, each an `n × k` matrix whose `k` columns are
orthonormal and therefore span a `k`-dimensional subspace of an
`n`-dimensional ambient space. An input projector, the interior blocks, and
the class anchors of the readout are all blocks in that one container.

Two consequences follow immediately:

- There is exactly one place to look for the model's knowledge, one place
  to checkpoint, and one place to measure. `CosmosState` exposes the
  diagnostics — per-block orthonormality residual, drift from the initial
  basis, Grassmann distance — that the rest of the system reports.
- A quantity that learns but is not a frame does not exist in this
  framework. There are no bias vectors, no scale parameters and no
  trainable normalization statistics hiding in a layer.

The invariant to remember: every block has orthonormal **columns**. The
library measures that with `geometry::matrix_exp::orthogonality_error_columns`,
which is `‖UᵀU − I‖_F` and is defined at any shape. Its sibling
`orthogonality_error` measures rows and is only the same thing when the
block is square.

## Arrangement

The Arrangement is the wiring, and it is gradient-free. It decides how an
input is projected into the first subspace, in which order the interior
blocks act, how their outputs are combined, and how the result reaches the
readout. It holds topology, routing and shapes. It holds no trainable
tensor at all.

Four interior topologies ship:

| Topology | Composition |
|---|---|
| `series` | each block acts on the output of the previous one |
| `parallel` | every block acts on the same input; the outputs are pooled |
| `parallel_series` | parallel groups composed in sequence |
| `residual` | each block's action is combined with its own input |

Because the Arrangement learns nothing, changing it changes the *shape* of
the computation without adding capacity in the usual sense. The article's
topology study varies exactly this axis, holding the State's parameter
count fixed per depth.

## Objectives

Objectives evaluate; they do not act. An objective function takes a State
and a batch and returns a number and a gradient-like quantity, and it never
mutates the State. The signature carries the rule: the functions in
`objectives` take `&CosmosState`, never `&mut`.

The supervised objective used throughout the article is the geodesic
class-manifold-region loss with an angular margin. Each class owns an
anchor subspace; a sample is scored by the geodesic angle between its
representation and each anchor; the loss is a hinge on the difference
between the angle to the correct anchor and the smallest angle to a wrong
one. The margin is an angle in radians, and the default is `π/5`.

## Rules

Rules move the State, globally and coherently. The rule this library ships
is `GaugeDynamics`. Given the evaluation produced by an objective, it
builds an antisymmetric connection per block, applies momentum and an
optional norm ceiling to it, exponentiates it, and applies the resulting
orthogonal transformation to the block.

Three properties matter:

- **The invariant is structural.** The exponential of an antisymmetric
  matrix is orthogonal; an orthogonal map sends an orthonormal frame to an
  orthonormal frame. No projection step is needed to stay on the manifold.
- **Retraction is a numerical hygiene measure, not a correctness
  requirement.** `retraction_interval` re-orthonormalizes every so many
  steps to sweep up floating-point drift. Setting it to `0` disables it,
  and the invariant still holds to the numerical floor; the article's
  retraction study is precisely the measurement of that claim.
- **Updates are coherent.** The connection is built for the whole State at
  once from the same evaluation, rather than block by block from
  independent local signals.

## The contracts

Four rules bind the parts together. They are the project's acceptance
criteria and the integration test suite encodes the first three
mechanically.

1. Everything that learns lives in the State.
2. The Arrangement is gradient-free.
3. Objectives evaluate; they do not act.
4. Relevant updates are global and coherent via gauge: no Euclidean descent
   on bases, invariants preserved by construction, compatibility
   diagnostics exposed.

A change that breaks one of them is not a trade-off to be discussed. It is
a different framework.
