//! State: the single owner of every learnable quantity in COSMOS.
//!
//! [`cosmos_state::CosmosState`] holds every learnable parameter of a model as a
//! list of orthonormal bases `W_i` on the Stiefel manifold `St(k_i, n_i)`. SAOR
//! contract 1 — everything that learns lives in the State — is why no other
//! module in this crate owns a trainable tensor.

pub mod cosmos_state;
pub mod keyed_init;
