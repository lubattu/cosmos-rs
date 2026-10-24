# Examples

Run any example with `cargo run --release --example <name>`.

1. `quickstart_synthetic` — trains a classifier on synthetic Gaussian clusters.
   ```
   cargo run --release --example quickstart_synthetic
   ```
2. `flat_classifier` — trains the flat configuration, prints its learning curve,
   and round-trips a checkpoint.
   ```
   cargo run --release --example flat_classifier
   ```
3. `patchwise_classifier` — trains the patchwise model on synthetic images.
   ```
   cargo run --release --example patchwise_classifier
   ```
4. `custom_data` — walks the decisions for bringing your own feature matrix
   and labels.
   ```
   cargo run --release --example custom_data
   ```
