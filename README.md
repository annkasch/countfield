# countfield

Learn a continuous field p(x) from sparse count data: at each point x_i, k_i successes
out of N_i trials, k_i ~ Binomial(N_i, p(x_i)). Models return a Gaussian over logit p at a
query point, so predictions are latent-mixed binomial count distributions.

Application-independent. First application: the LEGEND LAr optical map, with data from
[legend-optical-map-ml](https://github.com/annkasch/legend-optical-map-ml).

## Models (`model.kind`)

- `local_latent`: a query point reads its K nearest observed points (relative position and
  counts) through attention; trained leave-one-out on counts (`countfield.model`).
- `point_transformer`: points carry the set of all their trials ("photons": features relative
  to the point and outcome) as feature, summarised by a shared set encoder (conditional MC);
  a Point Transformer U-Net gives every point a feature. Masked training: masked points get a
  null feature and their photons enter no layer (`countfield.pointcloud`, `countfield.train_pt`).

## Usage

```bash
python -m countfield train CONFIG.yaml      # train, select on holdout, score, plot
python -m countfield rescore RUN_DIR        # recompute metrics/plots from saved latents
python -m countfield plot RUN_DIR
sbatch scripts/slurm/train.sbatch CONFIG.yaml   # Perlmutter CPU node
```

Inputs: count-set npz (`countfield.data.save_groups`: x, k, n, group) or photon npz
(`countfield.photons.save_photon_groups`, adds CSR photon arrays). Metrics match the
RESUM_FLEX binomial count GP (`metrics.json`), plus z-scores and denoised MSE.
