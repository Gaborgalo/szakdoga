# Prepared model datasets

`prepared/manifest.json` and `prepared/datasets-*.part` contain one compressed,
versioned cache of the real train/validation/calibration/test model inputs.
No real dataset cache has been generated in this checkout: the airfare parquet
is available in the author's Colab/Drive workflow, not in the cloud task.

In the notebook:

1. Run **Dataset cache startup**.
2. On the first run only, run the dependency cell if needed, then
   **Build/export all prepared datasets once**. A prepared Colab session can export
   its existing arrays/episodes rather than repeating their preparation.
3. Check the reported compressed size and set `PUBLISH_DATASET_CACHE=True` to
   publish the real archive using an authenticated Git checkout. A failure leaves
   the Drive/local cache intact. Alternatively call `dataset_cache.publish_cache`
   after preparing the cache. No credentials belong in the notebook or repository.
4. Later, run startup and the desired model cells. Original Spark/data-preparation
   cells skip work on a cache hit. Model weights are not part of the dataset cache.

The archive uses 40 MiB parts (below GitHub's 100 MiB per-file limit), SHA-256
checksums and restricted `torch.load(weights_only=True)` decoding. The publisher
refuses archives above 500 MiB to avoid filling Git history; keep larger caches
on mounted Drive, or set `FLIGHT_CACHE_DIR` to persistent local storage.

Contents cover the lag/no-time-series XGBoost splits, one-day DL sequences and
anchors/keys, later-validation stacking samples, buy-time and Q-learning datasets,
scales and chronological split metadata. Calibration remains separate from test.

Fingerprints cover saved preparation code and dataset source ID, not neural/XGBoost
hyperparameters. Update `DATASET_CACHE_SOURCE_ID` when changing the raw data.
Unsaved Colab editor changes cannot be fingerprinted from a Git checkout: save
those changes to the checkout and force a rebuild after editing feature preparation.

The loader does not restore Spark sessions or Spark pipeline models. Rebuild for
Spark-specific EDA, chronological Spark CV or the legacy Spark-derived feature-name
plots. Cached model training, financial evaluation, stacking and Q replay are supported.
