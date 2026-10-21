# Prepared datasets on Google Drive

Run **Dataset cache startup**, then **Save prepared datasets to Drive** once.
The notebook saves one `df_top5_mod_prepared.pt` file beside
`/content/drive/MyDrive/df_top5_mod.parquet`. If `FLIGHT_DATA_PATH` points elsewhere,
the prepared file follows that parquet location instead.

Later sessions load that file automatically and skip data preparation. It contains
train/validation/calibration/test inputs for every model family, not trained models.
Set `DATASET_CACHE_FORCE_REBUILD=True` to rebuild. Updating saved preparation code
invalidates the cache; update `DATASET_CACHE_SOURCE_ID` when replacing the raw data.
No dataset files are committed or pushed to GitHub. Spark-specific EDA still needs
raw data and Spark. The actual Drive file must be created in your Colab session.

Older caches stored as a directory of `.part` files plus `manifest.json` are also
loaded directly, even when the directory name ends in `.pt`. No rebuild is needed
when its preparation fingerprint still matches. New saves remain a single file.
