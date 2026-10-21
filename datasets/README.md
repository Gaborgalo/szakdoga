# Prepared datasets on Google Drive

Run **Dataset cache startup**, then **Save prepared datasets to Drive** once if no
cache exists. Later sessions load the existing cache beside `df_top5_mod.parquet`.
New saves use one `df_top5_mod_prepared.pt` file. Older directories with `.part`
files and `manifest.json` also load directly, with archive integrity checks.

There are no preparation-code fingerprints. Loading does not depend on edits to
the notebook. Set `DATASET_CACHE_FORCE_REBUILD=True` when you want new splits or
features. For a rebuild when an older directory occupies the cache path, choose a
new filename in `DATASET_CACHE_PATH` to preserve the existing data.

Startup refreshes matching helper definitions from GitHub and replaces any stale
import in the current runtime. No datasets are uploaded to GitHub.
