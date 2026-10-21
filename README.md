# Airfare prediction and optimal purchase timing

Thesis project by Gábor Galovics, supervised by Evelin Berekméri.
The notebook compares XGBoost and sequence models for airfare changes, then
evaluates fixed-date purchase policies, daily replanning and fitted Q-learning.

Open `Gabor_Galovics_Thesis_work_3_FIXED(2).ipynb` in Colab. Run **Dataset cache
startup** to load prepared inputs from Google Drive. For the first run, install
the notebook dependencies and run **Save prepared datasets to Drive**. The raw
`df_top5_mod.parquet` is expected in Drive; `FLIGHT_DATA_PATH` overrides its path.
See [dataset instructions](datasets/README.md) for rebuilding the cache.

Model fitting cells are independent of optional XGBoost optimization. In the
buy-time section, run the helpers before the desired model cells. Daily replanning
reuses the fixed-date XGBoost forecasters.

Synthetic checks: `python -m unittest discover -s tests`.
