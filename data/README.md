# Data

This folder is split into reproducible layers:

- `raw/`: clean source extracts used by the project.
- `processed/`: final monthly modelling dataset and model-comparison outputs.
- `external/`: optional full public downloads from CNMC or datos.gob.es. This folder is ignored by Git because the original public CSV dumps are large and can be downloaded again.

Main pipeline:

1. `src/prepare_cnmc_extract.py` converts downloaded CNMC annual CSV files into `data/raw/gasolina.csv`.
2. `src/prepare_consumption_extract.py` converts downloaded petroleum-consumption CSV files into `data/raw/consum_lleida_gasolina95.csv`.
3. `src/build_dataset.py` merges gasoline prices, Brent, FX, core CPI and consumption into `data/processed/dataset_mensual_final.csv`.
4. `src/create_model_comparison_and_selection.py` compares the final regression feature set against statistical and machine-learning challenger models and regenerates the model-comparison extension and notebook.

The repository already includes the clean raw extracts and the processed CSV dataset required to reproduce the report.
