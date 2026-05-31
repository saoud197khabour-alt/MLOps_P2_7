# MLOps Lab Block 2 — Dagster + MLflow Pipeline

## 1. Pipeline Architecture

Five Dagster assets are defined in `dagster_pipeline.py`:

```
raw_dataset ──► preprocessed_data ──► trained_model ──► model_evaluation
     └──────────► eda_plots
```

| Asset | Input | Output |
|---|---|---|
| `raw_dataset` | – | metadata DataFrame + image paths |
| `preprocessed_data` | `raw_dataset` | train/val PyTorch DataLoaders (80/20 plant-disjoint split) |
| `eda_plots` | `raw_dataset` | target distribution + biomass vs. age figures |
| `trained_model` | `preprocessed_data` | trained ResNet18 weights + per-epoch metrics |
| `model_evaluation` | `trained_model`, `preprocessed_data` | val metrics, training curves, parity plot |

MLflow tracking is integrated via the official `dagster-mlflow` resource. Each run logs:

- **9 parameters:** `epochs`, `batch_size`, `learning_rate`, `model_name`, `image_size`, `freeze_backbone`, `max_train_batches`, `n_train`, `n_val`
- **6 metrics:** `train_mse`, `val_mse`, `train_mae`, `val_mae` (per epoch), `final_val_mse`, `final_val_mae`
- **Artifacts:** model checkpoint (`model.pt`), training curves, parity plot, metrics file

## 2. Setup Instructions

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Place the dataset folder `mlops_biomass_data/` next to `dagster_pipeline.py`, or set the environment variable:

```bash
export BIOMASS_DATA_DIR=/path/to/mlops_biomass_data
```

Required packages: `dagster`, `dagster-webserver`, `dagster-mlflow`, `mlflow`, `torch`, `torchvision`, `pandas`, `matplotlib`, `Pillow`, `openpyxl`, `scikit-learn`

## 3. Running the Pipeline

**Dagster UI (recommended):**

```bash
export DAGSTER_HOME="$(pwd)/.dagster_home"
dagster dev -f dagster_pipeline.py
# Open http://localhost:3000 → Lineage → "Materialize all"
```

**Dagster CLI:**

```bash
export DAGSTER_HOME="$(pwd)/.dagster_home"
dagster asset materialize -f dagster_pipeline.py --select '*'
```

**Programmatic:**

```bash
python dagster_pipeline.py
```

**MLflow UI:**

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000 → experiment "plant_biomass_pipeline"
```

## 4. Results

### Dagster — successful run

![Dagster successful run](screenshots/dagster_successful_run.png)

### MLflow — experiments

![MLflow experiments](screenshots/mlflow_experiments.png)

**Tracked metrics (2-epoch run, 15 batches/epoch):**

| Metric | Value |
|---|---|
| n_train | 2,931 |
| n_val | 753 |
| Best val MSE | 0.0660 g² |
| Final val MAE | 0.176 g |
| Target std on val | 0.296 g |
