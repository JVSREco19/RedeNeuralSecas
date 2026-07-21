# 🌵 RedeNeuralSecas — Drought Forecasting with LSTMs in Northern Minas Gerais

> **Predicting the Standardized Precipitation Evapotranspiration Index (SPEI) for cities in the Brazilian semi-arid region, one LSTM window at a time.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.5-D00000?logo=keras&logoColor=white)](https://keras.io/)
[![License: unspecified](https://img.shields.io/badge/license-unspecified-lightgrey.svg)](#license)

---

## 📖 What is this?

Code evolved from a split of [**LuizHduarte/Drought**](https://github.com/LuizHduarte/Drought).

**RedeNeuralSecas** (literally *"Neural Network Droughts"*) is a research project that trains **Long Short-Term Memory (LSTM)** networks to forecast the **SPEI** — a multi-scalar drought index widely used in climatology — for cities in the **north of Minas Gerais, Brazil**.

The twist that makes this repository interesting is the **one-to-many learning strategy**: instead of training a separate model for every city, the project groups municipalities into **clusters of geographically close cities** and trains **one LSTM per cluster using the *central* city as training anchor**. That central-city model is then *reused* to predict SPEI for every other (**bordering**) city in the same cluster, testing how well a drought signal learned in one location generalises to its neighbours.

To make the comparison fair and informative, the project runs every experiment under **two windowing regimes** side by side:

| Technique   | Window length | Step | What it tests                                            |
|-------------|---------------|------|----------------------------------------------------------|
| `tumbling`  | 12            | 12   | Non-overlapping yearly chunks — strong generalisation.   |
| `sliding`   | 18            | 2    | Heavily-overlapping windows — high resolution, harder.   |

Both regimes use the same **6-month forecast horizon**, but they differ in every other windowing knob — and that is on purpose, because tumbling and sliding fail and overfit in very different ways:

| Technique   | Window length | Window step | Lookback | Horizon |
|-------------|---------------|-------------|----------|---------|
| `tumbling`  | 12            | 12          | 6        | 6       |
| `sliding`   | 18            | 2           | 12       | 6       |

The two heads also use **independent hyperparameters** (epochs, dense layers, units, dropout, optimiser LR) for the same reason.

---

## ✨ Highlights

- 🧠 **Two-headed LSTM design** — one model per cluster, two heads per model (tumbling / sliding), trained and evaluated in a single pass.
- 🏙️ **One-to-many transfer** — train on a *central* city, predict for every *bordering* city in the cluster using the *same* normalisation parameters.
- 📊 **Rich, side-by-side metrics** — MAE, RMSE, MSE and R² computed **twice** for every city: once in raw NumPy and once through Keras, plus a 3-way equality check (sign / integer / first-4-decimals) between them.
- 🖼️ **Matplotlib visualisations everywhere** — training curves, dataset overlays, and prediction-vs-real striped plots per (cluster, model, city, technique).
- 🧪 **Asserted sanity checks** — the code refuses to save a model whose training-set R² is ≤ 0.
- 💾 **Reusable model artefacts** — every trained LSTM is saved to `.keras` + `.weights.h5` under `Output/Models/`.
- 📁 **Excel reports** — `metrics_central_cities_{tumbling,sliding}.xlsx` and `metrics_bordering_cities_{tumbling,sliding}.xlsx`.

---

## 🗂️ Project structure

```
RedeNeuralSecas/
├── main.py                            # Orchestrator: load → train → apply → save
├── requirements.txt
├── README.md
│
├── NeuralNetwork/                     # ML core
│   ├── config.json                    # All hyperparameters (tumbling_* + sliding_*)
│   └── classes/
│       ├── __init__.py
│       ├── neural_network.py          # Two-headed LSTM wrapper
│       ├── dataset.py                 # IO, normalisation, windowing
│       ├── performance_evaluator.py   # Metrics, equality checks, Excel export
│       └── plotter.py                 # Training-curve & prediction plots
│
├── NeuralNetworkDriver/               # Data layer
│   └── classes/
│       ├── __init__.py
│       └── input_data_loader.py       # Walks ./Data/<CLUSTER>/<CITY>.xlsx
│
├── Data/                              # Input: one subfolder per cluster
│   ├── ESPINOSA/                      # Each folder keeps the '.xlsx' data files
│   ├── LASSANCE/
│   ├── RIO PARDO DE MINAS/
│   ├── SÃO FRANCISCO/
│   └── SÃO JOÃO DA PONTE/
│
└── Output/                            # Auto-recreated on every run
    ├── cluster <NAME>/model <NAME>/city <NAME>/   # PNGs, per city & technique
    ├── metrics_central_cities_tumbling.xlsx
    ├── metrics_central_cities_sliding.xlsx
    ├── metrics_bordering_cities_tumbling.xlsx
    ├── metrics_bordering_cities_sliding.xlsx
    └── Models/                           # Trained LSTM heads, one pair per cluster/technique
        ├── <CLUSTER>_tumbling.keras      # Full Keras model (architecture + weights)
        ├── <CLUSTER>_tumbling.weights.h5 # Weights-only companion to the .keras above
        ├── <CLUSTER>_sliding.keras
        └── <CLUSTER>_sliding.weights.h5
```

---

## 📥 Input data format

Each city file is a plain **`.xlsx`** with two columns:

| Column 1  | Column 2      |
|-----------|---------------|
| `Dates`   | `Series 1`    |

- **Column 1** — month stamp (the index, parsed as `datetime`).
- **Column 2** — the **SPEI** value for that month.

The header is read literally (`Series 1`) and the column is renamed to `SPEI Real` internally. Reference dataset generation lives in the companion repository [**JVSREco19/GenerateCitiesSPEI**](https://github.com/JVSREco19/GenerateCitiesSPEI).

### Layout under `Data/`

The folder name under `Data/` is the **cluster**, and the file names inside are the cities. The cluster and one of the cities **must share the same name** — that city is treated as the **central (training) city**, all others as **bordering (prediction) cities**. In the tree below, each `← central` arrow points at the city that shares its name with the cluster folder.

```
Data/
├───ESPINOSA
│       ESPINOSA.xlsx                  ← central city (training anchor)
│       GAMELEIRAS.xlsx
│       MAMONAS.xlsx
│       MONTE AZUL.xlsx
│       MONTEZUMA.xlsx
│       SANTO ANTÔNIO DO RETIRO.xlsx
│
├───LASSANCE
│       AUGUSTO DE LIMA.xlsx
│       BUENÓPOLIS.xlsx
│       …
│       LASSANCE.xlsx                  ← central
│       …
│
├───RIO PARDO DE MINAS
│       …
│       RIO PARDO DE MINAS.xlsx        ← central
│       …
│
├───SÃO FRANCISCO
│       …
│       SÃO FRANCISCO.xlsx             ← central
│
└───SÃO JOÃO DA PONTE
        …
        SÃO JOÃO DA PONTE.xlsx        ← central
```

---

## ⚙️ Configuration

All hyperparameters live in **`NeuralNetwork/config.json`**. They are split per technique so each windowing regime can be tuned independently:

```jsonc
{
    // Fraction of each city's series used to fit the LSTM (rest is holdout):
    "parcelDataTrain"      : 0.8,

    "tumbling_epochs"      : 150,        // how many full passes over the (non-overlapping) training set
    "tumbling_dense_layers":   3,        // 3 tanh Dense layers stacked after the LSTM cell
    "tumbling_dense_units" :   6,        // units per dense layer
    "tumbling_hidden_units":   9,        // LSTM cell width
    "tumbling_window_len"  :  12,        // total months packed into each input window
    "tumbling_window_step" :  12,        // step between windows (12 = non-overlapping yearly chunks)
    "tumbling_lookback_len":   6,        // how many past months the LSTM actually sees
    "tumbling_horizon_len" :   6,        // how many future months the LSTM predicts

    "sliding_epochs"       : 150,        // 150 was the sweet spot — overfits at 200/400/800, underfits at 100
    "sliding_dense_layers" :   2,        // 5, 4, 3, 1 were tried and lost to overfitting
    "sliding_dense_units"  :  12,
    "sliding_hidden_units" :  18,
    "sliding_window_len"   :  18,        // longer window than tumbling → more temporal context
    "sliding_window_step"  :   2,        // heavy overlap (step = 2 vs window = 18)
    "sliding_lookback_len" :  12,        // lookback is 2× tumbling's
    "sliding_horizon_len"  :   6         // same horizon as tumbling — direct comparison target
}
```

Defaults that are computed at runtime in `NeuralNetwork._set_configs()` and not exposed in the JSON:

| Knob                    | Tumbling | Sliding | Why                                                          |
|-------------------------|----------|---------|--------------------------------------------------------------|
| Optimiser               | Adam     | Adam    | Adam performed best across many LR sweeps.                   |
| Learning rate           | 0.0010   | 0.0015  | Sliding needs a bit more push.                               |
| Recurrent / dense dropout | 0.0    | 0.2     | Sliding overfits hard → needs regularisation.                |
| Loss / metrics          | MSE + MAE + RMSE + R² | same | R² is asserted > 0 on the training set.              |
| Dense activation        | tanh     | tanh    | ReLU on the 3 hidden layers helped sliding a bit (see code). |
| Output activation       | linear   | linear  | Multi-step regression over 6 months.                         |

---

## 🧠 Model architecture

Each `(cluster, technique)` pair becomes a small **LSTM** built by `NeuralNetwork._create_ml_model`:

```
Input(window = lookback_len, 1)
   ↓
LSTM(hidden_units, tanh, recurrent_dropout)         # recurrent_dropout = 0.2 for sliding
   ↓
Dropout(d)                                          # d = 0.0 tumbling / 0.2 sliding
   ↓
Dense(dense_units, tanh)   × dense_layers
   ↓
Dropout(d)
   ↓
Dense(horizon_len = 6, linear)                      # 6-month forecast
```

- **Input shape** is `(lookback_len, 1)` — per-technique (`tumbling_lookback_len` or `sliding_lookback_len`).
- **Output shape** is `(horizon_len,)` — a 6-month vector.
- **Two models per cluster**: one for tumbling, one for sliding. They share the same training/eval pipeline but never see each other's data.

---

## 🚀 How to run

```bash
# 1. Create a virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate           # on Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Drop your city .xlsx files under Data/<CLUSTER>/ (see layout above)

# 3. (Optional) tweak hyperparameters in NeuralNetwork/config.json

# 4. Run the orchestrator
python main.py
```

The pipeline is fully self-contained:

1. **PREPARATION** — wipe and recreate `./Output/`, walk `Data/`, build a `Dataset` per city and a `Plotter`.
2. **CREATION** — instantiate one `NeuralNetwork` per cluster, anchored on its central city.
3. **TRAINING** — train both heads (tumbling + sliding) on the central city's 80% split.
4. **APPLYING** — apply the trained model to the central city (20% holdout) **and** to every bordering city in the cluster, reusing the central city's normalisation parameters.
5. **TERMINATION** — write the two Excel reports per technique and persist the trained models under `Output/Models/`.

Console output is grouped by phase and by city, e.g.:

```
PREPARATION: START
PREPARATION: END
CREATION: START
	Created ML model(s) for ESPINOSA
	Created ML model(s) for LASSANCE
	…
CREATION: END
TRAINING: START
…
APPLYING: START
Model ESPINOSA:
	City GAMELEIRAS
	City MAMONAS
	…
APPLYING: END
TERMINATION: START
TERMINATION: END
```

---

## 📤 Outputs

For every run you get, under `Output/`:

### 📈 Per-city plots

```
Output/cluster <NAME>/model <NAME>/city <CITY>/
    ├── <CITY> training loss tumbling.png
    ├── <CITY> training loss sliding.png
    ├── <CITY> dataset plot.png
    ├── <CITY> tumbling predictions.png
    └── <CITY> sliding  predictions.png
```

- **Training-loss plots** — MSE / MAE / RMSE / R² per epoch, for each technique.
- **Dataset plot** — the SPEI series with the 80/20 train-test split highlighted.
- **Prediction plots** — real vs. predicted SPEI, **vertically striped** between adjacent months (no fake filling lines). Latest fix-set: issues #40, #41, #42, #43.

### 📊 Excel reports (4 files)

| File                                        | Rows                                              | Sort key             |
|---------------------------------------------|---------------------------------------------------|----------------------|
| `metrics_central_cities_tumbling.xlsx`      | one row per cluster (the central city, 80% + 20%) | `Agrupamento`, city  |
| `metrics_central_cities_sliding.xlsx`       | one row per cluster (the central city, 80% + 20%) | `Agrupamento`, city  |
| `metrics_bordering_cities_tumbling.xlsx`    | one row per *(cluster, bordering city)* pair      | `Agrupamento`, city  |
| `metrics_bordering_cities_sliding.xlsx`     | one row per *(cluster, bordering city)* pair      | `Agrupamento`, city  |

Each row contains, for both the 80% and 20% portions:

- `MAE / RMSE / MSE / R^2` from raw **NumPy** computation.
- `MAE / RMSE / MSE / R^2` from **Keras** metric tracking.
- Three **equality flags** (`sign_equal`, `integer_equal`, `first4_equal`) confirming the two implementations agree.

### 💾 Trained models

```
Output/Models/
    ESPINOSA_tumbling.keras   +  ESPINOSA_tumbling.weights.h5
    ESPINOSA_sliding.keras    +  ESPINOSA_sliding.weights.h5
    LASSANCE_tumbling.keras   +  LASSANCE_tumbling.weights.h5
    …
```

Ready to be reloaded with `tf.keras.models.load_model(...)` for downstream analysis or transfer learning.

---

## 🛠️ Implementation notes

- **Normalisation is shared** between central and bordering cities — the *(min, max)* computed on the central city's training split is reused for every prediction in the same cluster, so the model never sees out-of-distribution inputs.
- **Two windowing strategies are kept in lock-step** by `Dataset.format_data_for_model()`, which returns parallel `tumbling` / `sliding` dictionaries for both inputs and month stamps, plus a synthesized `'100%'` portion (concatenation of the 80% + 20% splits).
- **The model refuses to ship a broken result**: after training, `NeuralNetwork.use_neural_network` asserts that the R² on the 80% training split is strictly positive, for both techniques. A failed assertion aborts the run with a descriptive error pointing at the offending city.
- **Tumbling vs. Sliding tuning is asymmetric on purpose.** The source code is annotated with the values that *failed* (batch sizes, learning rates, dropout, dense layer counts, epoch counts), so future tuners know what has already been tried. For example: `Adam` with LRs in `{0.0001, 0.0002, 0.0003, 0.0005, 0.0010, 0.0020}` was tried for sliding and kept failing; `0.0015` is the surviving choice.

---

## 📚 Companion repository

The SPEI series consumed by this project are produced by [**JVSREco19/GenerateCitiesSPEI**](https://github.com/JVSREco19/GenerateCitiesSPEI).

---

## 📚 Citation & context

If you use this project in academic work, please reference the original dissertation it was built for. 


---

## 📄 License

No `LICENSE` file is currently included in this repository, so the licensing terms are **unspecified**. If you plan to reuse, redistribute, or build on this code, please contact the repository owner first to clarify the intended terms.

Data attribution belongs to the SPEI dataset produced by the companion repository [**JVSREco19/GenerateCitiesSPEI**](https://github.com/JVSREco19/GenerateCitiesSPEI).
