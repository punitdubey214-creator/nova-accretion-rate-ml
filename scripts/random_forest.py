import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# ==========================================================
# SETTINGS
# ==========================================================

WINDOW_SIZE = 200

OUTDIR = "results_RF"

os.makedirs(
    OUTDIR,
    exist_ok=True
)

# ==========================================================
# LOAD DATA
# ==========================================================

print("Loading dataset...")

df = pd.read_csv(
    "master_interpolated_dataset.csv"
)

print("Rows =", len(df))
print(
    "Simulations =",
    df["sim_id"].nunique()
)

# ==========================================================
# TRAIN / TEST SPLIT BY SIMULATION
# ==========================================================

all_sims = df["sim_id"].unique()

train_sims, test_sims = train_test_split(
    all_sims,
    test_size=0.2,
    random_state=42
)

print()
print(
    "Train simulations =",
    len(train_sims)
)

print(
    "Test simulations  =",
    len(test_sims)
)

# ==========================================================
# BUILD WINDOWS
# ==========================================================

X_train = []
y_train = []

X_test = []
y_test = []

for sim_id in all_sims:

    sim = df[
        df["sim_id"] == sim_id
    ].sort_values("time")

    logL = sim["logL"].values
    mdot = sim["log_mdot"].values

    for i in range(
        WINDOW_SIZE,
        len(sim)
    ):

        window = logL[
            i-WINDOW_SIZE:i
        ]

        target = mdot[i]

        if sim_id in train_sims:

            X_train.append(window)
            y_train.append(target)

        else:

            X_test.append(window)
            y_test.append(target)

# ==========================================================
# NUMPY ARRAYS
# ==========================================================

X_train = np.array(
    X_train,
    dtype=np.float32
)

X_test = np.array(
    X_test,
    dtype=np.float32
)

y_train = np.array(
    y_train,
    dtype=np.float32
)

y_test = np.array(
    y_test,
    dtype=np.float32
)

print()
print(
    "Train windows =",
    len(X_train)
)

print(
    "Test windows  =",
    len(X_test)
)

# ==========================================================
# RANDOM FOREST
# ==========================================================

print()
print("Training Random Forest...")

rf = RandomForestRegressor(

    n_estimators=200,

    max_depth=None,

    min_samples_split=2,

    min_samples_leaf=1,

    n_jobs=-1,

    random_state=42
)

rf.fit(
    X_train,
    y_train
)

# ==========================================================
# PREDICTIONS
# ==========================================================

pred = rf.predict(
    X_test
)

# ==========================================================
# METRICS
# ==========================================================

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        pred
    )
)

mae = mean_absolute_error(
    y_test,
    pred
)

r2 = r2_score(
    y_test,
    pred
)

print()
print("Random Forest")
print("RMSE =", rmse)
print("MAE  =", mae)
print("R2   =", r2)

# ==========================================================
# SAVE RESULTS
# ==========================================================

np.save(
    os.path.join(
        OUTDIR,
        "predictions.npy"
    ),
    pred
)

np.save(
    os.path.join(
        OUTDIR,
        "y_test.npy"
    ),
    y_test
)

with open(
    os.path.join(
        OUTDIR,
        "metrics.txt"
    ),
    "w"
) as f:

    f.write(
        f"RMSE = {rmse}\n"
    )

    f.write(
        f"MAE  = {mae}\n"
    )

    f.write(
        f"R2   = {r2}\n"
    )

print()
print("DONE")
print(
    "Results saved in:",
    OUTDIR
)
