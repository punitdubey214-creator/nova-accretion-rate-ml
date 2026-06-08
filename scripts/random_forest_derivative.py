import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

import joblib

# =====================================================
# SETTINGS
# =====================================================

WINDOW_SIZE = 800

OUTDIR = "results_RF_DERIVATIVE_800"
os.makedirs(OUTDIR, exist_ok=True)

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv(
    "master_interpolated_dataset.csv"
)

print("Rows =", len(df))
print("Simulations =", df["sim_id"].nunique())

# =====================================================
# SPLIT SIMULATIONS
# =====================================================

all_sims = df["sim_id"].unique()

train_sims, test_sims = train_test_split(
    all_sims,
    test_size=0.20,
    random_state=42
)

# =====================================================
# BUILD WINDOWS
# =====================================================

X_train = []
y_train = []

X_test = []
y_test = []

test_sim_ids = []
test_times = []

for sim_id in all_sims:

    sim = df[
        df["sim_id"] == sim_id
    ].sort_values("time")

    L = sim["logL"].values
    mdot = sim["log_mdot"].values
    times = sim["time"].values

    for i in range(WINDOW_SIZE, len(sim)):

        window = L[i-WINDOW_SIZE:i]

        # -------------------------
        # DERIVATIVE
        # -------------------------

        dL = np.gradient(window)

        # -------------------------
        # CONCATENATE
        # -------------------------

        features = np.concatenate([
            window,
            dL
        ])

        target = mdot[i]

        if sim_id in train_sims:

            X_train.append(features)
            y_train.append(target)

        else:

            X_test.append(features)
            y_test.append(target)

            test_sim_ids.append(sim_id)
            test_times.append(times[i])

# =====================================================
# NUMPY
# =====================================================

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

print("Train windows =", len(X_train))
print("Test windows  =", len(X_test))
print("Features =", X_train.shape[1])

# =====================================================
# RANDOM FOREST
# =====================================================

print("\nTraining RF + derivative...")

rf = RandomForestRegressor(

    n_estimators=200,

    random_state=42,

    n_jobs=-1

)

rf.fit(
    X_train,
    y_train
)

pred = rf.predict(
    X_test
)

# =====================================================
# METRICS
# =====================================================

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
print("RF + DERIVATIVE")
print("RMSE =", rmse)
print("MAE  =", mae)
print("R2   =", r2)

# =====================================================
# SAVE MODEL
# =====================================================

joblib.dump(
    rf,
    os.path.join(
        OUTDIR,
        "rf_derivative.pkl"
    )
)

# =====================================================
# SAVE RESULTS
# =====================================================

residual = pred - y_test

results = pd.DataFrame({

    "sim_id": test_sim_ids,
    "time": test_times,

    "true_mdot": y_test,
    "pred_mdot": pred,

    "residual": residual

})

results.to_csv(

    os.path.join(
        OUTDIR,
        "test_predictions.csv"
    ),

    index=False

)

with open(
    os.path.join(
        OUTDIR,
        "metrics.txt"
    ),
    "w"
) as f:

    f.write(f"RMSE = {rmse}\n")
    f.write(f"MAE  = {mae}\n")
    f.write(f"R2   = {r2}\n")

# =====================================================
# TRUE VS PRED
# =====================================================

plt.figure(figsize=(8,8))

plt.scatter(
    y_test,
    pred,
    s=1,
    alpha=0.3
)

mn = min(
    y_test.min(),
    pred.min()
)

mx = max(
    y_test.max(),
    pred.max()
)

plt.plot(
    [mn,mx],
    [mn,mx],
    'r--'
)

plt.xlabel("True log_mdot")
plt.ylabel("Predicted log_mdot")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTDIR,
        "true_vs_predicted.png"
    ),
    dpi=300
)

plt.close()

# =====================================================
# RESIDUAL HIST
# =====================================================

plt.figure(figsize=(8,6))

plt.hist(
    residual,
    bins=100
)

plt.xlabel("Residual")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTDIR,
        "residual_histogram.png"
    ),
    dpi=300
)

plt.close()

# =====================================================
# RESIDUAL VS TRUE
# =====================================================

plt.figure(figsize=(8,6))

plt.scatter(
    y_test,
    residual,
    s=1,
    alpha=0.3
)

plt.axhline(
    0,
    color='red',
    linestyle='--'
)

plt.xlabel("True log_mdot")
plt.ylabel("Residual")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTDIR,
        "residual_vs_true.png"
    ),
    dpi=300
)

plt.close()

# =====================================================
# ERROR VS TRUE
# =====================================================

plt.figure(figsize=(8,6))

plt.scatter(
    y_test,
    np.abs(residual),
    s=1,
    alpha=0.3
)

plt.xlabel("True log_mdot")
plt.ylabel("|Error|")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTDIR,
        "error_vs_true.png"
    ),
    dpi=300
)

plt.close()

print("\nDONE")
print("Saved in:", OUTDIR)
