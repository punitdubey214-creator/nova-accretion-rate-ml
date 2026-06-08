import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    Flatten,
    Dense
)

from tensorflow.keras.callbacks import EarlyStopping

# =====================================================
# SETTINGS
# =====================================================

WINDOW_SIZE = 50
MAX_TRAIN = 300000

OUTDIR = "results_CNN_1D_50"
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
# TRAIN TEST SPLIT
# =====================================================

all_sims = df["sim_id"].unique()

train_sims, test_sims = train_test_split(
    all_sims,
    test_size=0.2,
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

    logL = sim["logL"].values
    mdot = sim["log_mdot"].values
    times = sim["time"].values

    for i in range(WINDOW_SIZE, len(sim)):

        window = logL[i-WINDOW_SIZE:i]

        if sim_id in train_sims:

            X_train.append(window)
            y_train.append(mdot[i])

        else:

            X_test.append(window)
            y_test.append(mdot[i])

            test_sim_ids.append(sim_id)
            test_times.append(times[i])

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

# =====================================================
# SUBSAMPLE TRAINING
# =====================================================

if len(X_train) > MAX_TRAIN:

    idx = np.random.choice(
        len(X_train),
        MAX_TRAIN,
        replace=False
    )

    X_train = X_train[idx]
    y_train = y_train[idx]

print("Using train =", len(X_train))

# =====================================================
# SCALE INPUTS
# =====================================================

x_scaler = StandardScaler()

X_train = x_scaler.fit_transform(X_train)
X_test = x_scaler.transform(X_test)

# =====================================================
# SCALE TARGET
# =====================================================

y_scaler = StandardScaler()

y_train_scaled = y_scaler.fit_transform(
    y_train.reshape(-1,1)
).flatten()

# =====================================================
# RESHAPE FOR CNN
# =====================================================

X_train = X_train.reshape(
    X_train.shape[0],
    WINDOW_SIZE,
    1
)

X_test = X_test.reshape(
    X_test.shape[0],
    WINDOW_SIZE,
    1
)

# =====================================================
# CNN MODEL
# =====================================================

model = Sequential([

    Input(
        shape=(WINDOW_SIZE,1)
    ),

    Conv1D(
        filters=32,
        kernel_size=5,
        activation="relu"
    ),

    MaxPooling1D(
        pool_size=2
    ),

    Conv1D(
        filters=64,
        kernel_size=5,
        activation="relu"
    ),

    MaxPooling1D(
        pool_size=2
    ),

    Flatten(),

    Dense(
        128,
        activation="relu"
    ),

    Dense(
        64,
        activation="relu"
    ),

    Dense(1)

])

model.summary()

# =====================================================
# COMPILE
# =====================================================

model.compile(
    optimizer="adam",
    loss="mse"
)

# =====================================================
# TRAIN
# =====================================================

early_stop = EarlyStopping(
    patience=10,
    restore_best_weights=True
)

history = model.fit(

    X_train,
    y_train_scaled,

    validation_split=0.1,

    epochs=100,

    batch_size=1024,

    callbacks=[early_stop],

    verbose=1
)

# =====================================================
# PREDICT
# =====================================================

pred_scaled = model.predict(
    X_test,
    verbose=0
)

pred = y_scaler.inverse_transform(
    pred_scaled
).flatten()

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
print("CNN RESULTS")
print("RMSE =", rmse)
print("MAE  =", mae)
print("R2   =", r2)

# =====================================================
# SAVE METRICS
# =====================================================

with open(
    os.path.join(
        OUTDIR,
        "metrics.txt"
    ),
    "w"
) as f:

    f.write(f"RMSE = {rmse}\n")
    f.write(f"MAE = {mae}\n")
    f.write(f"R2 = {r2}\n")

# =====================================================
# SAVE MODEL
# =====================================================

model.save(
    os.path.join(
        OUTDIR,
        "cnn_model.keras"
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

# =====================================================
# TRUE VS PREDICTED
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
# RESIDUAL HISTOGRAM
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
print("Results saved in:", OUTDIR)
