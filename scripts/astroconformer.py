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

from tensorflow.keras.layers import (
    Input,
    Dense,
    Conv1D,
    Dropout,
    LayerNormalization,
    MultiHeadAttention,
    GlobalAveragePooling1D
)

from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

# =====================================================
# SETTINGS
# =====================================================

WINDOW_SIZE = 200
MAX_TRAIN = 300000

OUTDIR = "results_ASTROCONFORMER"

os.makedirs(
    OUTDIR,
    exist_ok=True
)

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv(
    "master_interpolated_dataset.csv"
)

print("Rows =", len(df))
print("Simulations =", df["sim_id"].nunique())

# =====================================================
# SPLIT BY SIMULATION
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

for sim_id in all_sims:

    sim = df[
        df["sim_id"] == sim_id
    ].sort_values("time")

    logL = sim["logL"].values
    mdot = sim["log_mdot"].values

    dlogL = np.gradient(logL)

    for i in range(WINDOW_SIZE, len(sim)):

        lum_window = logL[
            i-WINDOW_SIZE:i
        ]

        der_window = dlogL[
            i-WINDOW_SIZE:i
        ]

        features = np.column_stack(
            [
                lum_window,
                der_window
            ]
        )

        if sim_id in train_sims:

            X_train.append(features)
            y_train.append(mdot[i])

        else:

            X_test.append(features)
            y_test.append(mdot[i])

            test_sim_ids.append(sim_id)

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

# =====================================================
# LIMIT TRAIN SIZE
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
# SCALE FEATURES
# =====================================================

lum_scaler = StandardScaler()

X_train[:,:,0] = lum_scaler.fit_transform(
    X_train[:,:,0]
)

X_test[:,:,0] = lum_scaler.transform(
    X_test[:,:,0]
)

der_scaler = StandardScaler()

X_train[:,:,1] = der_scaler.fit_transform(
    X_train[:,:,1]
)

X_test[:,:,1] = der_scaler.transform(
    X_test[:,:,1]
)

# =====================================================
# SCALE TARGET
# =====================================================

y_scaler = StandardScaler()

y_train_scaled = y_scaler.fit_transform(
    y_train.reshape(-1,1)
).flatten()

# =====================================================
# TRANSFORMER BLOCK
# =====================================================

def transformer_block(
    x,
    head_size=32,
    num_heads=4,
    ff_dim=128,
    dropout=0.1
):

    attn = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=head_size
    )(x,x)

    attn = Dropout(
        dropout
    )(attn)

    x1 = LayerNormalization(
        epsilon=1e-6
    )(x + attn)

    ff = Dense(
        ff_dim,
        activation="relu"
    )(x1)

    ff = Dense(
        x.shape[-1]
    )(ff)

    ff = Dropout(
        dropout
    )(ff)

    return LayerNormalization(
        epsilon=1e-6
    )(x1 + ff)

# =====================================================
# ASTROCONFORMER STYLE MODEL
# =====================================================

inputs = Input(
    shape=(WINDOW_SIZE,2)
)

x = Conv1D(
    64,
    kernel_size=5,
    padding="same",
    activation="relu"
)(inputs)

x = Conv1D(
    128,
    kernel_size=5,
    padding="same",
    activation="relu"
)(x)

x = transformer_block(x)

x = transformer_block(x)

x = GlobalAveragePooling1D()(x)

x = Dense(
    128,
    activation="relu"
)(x)

x = Dense(
    64,
    activation="relu"
)(x)

outputs = Dense(1)(x)

model = Model(
    inputs,
    outputs
)

model.compile(
    optimizer="adam",
    loss="mse"
)

model.summary()

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

    batch_size=512,

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
print("ASTROCONFORMER")
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

    f.write(
        f"RMSE={rmse}\n"
        f"MAE={mae}\n"
        f"R2={r2}\n"
    )

# =====================================================
# PLOT 1
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
# PLOT 2
# RESIDUAL HISTOGRAM
# =====================================================

residual = pred - y_test

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
# PLOT 3
# SIMULATION R² DISTRIBUTION
# =====================================================

scores = []

test_sim_ids = np.array(
    test_sim_ids
)

for sim in np.unique(
    test_sim_ids
):

    mask = (
        test_sim_ids == sim
    )

    r2_sim = r2_score(
        y_test[mask],
        pred[mask]
    )

    scores.append(
        r2_sim
    )

plt.figure(figsize=(8,5))

plt.hist(
    scores,
    bins=20
)

plt.xlabel("Simulation R²")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTDIR,
        "all_simulation_scores.png"
    ),
    dpi=300
)

plt.close()

print("\nDONE")
print("Saved in:", OUTDIR)
