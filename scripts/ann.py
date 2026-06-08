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
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# =====================================================
# SETTINGS
# =====================================================

WINDOW_SIZE = 200
MAX_TRAIN = 300000

BASE_DIR = "results_ANN"
os.makedirs(BASE_DIR, exist_ok=True)

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv("master_interpolated_dataset.csv")

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
test_times = []

for sim_id in all_sims:

    sim = df[df["sim_id"] == sim_id]
    sim = sim.sort_values("time")

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

X_train = np.array(X_train, dtype=np.float32)
X_test  = np.array(X_test , dtype=np.float32)

y_train = np.array(y_train, dtype=np.float32)
y_test  = np.array(y_test , dtype=np.float32)

print("Train windows =", len(X_train))
print("Test windows =", len(X_test))

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
# SCALE
# =====================================================

x_scaler = StandardScaler()

X_train = x_scaler.fit_transform(X_train)
X_test = x_scaler.transform(X_test)

y_scaler = StandardScaler()

y_train_scaled = y_scaler.fit_transform(
    y_train.reshape(-1,1)
).flatten()

# =====================================================
# MODELS
# =====================================================

def ANN_2layers():

    model = Sequential([
        Input(shape=(WINDOW_SIZE,)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1)
    ])

    return model


def ANN_3layers():

    model = Sequential([
        Input(shape=(WINDOW_SIZE,)),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1)
    ])

    return model


def ANN_4layers():

    model = Sequential([
        Input(shape=(WINDOW_SIZE,)),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1)
    ])

    return model


def ANN_dropout():

    model = Sequential([
        Input(shape=(WINDOW_SIZE,)),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(1)
    ])

    return model


models = {

    "ANN_2layers": ANN_2layers(),
    "ANN_3layers": ANN_3layers(),
    "ANN_4layers": ANN_4layers(),
    "ANN_dropout": ANN_dropout()

}

# =====================================================
# TRAIN LOOP
# =====================================================

for name, model in models.items():

    print("\n")
    print("="*60)
    print(name)
    print("="*60)

    model_dir = os.path.join(
        BASE_DIR,
        name
    )

    os.makedirs(
        model_dir,
        exist_ok=True
    )

    model.compile(
        optimizer='adam',
        loss='mse'
    )

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

    pred_scaled = model.predict(
        X_test,
        verbose=0
    )

    pred = y_scaler.inverse_transform(
        pred_scaled
    ).flatten()

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
    print("RMSE =", rmse)
    print("MAE  =", mae)
    print("R2   =", r2)

    with open(
        os.path.join(
            model_dir,
            "metrics.txt"
        ),
        "w"
    ) as f:

        f.write(f"RMSE = {rmse}\n")
        f.write(f"MAE = {mae}\n")
        f.write(f"R2 = {r2}\n")

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
            model_dir,
            "test_predictions.csv"
        ),

        index=False

    )

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
            model_dir,
            "true_vs_predicted.png"
        ),
        dpi=300
    )

    plt.close()

    plt.figure(figsize=(8,6))

    plt.hist(
        residual,
        bins=100
    )

    plt.xlabel("Residual")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            model_dir,
            "residual_histogram.png"
        ),
        dpi=300
    )

    plt.close()

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
            model_dir,
            "residual_vs_true.png"
        ),
        dpi=300
    )

    plt.close()

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
            model_dir,
            "error_vs_true.png"
        ),
        dpi=300
    )

    plt.close()

    model.save(
        os.path.join(
            model_dir,
            "model.keras"
        )
    )

print("\nALL ANN MODELS COMPLETE")
