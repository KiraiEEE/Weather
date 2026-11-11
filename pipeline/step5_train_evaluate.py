"""Walk-forward training and prediction, one month at a time.

For each model and month M: train on rows dated before M - 8 days, predict M.
Months before TEST_START are predicted too, but only to calibrate the conformal intervals.

Temperature (tmax, tmin):
  raw, climatology, persistence
  bias_corrected  raw minus the mean error of the last 30 days
  linear          linear regression (MOS)
  lightgbm        quantile LightGBM on the raw-forecast error (10/50/90 %)
  conformal       lightgbm interval rescaled on the past 365 days
Rain:
  climatology, raw (1 if forecast >= 1 mm), logistic
  lightgbm        calibrated classifier for P(rain) + quantile model for the amount on wet days

Writes results/predictions_temperature.csv and results/predictions_rain.csv
"""
import math

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pipeline.config import (CLEAN, FIRST_PREDICTION, GAP_DAYS, INTERVAL, LGBM_PARAMS, MAX_LEAD_RAIN,
                             MIN_TRAIN_ROWS, MODELS, RESULTS, TEST_END, WET_DAY_MM)
from pipeline.step4_features import RAIN_FEATURES, TEMP_FEATURES

KEYS = ["model", "station", "date", "lead"]
LOW_Q, HIGH_Q = (1 - INTERVAL) / 2, 1 - (1 - INTERVAL) / 2      # 0.1 and 0.9


def filled(train, test, columns):
    """Linear models cannot handle missing values: fill them with the training median."""
    median = train[columns].median()
    return train[columns].fillna(median).fillna(0), test[columns].fillna(median).fillna(0)


def temperature_month(train, test, var):
    out = test[KEYS].copy()
    out["obs"] = test[f"obs_{var}"]
    out["raw"] = test[f"fc_{var}"]
    out["climatology"] = test[f"clim_{var}"]
    out["persistence"] = test[f"last_{var}"]
    out["bias_corrected"] = test[f"fc_{var}"] - test[f"bias30_{var}"].fillna(0)

    x_train, x_test = filled(train, test, TEMP_FEATURES)
    out["linear"] = LinearRegression().fit(x_train, train[f"obs_{var}"]).predict(x_test)

    # learn the error of the raw forecast, not the value, so record forecasts are not clipped
    error = train[f"obs_{var}"] - train[f"fc_{var}"]
    quantiles = []
    for q in (LOW_Q, 0.5, HIGH_Q):
        model = LGBMRegressor(objective="quantile", alpha=q, **LGBM_PARAMS)
        model.fit(train[TEMP_FEATURES], error)
        quantiles.append(model.predict(test[TEMP_FEATURES]))
    low, mid, high = np.sort(np.column_stack(quantiles), axis=1).T     # keep low <= mid <= high
    out["lightgbm"] = test[f"fc_{var}"] + mid
    out["lightgbm_low"] = test[f"fc_{var}"] + low
    out["lightgbm_high"] = test[f"fc_{var}"] + high
    return out.assign(target=var)


def rain_month(train, test):
    out = test[KEYS].copy()
    out["obs"] = test.obs_prcp
    wet_train = (train.obs_prcp >= WET_DAY_MM).astype(int)
    out["p_climatology"] = test.clim_wet
    out["p_raw"] = (test.fc_prcp >= WET_DAY_MM).astype(float)
    out["amount_raw"] = test.fc_prcp

    x_train, x_test = filled(train, test, RAIN_FEATURES)
    logistic = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(x_train, wet_train)
    out["p_logistic"] = logistic.predict_proba(x_test)[:, 1]

    # Stage A: chance of rain. Calibrate on the last 25% of the training period (isotonic regression).
    train = train.sort_values("date")
    wet_train = (train.obs_prcp >= WET_DAY_MM).astype(int)
    cut = int(len(train) * 0.75)
    first = LGBMClassifier(**LGBM_PARAMS).fit(train[RAIN_FEATURES][:cut], wet_train[:cut])
    calibration = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    calibration.fit(first.predict_proba(train[RAIN_FEATURES][cut:])[:, 1], wet_train[cut:])
    final = LGBMClassifier(**LGBM_PARAMS).fit(train[RAIN_FEATURES], wet_train)
    out["p_lightgbm"] = calibration.predict(final.predict_proba(test[RAIN_FEATURES])[:, 1])

    # Stage B: how much, if it rains (rainy training days only, log scale because rain is skewed)
    rainy = train[train.obs_prcp >= WET_DAY_MM]
    for name, q in (("amount_lightgbm", 0.5), ("amount_lightgbm_high", HIGH_Q)):
        model = LGBMRegressor(objective="quantile", alpha=q, **{**LGBM_PARAMS, "min_child_samples": 15})
        model.fit(rainy[RAIN_FEATURES], np.log1p(rainy.obs_prcp))
        out[name] = np.maximum(np.expm1(model.predict(test[RAIN_FEATURES])), WET_DAY_MM)
    return out.assign(target="rain")


def add_conformal(temp):
    """Conformal intervals: for each month, look at the last 365 days of lightgbm predictions
    (same model, target, lead) and widen the interval until 80% of those observations fit inside."""
    temp = temp.copy()
    score = np.maximum(temp.lightgbm_low - temp.obs, temp.obs - temp.lightgbm_high)   # >0 = outside
    temp["conformal_low"] = np.nan
    temp["conformal_high"] = np.nan
    month = temp.date.dt.to_period("M").dt.to_timestamp()
    for (_model, _target, _lead), rows in temp.groupby(["model", "target", "lead"]):
        for m in month[rows.index].unique():
            past = rows[(rows.date >= m - pd.Timedelta(days=365)) & (rows.date < m - pd.Timedelta(days=GAP_DAYS))]
            past_scores = np.sort(score[past.index].dropna().to_numpy())
            n = len(past_scores)
            if n < 50:
                continue
            k = min(math.ceil((n + 1) * INTERVAL), n)            # finite-sample conformal quantile
            widen = past_scores[k - 1]
            now = rows.index[month[rows.index] == m]
            temp.loc[now, "conformal_low"] = temp.loc[now, "lightgbm_low"] - widen
            temp.loc[now, "conformal_high"] = temp.loc[now, "lightgbm_high"] + widen
    return temp


def main():
    features = pd.read_csv(CLEAN / "features.csv", dtype={"station": str}, parse_dates=["date"])
    temp_parts, rain_parts = [], []
    for model in MODELS:
        data = features[features.model == model]
        for month in pd.date_range(FIRST_PREDICTION, TEST_END, freq="MS"):
            train = data[data.date < month - pd.Timedelta(days=GAP_DAYS)]
            test = data[(data.date >= month) & (data.date < month + pd.DateOffset(months=1))]
            for var in ("tmax", "tmin"):
                tr = train.dropna(subset=[f"obs_{var}", f"fc_{var}"])
                te = test.dropna(subset=[f"obs_{var}", f"fc_{var}"])
                if len(tr) >= MIN_TRAIN_ROWS and len(te):
                    temp_parts.append(temperature_month(tr, te, var))
            tr = train[train.lead <= MAX_LEAD_RAIN].dropna(subset=["obs_prcp", "fc_prcp"])
            te = test[test.lead <= MAX_LEAD_RAIN].dropna(subset=["obs_prcp", "fc_prcp"])
            if len(tr) >= MIN_TRAIN_ROWS and len(te) and (tr.obs_prcp >= WET_DAY_MM).sum() >= 50:
                rain_parts.append(rain_month(tr, te))
            print(f"{model} {month:%Y-%m}: trained on {len(train)} rows, predicted {len(test)}")
    temp = add_conformal(pd.concat(temp_parts, ignore_index=True))
    temp.to_csv(RESULTS / "predictions_temperature.csv", index=False)
    pd.concat(rain_parts, ignore_index=True).to_csv(RESULTS / "predictions_rain.csv", index=False)
    print("Saved results/predictions_temperature.csv and results/predictions_rain.csv")


if __name__ == "__main__":
    main()
