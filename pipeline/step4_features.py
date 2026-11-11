"""Feature table, one row per (station, model, valid day D, lead L).

A forecast at lead L is issued at the end of day I = D - L. At that time we know tmax(I), tmin(I),
prcp(I-1) (prcp(I) only ends at 06 UTC on I+1) and forecast errors up to I.
tests/test_pipeline.py checks this by changing later observations.

Writes data/clean/features.csv
"""
import numpy as np
import pandas as pd

from pipeline.config import CLEAN, CLIMATOLOGY_END, STATIONS, WET_DAY_MM

TEMP_FEATURES = ["fc_tmax", "fc_tmin", "fc_dew", "fc_cloud", "fc_wind", "lead", "doy_sin", "doy_cos",
                 "lat", "lon", "elev", "last_tmax", "last_tmin", "clim_tmax", "clim_tmin",
                 "bias30_tmax", "bias30_tmin"]
RAIN_FEATURES = ["fc_prcp", "fc_tmax", "fc_tmin", "fc_dew", "fc_cloud", "fc_wind", "lead", "doy_sin",
                 "doy_cos", "lat", "lon", "elev", "last_prcp", "rain_30d", "clim_wet"]


def climatology(obs):
    """Usual tmax, tmin and chance of rain for each station and day of the year (years <= 2023),
    smoothed with a 31-day moving average."""
    old = obs[obs.date <= CLIMATOLOGY_END].copy()
    old["doy"] = old.date.dt.day_of_year.clip(upper=365)
    old["wet"] = (old.prcp >= WET_DAY_MM).where(old.prcp.notna())
    tables = []
    for station, rows in old.groupby("station"):
        by_day = rows.groupby("doy")[["tmax", "tmin", "wet"]].mean().reindex(range(1, 366))
        wrapped = pd.concat([by_day.iloc[-15:], by_day, by_day.iloc[:15]])        # December wraps to January
        smooth = wrapped.rolling(31, center=True, min_periods=10).mean().iloc[15:-15]
        smooth.columns = ["clim_tmax", "clim_tmin", "clim_wet"]
        tables.append(smooth.reset_index(drop=True).assign(station=station, doy=range(1, 366)))
    return pd.concat(tables)


def known_at_issue(obs):
    """Observations known at the end of each day (that day is called the issue day)."""
    tables = []
    for station, rows in obs.groupby("station"):
        rows = rows.set_index("date").asfreq("D")
        tables.append(pd.DataFrame({
            "station": station,
            "issue": rows.index,
            "last_tmax": rows.tmax.values,
            "last_tmin": rows.tmin.values,
            "last_prcp": rows.prcp.shift(1).values,                   # rain of the day before
            "rain_30d": rows.prcp.shift(1).rolling(30, min_periods=20).sum().values,
        }))
    return pd.concat(tables)


def add_recent_bias(df):
    """Average forecast error over the 30 days up to the issue day (same station, model and lead)."""
    parts = []
    for _key, rows in df.groupby(["station", "model", "lead"]):
        lead = rows.lead.iloc[0]
        rows = rows.set_index("date").asfreq("D")
        for var in ("tmax", "tmin"):
            error = rows[f"fc_{var}"] - rows[f"obs_{var}"]
            # value at day t covers days t-29..t ; shifting by `lead` puts it on day t+lead
            rows[f"bias30_{var}"] = error.rolling(30, min_periods=10).mean().shift(lead)
        parts.append(rows.dropna(subset=["station"]).reset_index())
    return pd.concat(parts)


def build_features(forecasts, obs):
    names = obs.rename(columns={"tmax": "obs_tmax", "tmin": "obs_tmin", "prcp": "obs_prcp"})
    df = forecasts.merge(names[["station", "date", "obs_tmax", "obs_tmin", "obs_prcp"]], on=["station", "date"], how="left")
    df["issue"] = df.date - pd.to_timedelta(df.lead, unit="D")
    df = df.merge(known_at_issue(obs), on=["station", "issue"], how="left")
    df["doy"] = df.date.dt.day_of_year.clip(upper=365)
    df = df.merge(climatology(obs), on=["station", "doy"], how="left")
    df["doy_sin"] = np.sin(2 * np.pi * df.doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df.doy / 365.25)
    info = pd.DataFrame([(k, lat, lon, elev) for k, (_n, lat, lon, elev) in STATIONS.items()],
                        columns=["station", "lat", "lon", "elev"])
    df = df.merge(info, on="station", how="left")
    df = add_recent_bias(df)
    df["lead"] = df.lead.astype(int)
    return df.sort_values(["model", "station", "lead", "date"]).reset_index(drop=True)


def main():
    forecasts = pd.read_csv(CLEAN / "forecasts.csv", dtype={"station": str}, parse_dates=["date"])
    obs = pd.read_csv(CLEAN / "observations.csv", dtype={"station": str}, parse_dates=["date"])
    df = build_features(forecasts, obs)
    df.to_csv(CLEAN / "features.csv", index=False)
    print(f"Saved data/clean/features.csv ({len(df)} rows)")


if __name__ == "__main__":
    main()
