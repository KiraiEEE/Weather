"""Quality control of the daily observations.

Values failing a rule are set to NaN and counted in the report. Isolation Forest flags are
only listed, not removed, since it can't tell a broken sensor from a real heatwave.

Writes data/clean/observations.csv and results/qc_report.md
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from pipeline.config import CLEAN, CLIMATOLOGY_END, RESULTS, SEED, STATIONS


def find_problems(obs):
    """Return a dict {rule name: True/False per row}."""
    problems = {}
    problems["tmax out of range (-5..52)"] = (obs.tmax < -5) | (obs.tmax > 52)
    problems["tmin out of range (-12..38)"] = (obs.tmin < -12) | (obs.tmin > 38)
    problems["rain out of range (0..300)"] = (obs.prcp < 0) | (obs.prcp > 300)
    problems["tmin higher than tmax"] = obs.tmin > obs.tmax + 0.5

    by_station = obs.groupby("station")
    for var in ("tmax", "tmin"):
        before, after = by_station[var].shift(1), by_station[var].shift(-1)
        problems[f"{var} spike (12 C jump up and back)"] = ((obs[var] - before).abs() > 12) & ((obs[var] - after).abs() > 12)
        same_as_before = obs[var] == by_station[var].shift(1)
        run_length = same_as_before.groupby((~same_as_before).cumsum()).cumsum() + 1
        problems[f"{var} stuck sensor (5 equal days)"] = obs[var].notna() & (run_length >= 5)

        # far from the usual value of that station and month (robust z-score, years <= 2023)
        history = obs[obs.date <= CLIMATOLOGY_END]
        median = history.groupby(["station", history.date.dt.month])[var].median()
        spread = history.groupby(["station", history.date.dt.month])[var].apply(
            lambda x: 1.4826 * (x - x.median()).abs().median())
        key = pd.MultiIndex.from_arrays([obs.station, obs.date.dt.month])
        z = (obs[var].to_numpy() - median.reindex(key).to_numpy()) / np.maximum(spread.reindex(key).to_numpy(), 0.5)
        problems[f"{var} very unusual for the month (|z| > 6)"] = pd.Series(np.abs(z) > 6, index=obs.index)
    return problems


def isolation_forest_flags(obs):
    flags = pd.Series(False, index=obs.index)
    for _station, rows in obs.dropna(subset=["tmax", "tmin"]).groupby("station"):
        features = pd.DataFrame({"tmax": rows.tmax, "tmin": rows.tmin, "range": rows.tmax - rows.tmin,
                                 "month_sin": np.sin(2 * np.pi * rows.date.dt.month / 12),
                                 "month_cos": np.cos(2 * np.pi * rows.date.dt.month / 12)})
        model = IsolationForest(contamination=0.002, random_state=SEED).fit(features)
        flags[rows.index] = model.predict(features) == -1
    return flags


def main():
    obs = pd.read_csv(CLEAN / "observations_raw.csv", dtype={"station": str}, parse_dates=["date"])
    obs = obs.sort_values(["station", "date"]).reset_index(drop=True)
    problems = find_problems(obs)
    unusual = isolation_forest_flags(obs)

    clean = obs.copy()
    for rule, bad in problems.items():
        var = "prcp" if rule.startswith("rain") else rule.split()[0]
        clean.loc[bad.fillna(False).to_numpy(), var] = np.nan
        if rule == "tmin higher than tmax":
            clean.loc[bad.fillna(False).to_numpy(), "tmax"] = np.nan
    clean.to_csv(CLEAN / "observations.csv", index=False)

    counts = pd.DataFrame({"rule": list(problems), "values removed": [int(b.fillna(False).sum()) for b in problems.values()]})
    names = {k: v[0] for k, v in STATIONS.items()}
    valid = clean.groupby("station")[["tmax", "tmin", "prcp"]].count().rename(index=names)
    lines = ["# Quality control report", "",
             f"{len(obs)} station-days, {obs.station.nunique()} stations, {obs.date.min().date()} to {obs.date.max().date()}", "",
             "## Values removed per rule", "", counts.to_markdown(index=False), "",
             f"Isolation Forest marked {int(unusual.sum())} unusual days (kept, for review only).", "",
             "## Valid values per station after QC", "", valid.to_markdown(), ""]
    check_file = CLEAN / "noaa_vs_ogimet_check.csv"
    if check_file.exists():
        lines += ["## Check of our SYNOP decoder: NOAA vs Ogimet on the same days", "",
                  pd.read_csv(check_file).to_markdown(index=False, floatfmt=".1f"), ""]
    (RESULTS / "qc_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("Saved data/clean/observations.csv and results/qc_report.md")


if __name__ == "__main__":
    main()
