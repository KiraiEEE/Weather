"""Scores, tables and figures for the test period -> results/results.md

Every method is scored on the same rows.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from pipeline.config import (FIGURES, HEAVY_RAIN_MM, MAIN_STATION, MODELS, RESULTS, STATIONS,  # noqa: E402
                             TEST_END, TEST_START, WET_DAY_MM)
from pipeline.utils import bootstrap_ci, brier, coverage, mae  # noqa: E402

TEMP_METHODS = ["raw", "climatology", "persistence", "bias_corrected", "linear", "lightgbm"]
# colour-blind safe colours
COLORS = {"lightgbm": "#2a78d6", "raw": "#eb6834", "linear": "#1baf7a", "bias_corrected": "#eda100",
          "logistic": "#1baf7a", "climatology": "#eda100"}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e1e0d9", "font.size": 9})


def load():
    temp = pd.read_csv(RESULTS / "predictions_temperature.csv", dtype={"station": str}, parse_dates=["date"])
    rain = pd.read_csv(RESULTS / "predictions_rain.csv", dtype={"station": str}, parse_dates=["date"])
    in_test = lambda df: df[(df.date >= TEST_START) & (df.date <= TEST_END)]  # noqa: E731
    temp = in_test(temp).dropna(subset=TEMP_METHODS)        # same rows for every method
    rain = in_test(rain).dropna(subset=["p_climatology", "p_raw", "p_logistic", "p_lightgbm"])
    return temp, rain


def temperature_tables(temp):
    mae_rows, skill_rows = [], []
    for (model, target, lead), g in temp.groupby(["model", "target", "lead"]):
        mae_rows.append({"model": model, "target": target, "lead": lead,
                         **{m: mae(g.obs, g[m]) for m in TEMP_METHODS}})
        # skill vs raw with a 95% confidence interval
        diff = (g.obs - g.lightgbm).abs() - (g.obs - g.raw).abs()
        low, high = bootstrap_ci(g.date, diff)
        raw_mae = mae(g.obs, g.raw)
        skill_rows.append({"model": model, "target": target, "lead": lead, "days": len(g),
                           "MAE raw": raw_mae, "MAE lightgbm": mae(g.obs, g.lightgbm),
                           "improvement %": -100 * diff.mean() / raw_mae,
                           "95% CI low %": -100 * high / raw_mae, "95% CI high %": -100 * low / raw_mae})
    return pd.DataFrame(mae_rows), pd.DataFrame(skill_rows)


def interval_table(temp):
    rows = []
    for (model, target), g in temp.groupby(["model", "target"]):
        c = g.dropna(subset=["conformal_low"])
        rows.append({"model": model, "target": target,
                     "lightgbm coverage %": 100 * coverage(g.obs, g.lightgbm_low, g.lightgbm_high),
                     "lightgbm width": (g.lightgbm_high - g.lightgbm_low).mean(),
                     "conformal coverage %": 100 * coverage(c.obs, c.conformal_low, c.conformal_high),
                     "conformal width": (c.conformal_high - c.conformal_low).mean()})
    return pd.DataFrame(rows)


def rain_table(rain):
    rows = []
    for model, g in rain.groupby("model"):
        wet = (g.obs >= WET_DAY_MM).astype(int)
        heavy = g.obs >= HEAVY_RAIN_MM
        brier_clim = brier(wet, g.p_climatology)
        for method in ("climatology", "raw", "logistic", "lightgbm"):
            p = g[f"p_{method}"]
            row = {"model": model, "method": method, "days": len(g), "rain days": int(wet.sum()),
                   "Brier": brier(wet, p), "Brier skill vs climatology": 1 - brier(wet, p) / brier_clim,
                   "ROC AUC": roc_auc_score(wet, p), "PR AUC": average_precision_score(wet, p)}
            if method in ("raw", "lightgbm"):
                if method == "raw":
                    amount, warning = g.amount_raw, g.amount_raw >= HEAVY_RAIN_MM
                else:   # simple rule: likely rain AND the high estimate reaches the heavy threshold
                    amount, warning = g.amount_lightgbm, (g.p_lightgbm >= 0.5) & (g.amount_lightgbm_high >= HEAVY_RAIN_MM)
                row["MAE on rain days (mm)"] = mae(g.obs[wet == 1], amount[wet == 1])
                row["heavy days"] = int(heavy.sum())
                row["heavy found %"] = 100 * (warning & heavy).sum() / max(heavy.sum(), 1)
                row["false alarms %"] = 100 * (warning & ~heavy).sum() / max(warning.sum(), 1)
            rows.append(row)
    return pd.DataFrame(rows)


def season_table(temp):
    seasons = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
               6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}
    t = temp[(temp.target == "tmax") & (temp.lead <= 3)].assign(season=lambda d: d.date.dt.month.map(seasons))
    rows = [{"model": m, "season": s, "MAE raw": mae(g.obs, g.raw), "MAE lightgbm": mae(g.obs, g.lightgbm)}
            for (m, s), g in t.groupby(["model", "season"])]
    table = pd.DataFrame(rows)
    table["improvement %"] = 100 * (1 - table["MAE lightgbm"] / table["MAE raw"])
    return table


# figures
def plot_mae_by_lead(mae_table):
    fig, axes = plt.subplots(2, len(MODELS), figsize=(3.3 * len(MODELS), 5.5), sharey="row", squeeze=False)
    for i, target in enumerate(("tmax", "tmin")):
        for j, model in enumerate(MODELS):
            ax = axes[i, j]
            t = mae_table[(mae_table.model == model) & (mae_table.target == target)].sort_values("lead")
            for method in ("raw", "bias_corrected", "linear", "lightgbm"):
                if t.empty:
                    continue
                ax.plot(t.lead, t[method], "--" if method == "raw" else "-", marker="o", ms=4, lw=2,
                        color=COLORS[method], label=method)
            ax.set_title(f"{model} - {target}")
            ax.set_xlabel("days ahead")
        axes[i, 0].set_ylabel("mean absolute error (°C)")
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "mae_by_lead.png", dpi=140)
    plt.close(fig)


def plot_reliability(rain):
    fig, axes = plt.subplots(1, len(MODELS), figsize=(3.3 * len(MODELS), 3.3), sharey=True, squeeze=False)
    for ax, model in zip(axes[0], MODELS, strict=True):
        g = rain[rain.model == model]
        ax.plot([0, 1], [0, 1], color="#c3c2b7", lw=1)
        for method in ("logistic", "lightgbm"):
            bins = np.clip((g[f"p_{method}"] * 10).astype(int), 0, 9)
            r = pd.DataFrame({"p": g[f"p_{method}"], "o": g.obs >= WET_DAY_MM, "b": bins}).groupby("b").agg(
                p=("p", "mean"), o=("o", "mean"), n=("o", "size"))
            r = r[r.n >= 10]
            ax.plot(r.p, r.o, marker="o", ms=4, lw=2, color=COLORS[method], label=method)
        ax.set_title(f"{model}: chance of rain")
        ax.set_xlabel("forecast probability")
        ax.legend(fontsize=7)
    axes[0, 0].set_ylabel("observed frequency")
    fig.tight_layout()
    fig.savefig(FIGURES / "rain_reliability.png", dpi=140)
    plt.close(fig)


# report
def main():
    temp, rain = load()
    mae_table, skill = temperature_tables(temp)
    intervals = interval_table(temp)
    rain_scores = rain_table(rain)
    seasons = season_table(temp)
    sfax = temperature_tables(temp[temp.station == MAIN_STATION])[1]
    for name, table in [("temperature_mae", mae_table), ("temperature_skill", skill), ("intervals", intervals),
                        ("rain_scores", rain_scores), ("seasons", seasons), ("sfax_skill", sfax)]:
        table.to_csv(RESULTS / f"{name}.csv", index=False)
    plot_mae_by_lead(mae_table)
    plot_reliability(rain)

    md = lambda t: t.to_markdown(index=False, floatfmt=".2f")  # noqa: E731
    text = [
        "# Results", "",
        f"Test period {TEST_START} to {TEST_END}. Stations: {', '.join(v[0] for v in STATIONS.values())}.", "",
        "## LightGBM vs raw, all stations", "",
        "improvement % = relative drop in MAE; 95% CI from a weekly block bootstrap.", "", md(skill), "",
        "![MAE by lead](figures/mae_by_lead.png)", "",
        "## LightGBM vs raw, Sfax", "", md(sfax), "",
        "## MAE of all temperature methods (°C)", "", md(mae_table), "",
        "## 80% intervals", "", md(intervals), "",
        "## Rain", "", md(rain_scores), "", "![reliability](figures/rain_reliability.png)", "",
        "## Tmax by season, leads 1-3", "", md(seasons), "",
    ]
    (RESULTS / "results.md").write_text("\n".join(text), encoding="utf-8")
    print("Saved results/results.md, tables and figures")


if __name__ == "__main__":
    main()
