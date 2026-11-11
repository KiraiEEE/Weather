"""SHAP for the IFS models (trained on everything before TEST_START).

Summary plots for tmax, tmin and rain, and waterfall plots for the hottest and the wettest
lead-1 day at Sfax in the test period.

Writes results/figures/shap_*.png, results/figures/case_*.png, results/shap_summary.md
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from lightgbm import LGBMClassifier, LGBMRegressor  # noqa: E402

from pipeline.config import CLEAN, FIGURES, LGBM_PARAMS, MAIN_STATION, RESULTS, TEST_START, WET_DAY_MM  # noqa: E402
from pipeline.step4_features import RAIN_FEATURES, TEMP_FEATURES  # noqa: E402

MODEL = "IFS"


def main():
    data = pd.read_csv(CLEAN / "features.csv", dtype={"station": str}, parse_dates=["date"])
    data = data[data.model == MODEL]
    train, test = data[data.date < TEST_START], data[data.date >= TEST_START]
    notes = [f"# SHAP summary ({MODEL}, trained before {TEST_START})", ""]

    for target in ("tmax", "tmin", "rain"):
        if target == "rain":
            features, obs_col = RAIN_FEATURES, "obs_prcp"
            tr, te = train.dropna(subset=["obs_prcp", "fc_prcp"]), test.dropna(subset=["obs_prcp", "fc_prcp"])
            model = LGBMClassifier(**LGBM_PARAMS).fit(tr[features], tr.obs_prcp >= WET_DAY_MM)
            meaning = "effect on chance of rain (log-odds)"
        else:
            features, obs_col = TEMP_FEATURES, f"obs_{target}"
            tr, te = train.dropna(subset=[obs_col, f"fc_{target}"]), test.dropna(subset=[obs_col, f"fc_{target}"])
            model = LGBMRegressor(**LGBM_PARAMS).fit(tr[features], tr[obs_col] - tr[f"fc_{target}"])
            meaning = "correction added to the raw forecast (°C)"

        explanation = shap.TreeExplainer(model)(te[features])
        if explanation.values.ndim == 3:                 # classifier: keep the "rain" class
            explanation = explanation[:, :, 1]

        shap.plots.beeswarm(explanation, max_display=12, show=False)
        plt.title(f"{MODEL} {target}: {meaning}", fontsize=9)
        plt.tight_layout()
        plt.savefig(FIGURES / f"shap_{target}.png", dpi=130)
        plt.close("all")

        importance = pd.Series(np.abs(explanation.values).mean(axis=0), index=features).sort_values(ascending=False)
        notes += [f"## {target}", "", "Most important inputs: " +
                  ", ".join(f"{k} ({v:.2f})" for k, v in importance.head(6).items()), ""]

        if target == "tmin":
            continue
        sfax = te[(te.station == MAIN_STATION) & (te.lead == 1)]
        if sfax.empty:
            continue
        day = sfax[obs_col].idxmax()                     # hottest / wettest day
        position = te.index.get_loc(day)
        shap.plots.waterfall(explanation[position], max_display=10, show=False)
        label = "hottest day" if target == "tmax" else "wettest day"
        date = te.loc[day, "date"].date()
        plt.title(f"Sfax {label} {date}: observed {te.loc[day, obs_col]:.1f}", fontsize=9)
        plt.tight_layout()
        plt.savefig(FIGURES / f"case_{target}_{date}.png", dpi=130)
        plt.close("all")
        notes += [f"Case study: Sfax {label} ({date}) -> figures/case_{target}_{date}.png", ""]

    (RESULTS / "shap_summary.md").write_text("\n".join(notes), encoding="utf-8")
    print("Saved SHAP figures and results/shap_summary.md")


if __name__ == "__main__":
    main()
