"""Small web app on top of the pipeline: browse results, try the correction on one case, run steps.

    python app.py        then open http://127.0.0.1:7860
"""
import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import subprocess  # noqa: E402
import sys  # noqa: E402

import gradio as gr  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from lightgbm import LGBMRegressor  # noqa: E402

from pipeline.config import (CLEAN, LGBM_PARAMS, MAIN_STATION, MODELS, RESULTS, ROOT, STATIONS,  # noqa: E402
                             TEST_END, TEST_START, WET_DAY_MM)
from pipeline.step4_features import TEMP_FEATURES  # noqa: E402
from pipeline.utils import brier, coverage, mae  # noqa: E402

STATION_IDS = {name: wmo for wmo, (name, *_rest) in STATIONS.items()}
STATION_NAMES = list(STATION_IDS)
EDITABLE = ["fc_tmax", "fc_tmin", "fc_dew", "fc_cloud", "fc_wind",
            "last_tmax", "last_tmin", "bias30_tmax", "bias30_tmin"]
LABELS = {"fc_tmax": "Forecast Tmax (°C)", "fc_tmin": "Forecast Tmin (°C)", "fc_dew": "Forecast dew point (°C)",
          "fc_cloud": "Forecast cloud cover (%)", "fc_wind": "Forecast wind (km/h)",
          "last_tmax": "Last observed Tmax (°C)", "last_tmin": "Last observed Tmin (°C)",
          "bias30_tmax": "Tmax forecast error, last 30 days", "bias30_tmin": "Tmin forecast error, last 30 days"}

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e1e0d9", "font.size": 9})

_tables = {}


def table(path, **kwargs):
    """Read a CSV once; read it again only if the file changed (e.g. after a run)."""
    mtime = path.stat().st_mtime
    if path not in _tables or _tables[path][0] != mtime:
        _tables[path] = (mtime, pd.read_csv(path, dtype={"station": str}, **kwargs))
    return _tables[path][1]


def missing(path, steps):
    return f"`{path.relative_to(ROOT)}` not found. Run {steps} in the **Run** tab first."


# Results tab
def show_results(model, station, target, lead):
    path = RESULTS / ("predictions_rain.csv" if target == "rain" else "predictions_temperature.csv")
    if not path.exists():
        return None, missing(path, "steps 1-5")
    df = table(path, parse_dates=["date"])
    g = df[(df.model == model) & (df.station == STATION_IDS[station]) & (df.target == target)
           & (df.lead == lead) & (df.date >= TEST_START) & (df.date <= TEST_END)].sort_values("date")
    g = g.dropna(subset=["obs", "p_lightgbm"] if target == "rain" else ["obs", "raw", "lightgbm"])
    if g.empty:
        return None, "No data for this choice (rain is only available up to lead 6)."

    fig, ax = plt.subplots(figsize=(10, 3.8))
    if target == "rain":
        wet = (g.obs >= WET_DAY_MM).astype(int)
        ax.bar(g.date, g.obs, color="#9bb7d4", width=1, label="observed rain (mm)")
        ax.set_ylabel("mm")
        ax2 = ax.twinx()
        ax2.plot(g.date, g.p_lightgbm, color="#2a78d6", lw=1.2, label="P(rain) LightGBM")
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("probability")
        ax2.grid(False)
        fig.legend(loc="upper right", fontsize=8, frameon=False)
        rows = [("climatology", brier(wet, g.p_climatology)), ("raw (yes/no)", brier(wet, g.p_raw)),
                ("logistic", brier(wet, g.p_logistic)), ("LightGBM", brier(wet, g.p_lightgbm))]
        text = (f"**{len(g)} days, {wet.sum()} wet.** Brier score (lower is better):\n\n| method | Brier |\n|---|---|\n"
                + "\n".join(f"| {m} | {b:.3f} |" for m, b in rows))
    else:
        band = g.dropna(subset=["conformal_low"])
        ax.fill_between(band.date, band.conformal_low, band.conformal_high, color="#2a78d6", alpha=0.15,
                        lw=0, label="80% interval")
        ax.plot(g.date, g.raw, "--", color="#eb6834", lw=1, label=f"raw {model}")
        ax.plot(g.date, g.lightgbm, color="#2a78d6", lw=1.2, label="LightGBM")
        ax.plot(g.date, g.obs, ".", color="black", ms=3, label="observed")
        ax.set_ylabel("°C")
        ax.legend(fontsize=8, frameon=False, ncol=4)
        methods = {"raw": "raw", "bias_corrected": "30-day bias correction", "linear": "linear MOS",
                   "lightgbm": "LightGBM"}
        rows = "\n".join(f"| {label} | {mae(g.obs, g[col]):.2f} |" for col, label in methods.items())
        cover = f"{100 * coverage(band.obs, band.conformal_low, band.conformal_high):.0f}%" if len(band) else "n/a"
        text = (f"**{len(g)} days.** Mean absolute error (°C):\n\n| method | MAE |\n|---|---|\n{rows}\n\n"
                f"Observations inside the 80% interval: {cover}")
    ax.set_title(f"{station}, {model}, {target}, lead {lead} day(s)")
    fig.tight_layout()
    return fig, text


# Try tab
_models = {}


def features():
    path = CLEAN / "features.csv"
    if not path.exists():
        raise gr.Error("data/clean/features.csv not found. Run steps 1-4 first.")
    return table(path, parse_dates=["date", "issue"])


def quantile_models(model, target):
    """LightGBM quantile models (10/50/90%) trained on everything before the test period."""
    if (model, target) not in _models:
        data = features()
        train = data[(data.model == model) & (data.date < TEST_START)].dropna(subset=[f"obs_{target}", f"fc_{target}"])
        error = train[f"obs_{target}"] - train[f"fc_{target}"]
        _models[(model, target)] = [LGBMRegressor(objective="quantile", alpha=q, **LGBM_PARAMS)
                                    .fit(train[TEMP_FEATURES], error) for q in (0.1, 0.5, 0.9)]
    return _models[(model, target)]


def number(value):
    return None if pd.isna(value) else round(float(value), 2)


def load_case(model, station, date, lead):
    data = features()
    try:
        day = pd.Timestamp(date)
    except ValueError:
        raise gr.Error("Date must look like 2026-08-23.") from None
    rows = data[(data.model == model) & (data.station == STATION_IDS[station]) & (data.date == day)
                & (data.lead == int(lead))]
    if rows.empty:
        raise gr.Error(f"No {model} forecast for {station} on {date} at lead {lead}.")
    case = rows.iloc[0].to_dict()
    info = (f"Issued at the end of {case['issue'].date()}. Observed on {day.date()}: "
            f"Tmax {number(case['obs_tmax'])} °C, Tmin {number(case['obs_tmin'])} °C. "
            "Change any value below and press **Correct**.")
    return [case, info] + [number(case[c]) for c in EDITABLE]


def correct(case, *values):
    if not case:
        raise gr.Error("Load a case first.")
    x = pd.DataFrame([{c: case[c] for c in TEMP_FEATURES}], dtype=float)
    for column, value in zip(EDITABLE, values, strict=True):
        x[column] = np.nan if value is None else value

    lines = ["| | raw | corrected | 80% interval | observed |", "|---|---|---|---|---|"]
    for target in ("tmax", "tmin"):
        low, mid, high = sorted(m.predict(x[TEMP_FEATURES])[0] for m in quantile_models(case["model"], target))
        raw = x[f"fc_{target}"].iloc[0]
        lines.append(f"| {target.capitalize()} | {raw:.1f} | **{raw + mid:.1f}** | {raw + low:.1f} to {raw + high:.1f} "
                     f"| {number(case[f'obs_{target}'])} |")
    lines.append("\nModels trained on all data before the test period. The interval is the raw quantile "
                 "interval, without the conformal step.")

    median = quantile_models(case["model"], "tmax")[1]
    contributions = pd.Series(shap.TreeExplainer(median).shap_values(x[TEMP_FEATURES])[0], index=TEMP_FEATURES)
    top = contributions.reindex(contributions.abs().sort_values().index[-8:])
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.barh(top.index, top.values, color=np.where(top.values > 0, "#d6604d", "#2a78d6"))
    ax.axvline(0, color="#888", lw=0.8)
    ax.set_xlabel("°C, relative to the average correction")
    ax.set_title("Why the Tmax correction (SHAP, 8 largest)")
    fig.tight_layout()
    return "\n".join(lines), fig


# Run tab
COMMANDS = {
    "Step 1: observations (SYNOP)": ["-m", "pipeline.step1_observations"],
    "Step 2: archived forecasts": ["-m", "pipeline.step2_forecasts"],
    "Step 3: quality control": ["-m", "pipeline.step3_quality_control"],
    "Step 4: features": ["-m", "pipeline.step4_features"],
    "Step 5: train and predict": ["-m", "pipeline.step5_train_evaluate"],
    "Step 6: report": ["-m", "pipeline.step6_report"],
    "Step 7: SHAP": ["-m", "pipeline.step7_explain"],
    "Step 8: GRU (optional)": ["-m", "pipeline.step8_gru_optional"],
    "Step 9: drought (optional)": ["-m", "pipeline.step9_drought_optional"],
    "Steps 1-7 in order": ["run_all.py"],
    "Steps 3-7 (data already downloaded)": ["run_all.py", "3"],
    "Tests": ["-m", "pytest", "-q"],
    "Build paper PDF": ["paper/make_paper.py"],
}


def run(choice):
    args = COMMANDS[choice]
    log = f"$ python {' '.join(args)}\n"
    yield log
    process = subprocess.Popen([sys.executable, "-u", *args], cwd=ROOT, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                               env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    for line in process.stdout:
        log += line
        yield log[-30000:]
    process.wait()
    yield log[-30000:] + f"\nFinished with exit code {process.returncode}."


def results_summary():
    path = RESULTS / "results.md"
    if not path.exists():
        return missing(path, "step 6")
    return path.read_text(encoding="utf-8").replace("](figures/", "](/gradio_api/file=results/figures/")


def box(title, note, accent=False):
    border = "var(--color-accent)" if accent else "var(--border-color-primary)"
    return (f'<div style="border:1px solid {border};border-radius:8px;padding:6px 10px;margin:3px 0;'
            f'background:var(--block-background-fill);min-width:120px">'
            f'<div style="font-weight:600">{title}</div><div style="font-size:12px;opacity:.75">{note}</div></div>')


def column(*boxes):
    return f'<div style="display:flex;flex-direction:column;justify-content:center">{"".join(boxes)}</div>'


ARROW = '<div style="align-self:center;font-size:20px;opacity:.6;padding:0 4px">&rarr;</div>'

WORKFLOW = (
    '<div style="display:flex;flex-wrap:wrap;align-items:stretch;gap:2px;color:var(--body-text-color)">'
    + column(box("NOAA + Ogimet", "SYNOP station reports"), box("Open-Meteo", "archived IFS, AIFS, GFS runs"))
    + ARROW
    + column(box("1 Observations", "daily Tmax, Tmin, rain"), box("2 Forecasts", "same UTC windows, leads 1-7"))
    + ARROW + column(box("3 Quality control", "range, spikes, stuck values"))
    + ARROW + column(box("4 Features", "only data known at issue time"))
    + ARROW + column(box("5 Walk-forward", "retrain monthly, 8-day gap", accent=True))
    + ARROW
    + column(box("6 Report", "scores, CIs, figures"), box("7 SHAP", "why the corrections"), box("Paper", "paper/main.pdf"))
    + "</div>"
    + '<div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:6px">'
    + box("8 GRU (optional)", "same rows as step 5") + box("9 Drought (optional)", "SPI-3, 1-3 months ahead")
    + "</div>"
)

DETAILS = """
- **Truth**: WMO SYNOP reports. Tmax over 06-18 UTC, Tmin over the 24 h to 06 UTC, rain 06-06 UTC.
- **Forecasts**: for lead L and day D, every hour comes from the run available at the end of day D - L.
- **Temperature**: LightGBM predicts the error of the raw forecast (10/50/90% quantiles); the interval is then
  recalibrated with conformal prediction on the past year.
- **Rain**: calibrated probability of a wet day (>= 1 mm), then the amount on wet days.
- **Testing**: train on data before each month, predict that month; test period 2025-09 to 2026-10;
  95% intervals from resampling whole weeks.
"""


def summary_plot():
    """MAE of each method averaged over leads, per model (Tmax and Tmin)."""
    path = RESULTS / "temperature_mae.csv"
    if not path.exists():
        return None
    mae_table = pd.read_csv(path).groupby(["model", "target"])[
        ["raw", "bias_corrected", "linear", "lightgbm"]].mean()
    labels = {"raw": "raw", "bias_corrected": "30-day bias corr.", "linear": "linear MOS", "lightgbm": "LightGBM"}
    colors = ["#eb6834", "#eda100", "#1baf7a", "#2a78d6"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.1), sharey=True)
    for ax, target in zip(axes, ("tmax", "tmin"), strict=True):
        x = np.arange(len(MODELS))
        for i, (col, label) in enumerate(labels.items()):
            values = [mae_table.loc[(m, target), col] for m in MODELS]
            ax.bar(x + (i - 1.5) * 0.2, values, 0.2, color=colors[i], label=label)
        ax.set_xticks(x, list(MODELS))
        ax.set_title(f"{target.capitalize()}, MAE averaged over leads 1-7")
    axes[0].set_ylabel("°C")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


with gr.Blocks(title="TunCast") as demo:
    gr.Markdown("# TunCast\nCorrection of IFS, AIFS and GFS forecasts at 12 Tunisian stations.")

    with gr.Tab("Results"):
        with gr.Row():
            r_model = gr.Dropdown(list(MODELS), value="IFS", label="Model")
            r_station = gr.Dropdown(STATION_NAMES, value=STATIONS[MAIN_STATION][0], label="Station")
            r_target = gr.Dropdown(["tmax", "tmin", "rain"], value="tmax", label="Variable")
            r_lead = gr.Slider(1, 7, value=1, step=1, label="Lead (days)")
        r_plot = gr.Plot()
        r_text = gr.Markdown()
        inputs = [r_model, r_station, r_target, r_lead]
        for widget in inputs:
            widget.change(show_results, inputs, [r_plot, r_text])
        demo.load(show_results, inputs, [r_plot, r_text])
        with gr.Accordion("All tables (results/results.md)", open=False):
            gr.Markdown(results_summary)

    with gr.Tab("Try the model"):
        gr.Markdown("Load a real forecast from the archive, change the inputs, and see how LightGBM corrects it.")
        with gr.Row():
            t_model = gr.Dropdown(list(MODELS), value="IFS", label="Model")
            t_station = gr.Dropdown(STATION_NAMES, value=STATIONS[MAIN_STATION][0], label="Station")
            t_date = gr.Textbox("2026-08-23", label="Valid day (YYYY-MM-DD)")
            t_lead = gr.Slider(1, 7, value=1, step=1, label="Lead (days)")
        load_button = gr.Button("Load case")
        case_info = gr.Markdown()
        case = gr.State()
        with gr.Row():
            boxes = [gr.Number(label=LABELS[c]) for c in EDITABLE[:5]]
        with gr.Row():
            boxes += [gr.Number(label=LABELS[c]) for c in EDITABLE[5:]]
        correct_button = gr.Button("Correct", variant="primary")
        with gr.Row():
            t_text = gr.Markdown()
            t_plot = gr.Plot()
        load_button.click(load_case, [t_model, t_station, t_date, t_lead], [case, case_info, *boxes])
        correct_button.click(correct, [case, *boxes], [t_text, t_plot])

    with gr.Tab("Run"):
        gr.Markdown("Run a pipeline step. Downloads are cached, so steps 1-2 are fast after the first time.")
        with gr.Row():
            command = gr.Dropdown(list(COMMANDS), value="Tests", label="What to run", scale=3)
            run_button = gr.Button("Run", variant="primary", scale=1)
        log_box = gr.Textbox(label="Output", lines=22, max_lines=22, autoscroll=True)
        run_button.click(run, command, log_box)

    gr.Markdown("## How it works")
    gr.HTML(WORKFLOW)
    with gr.Row():
        gr.Markdown(DETAILS)
        summary = gr.Plot(show_label=False)
    demo.load(summary_plot, None, summary)


if __name__ == "__main__":
    demo.launch(allowed_paths=[str(RESULTS / "figures")])
