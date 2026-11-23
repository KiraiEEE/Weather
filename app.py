"""TunCast app: look at the corrected forecasts station by station and day by day.

    python app.py        then open http://127.0.0.1:7860
"""
import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import subprocess  # noqa: E402
import sys  # noqa: E402

import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import shap  # noqa: E402
from lightgbm import LGBMRegressor  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from pipeline.config import (CLEAN, LGBM_PARAMS, MAIN_STATION, MODELS, RESULTS, ROOT, STATIONS,  # noqa: E402
                             TEST_END, TEST_START)
from pipeline.step4_features import TEMP_FEATURES  # noqa: E402
from pipeline.utils import mae  # noqa: E402

INK, BRICK, OCHRE, SAGE, GRID, PAPER = "#1f3a5f", "#c8553d", "#b08d57", "#5f7f6e", "#e7e2d8", "#fffdf8"
FONT = "IBM Plex Sans, Helvetica, Arial, sans-serif"
NAMES = {wmo: name for wmo, (name, *_rest) in STATIONS.items()}
IDS = {name: wmo for wmo, name in NAMES.items()}
LABEL_SIDE = {"Tunis": "middle left", "Gabes": "middle left", "Sfax": "middle left", "Enfidha": "middle left"}
DRIVERS = {"fc_tmax": "the forecast Tmax itself", "fc_tmin": "the forecast Tmin", "fc_dew": "the forecast dew point",
           "fc_cloud": "the forecast cloud cover", "fc_wind": "the forecast wind",
           "bias30_tmax": "the Tmax error of the last 30 days", "bias30_tmin": "the Tmin error of the last 30 days",
           "clim_tmax": "the usual Tmax for the date", "clim_tmin": "the usual Tmin for the date",
           "last_tmax": "the last observed Tmax", "last_tmin": "the last observed Tmin", "lead": "the lead time",
           "doy_sin": "the time of year", "doy_cos": "the time of year", "lat": "the station latitude",
           "lon": "the station longitude", "elev": "the station elevation"}


# data
_cache = {}


def table(name, folder=RESULTS, dates=("date",)):
    path = folder / name
    if not path.exists():
        raise gr.Error(f"{path.relative_to(ROOT)} is missing. Run the pipeline first (bottom of the page).")
    mtime = path.stat().st_mtime
    if name not in _cache or _cache[name][0] != mtime:
        _cache[name] = (mtime, pd.read_csv(path, dtype={"station": str}, parse_dates=list(dates)))
    return _cache[name][1]


def temperature():
    t = table("predictions_temperature.csv")
    return t[(t.date >= TEST_START) & (t.date <= TEST_END)]


def rain():
    r = table("predictions_rain.csv")
    return r[(r.date >= TEST_START) & (r.date <= TEST_END)]


def style(fig, height):
    fig.update_layout(height=height, margin=dict(l=50, r=20, t=30, b=40), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor=PAPER, font=dict(family=FONT, size=12, color="#2b2b2b"),
                      legend=dict(orientation="h", y=-0.22, x=0, font=dict(size=11)), hoverlabel=dict(font_family=FONT))
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor="#bdb6a6")
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor="#bdb6a6")
    return fig


# station map
def station_scores(model):
    t = temperature()
    t = t[(t.model == model) & (t.lead == 1) & (t.target == "tmax")].dropna(subset=["obs", "raw", "lightgbm"])
    rows = []
    for wmo, g in t.groupby("station"):
        raw, corrected = mae(g.obs, g.raw), mae(g.obs, g.lightgbm)
        rows.append({"station": wmo, "raw": raw, "corrected": corrected, "gain": 100 * (1 - corrected / raw)})
    return pd.DataFrame(rows).set_index("station")


def station_map(model, station):
    scores = station_scores(model)
    wmos = list(STATIONS)
    lat = [STATIONS[w][1] for w in wmos]
    lon = [STATIONS[w][2] for w in wmos]
    gain = [scores.gain.get(w, np.nan) for w in wmos]
    hover = [f"<b>{NAMES[w]}</b><br>Tmax day 1, {model}<br>raw {scores.raw.get(w, np.nan):.2f} °C → "
             f"corrected {scores.corrected.get(w, np.nan):.2f} °C" for w in wmos]
    fig = go.Figure()
    fig.add_trace(go.Scattergeo(
        lat=lat, lon=lon, text=[NAMES[w] for w in wmos], hovertext=hover, hoverinfo="text", mode="markers+text",
        textposition=[LABEL_SIDE.get(NAMES[w], "middle right") for w in wmos],
        textfont=dict(family=FONT, size=11, color="#2b2b2b"),
        marker=dict(size=13, color=gain, colorscale=[[0, "#e8dcc4"], [1, INK]], cmin=10, cmax=65,
                    line=dict(width=0.6, color="#555"),
                    colorbar=dict(title=dict(text="error cut %", font=dict(size=11)), thickness=8, len=0.5, x=0.92,
                                  tickfont=dict(size=10)))))
    wmo = IDS[station]
    fig.add_trace(go.Scattergeo(lat=[STATIONS[wmo][1]], lon=[STATIONS[wmo][2]], mode="markers", hoverinfo="skip",
                                marker=dict(size=24, color="rgba(0,0,0,0)", line=dict(width=2, color=BRICK))))
    fig.update_geos(lonaxis_range=[7.4, 12.0], lataxis_range=[32.8, 37.7], showcountries=True,
                    countrycolor="#bdb6a6", showcoastlines=True, coastlinecolor="#8c8473", showland=True,
                    landcolor="#f1ece1", showocean=True, oceancolor="#dfe7ea", resolution=50, bgcolor="rgba(0,0,0,0)")
    fig.update_layout(height=430, margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(family=FONT))
    return fig


def station_card(model, station):
    wmo = IDS[station]
    _name, lat, lon, elev = STATIONS[wmo]
    t = temperature()
    t = t[(t.model == model) & (t.station == wmo) & (t.lead == 1)].dropna(subset=["obs", "raw", "lightgbm"])
    lines = [f"WMO {wmo} · {lat:.2f}° N {lon:.2f}° E · {elev} m", ""]
    for target in ("tmax", "tmin"):
        g = t[t.target == target]
        if len(g):
            lines.append(f"{target.capitalize()} day 1   raw {mae(g.obs, g.raw):.2f} °C   corrected "
                         f"{mae(g.obs, g.lightgbm):.2f} °C   ({len(g)} days)")
    return "```\n" + "\n".join(lines) + "\n```"


# dProg/dt for one day
def notable_day(model, station, kind):
    t = temperature()
    t = t[(t.model == model) & (t.station == IDS[station]) & (t.lead == 1)]
    if kind == "hottest":
        g = t[t.target == "tmax"].dropna(subset=["obs"])
        day = g.loc[g.obs.idxmax(), "date"]
    elif kind == "worst":
        g = t[t.target == "tmax"].dropna(subset=["obs", "raw"])
        day = g.loc[(g.obs - g.raw).abs().idxmax(), "date"]
    else:
        r = rain()
        g = r[(r.model == model) & (r.station == IDS[station]) & (r.lead == 1)].dropna(subset=["obs"])
        day = g.loc[g.obs.idxmax(), "date"]
    return day.strftime("%Y-%m-%d")


def parse_day(text):
    try:
        day = pd.Timestamp(text)
    except ValueError:
        raise gr.Error("Write the day as YYYY-MM-DD, e.g. 2026-08-23.") from None
    if not pd.Timestamp(TEST_START) <= day <= pd.Timestamp(TEST_END):
        raise gr.Error(f"Pick a day between {TEST_START} and {TEST_END}.")
    return day


def dprog(model, station, day_text):
    day = parse_day(day_text)
    t = temperature()
    t = t[(t.model == model) & (t.station == IDS[station]) & (t.date == day)]
    r = rain()
    r = r[(r.model == model) & (r.station == IDS[station]) & (r.date == day)].sort_values("lead")
    obs_rain = r.obs.dropna()
    rain_title = "P(rain ≥ 1 mm)" + (f" · observed {obs_rain.iloc[0]:.1f} mm" if len(obs_rain) else "")
    fig = make_subplots(rows=1, cols=3, column_widths=[0.38, 0.38, 0.24], horizontal_spacing=0.07,
                        subplot_titles=("Tmax (°C)", "Tmin (°C)", rain_title))
    for col, target in ((1, "tmax"), (2, "tmin")):
        g = t[t.target == target].sort_values("lead")
        if g.empty:
            continue
        fig.add_trace(go.Scatter(x=g.lead, y=g.raw, mode="lines+markers", name=f"raw {model}",
                                 line=dict(color=BRICK, width=1.2, dash="dot"),
                                 marker=dict(size=8, symbol="circle-open", line=dict(width=1.6)),
                                 showlegend=col == 1), row=1, col=col)
        err = g.dropna(subset=["conformal_low"])
        fig.add_trace(go.Scatter(x=g.lead, y=g.lightgbm, mode="lines+markers", name="corrected (80% interval)",
                                 line=dict(color=INK, width=1.8), marker=dict(size=8, color=INK),
                                 error_y=dict(type="data", symmetric=False, color="rgba(31,58,95,0.45)", thickness=1.2,
                                              array=(err.conformal_high - err.lightgbm).reindex(g.index).to_numpy(),
                                              arrayminus=(err.lightgbm - err.conformal_low).reindex(g.index).to_numpy()),
                                 showlegend=col == 1), row=1, col=col)
        observed = g.obs.dropna()
        if len(observed):
            fig.add_hline(y=observed.iloc[0], line=dict(color="#222", width=1.2, dash="dash"), row=1, col=col,
                          annotation_text=f"observed {observed.iloc[0]:.1f}", annotation_position="top left",
                          annotation_font=dict(size=11))
    if len(r):
        fig.add_trace(go.Bar(x=r.lead, y=r.p_lightgbm, marker_color=INK, name="P(rain), corrected",
                             showlegend=False, hovertemplate="lead %{x}: %{y:.2f}<extra></extra>"), row=1, col=3)
        fig.add_trace(go.Scatter(x=r.lead, y=r.p_raw, mode="markers", marker=dict(symbol="line-ew-open", size=18,
                                 color=BRICK, line=dict(width=2)), name="raw says rain (1) / dry (0)",
                                 showlegend=False, hovertemplate="raw: %{y:.0f}<extra></extra>"), row=1, col=3)
    for col in (1, 2, 3):
        fig.update_xaxes(autorange="reversed", dtick=1, title_text="issued … days before", row=1, col=col)
    fig.update_yaxes(range=[0, 1.05], row=1, col=3)
    style(fig, 380)
    fig.update_annotations(font=dict(family=FONT, size=12))
    caption = (f"*Forecasts for {station} on {day:%d %B %Y}, issued 7 days before (left of each panel) to "
               f"1 day before (right). Open circles: raw {model}. Filled: after correction, with the conformal 80% "
               "interval. Dashed: what the station measured. Right panel: corrected rain probability (bars) and "
               "the raw yes/no forecast (red ticks).*")
    return fig, caption


# error by lead at one station
def error_by_lead(model, station):
    t = temperature()
    all_rows = t[t.model == model]
    t = all_rows[all_rows.station == IDS[station]]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Tmax", "Tmin"), horizontal_spacing=0.08, shared_yaxes=True)
    methods = [("raw", "raw", BRICK, "dot"), ("bias_corrected", "30-day bias correction", OCHRE, "solid"),
               ("linear", "linear MOS", SAGE, "solid"), ("lightgbm", "LightGBM", INK, "solid")]
    for col, target in ((1, "tmax"), (2, "tmin")):
        g = t[t.target == target].dropna(subset=["obs", "raw", "lightgbm", "bias_corrected", "linear"])
        for column, label, color, dash in methods:
            by_lead = g.groupby("lead").apply(lambda d, c=column: mae(d.obs, d[c]), include_groups=False)
            fig.add_trace(go.Scatter(x=by_lead.index, y=by_lead.values, mode="lines+markers", name=label,
                                     line=dict(color=color, width=2 if column == "lightgbm" else 1.4, dash=dash),
                                     marker=dict(size=5), showlegend=col == 1), row=1, col=col)
        everyone = all_rows[all_rows.target == target].dropna(subset=["obs", "lightgbm"])
        by_lead = everyone.groupby("lead").apply(lambda d: mae(d.obs, d.lightgbm), include_groups=False)
        fig.add_trace(go.Scatter(x=by_lead.index, y=by_lead.values, mode="lines", name="LightGBM, all 12 stations",
                                 line=dict(color="#9a9486", width=1, dash="dash"), showlegend=col == 1), row=1, col=col)
        fig.update_xaxes(dtick=1, title_text="lead (days)", row=1, col=col)
    fig.update_yaxes(title_text="mean absolute error (°C)", row=1, col=1)
    style(fig, 340)
    fig.update_annotations(font=dict(family=FONT, size=12))
    return fig


# sensitivity: move the inputs
_models = {}


def quantile_models(model, target):
    if (model, target) not in _models:
        data = table("features.csv", CLEAN, dates=("date", "issue"))
        train = data[(data.model == model) & (data.date < TEST_START)].dropna(subset=[f"obs_{target}", f"fc_{target}"])
        error = train[f"obs_{target}"] - train[f"fc_{target}"]
        _models[(model, target)] = [LGBMRegressor(objective="quantile", alpha=q, **LGBM_PARAMS)
                                    .fit(train[TEMP_FEATURES], error) for q in (0.1, 0.5, 0.9)]
    return _models[(model, target)]


def load_case(model, station, day_text, lead):
    day = parse_day(day_text)
    data = table("features.csv", CLEAN, dates=("date", "issue"))
    rows = data[(data.model == model) & (data.station == IDS[station]) & (data.date == day) & (data.lead == int(lead))]
    if rows.empty:
        raise gr.Error(f"No {model} forecast for {station} on {day_text} at lead {lead}.")
    case = rows.iloc[0].to_dict()
    tmax = case["fc_tmax"]
    sliders = [
        gr.Slider(minimum=round(tmax - 10), maximum=round(tmax + 10), value=round(tmax, 1)),
        gr.Slider(value=round(case["fc_wind"], 1)),
        gr.Slider(value=round(case["fc_cloud"])),
        gr.Slider(value=0.0 if pd.isna(case["bias30_tmax"]) else round(case["bias30_tmax"], 2)),
    ]
    return [case, *sliders, *explain(case, tmax, case["fc_wind"], case["fc_cloud"], case["bias30_tmax"])]


def explain(case, fc_tmax, fc_wind, fc_cloud, bias30):
    if not case:
        return None, ""
    x = pd.DataFrame([{c: case[c] for c in TEMP_FEATURES}], dtype=float)
    x["fc_tmax"], x["fc_wind"], x["fc_cloud"], x["bias30_tmax"] = fc_tmax, fc_wind, fc_cloud, bias30
    models = quantile_models(case["model"], "tmax")
    low, mid, high = sorted(fc_tmax + m.predict(x[TEMP_FEATURES])[0] for m in models)
    observed = case["obs_tmax"]

    fig = go.Figure()
    fig.add_shape(type="line", x0=low, x1=high, y0=0, y1=0, line=dict(color="rgba(31,58,95,0.35)", width=14))
    fig.add_trace(go.Scatter(x=[fc_tmax], y=[0], mode="markers+text", text=[f"raw {fc_tmax:.1f}"],
                             textposition="top center", marker=dict(size=13, symbol="circle-open", color=BRICK,
                                                                    line=dict(width=2)), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[mid], y=[0], mode="markers+text", text=[f"corrected {mid:.1f}"],
                             textposition="bottom center", marker=dict(size=13, color=INK), hoverinfo="skip"))
    if not pd.isna(observed):
        fig.add_vline(x=observed, line=dict(color="#222", width=1.2, dash="dash"),
                      annotation_text=f"observed {observed:.1f}", annotation_position="top")
    values = [low, high, fc_tmax] + ([] if pd.isna(observed) else [observed])
    fig.update_xaxes(range=[min(values) - 1.5, max(values) + 1.5], title_text="Tmax (°C)")
    fig.update_yaxes(visible=False, range=[-1, 1])
    style(fig, 190)
    fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=30, b=45))

    contributions = pd.Series(shap.TreeExplainer(models[1]).shap_values(x[TEMP_FEATURES])[0], index=TEMP_FEATURES)
    top = contributions.reindex(contributions.abs().sort_values(ascending=False).index[:3])
    base = float(np.ravel(shap.TreeExplainer(models[1]).expected_value)[0])
    parts = [f"{DRIVERS[name]} {'adds' if value > 0 else 'removes'} **{abs(value):.1f} °C**" for name, value in top.items()]
    text = (f"On average the model adds {base:+.1f} °C to this model's Tmax. For this case, "
            + "; ".join(parts) + f". Net correction: **{mid - fc_tmax:+.1f} °C**, 80% range {low:.1f} to {high:.1f} °C."
            "\n\n<span class='note'>Models here are trained once on all data before the test period, so numbers can "
            "differ slightly from the month-by-month runs in the paper. The interval is before the conformal step.</span>")
    return fig, text


# timing figure
def timing_figure():
    """What is known when a lead-2 forecast is issued, and which UTC hours make up each daily value."""
    fig = go.Figure()
    days = ["I − 1", "I  (issue day)", "I + 1", "D  (valid day)", "D + 1"]
    for i, label in enumerate(days):
        fig.add_vrect(x0=24 * i, x1=24 * (i + 1), fillcolor="#f1ece1" if i % 2 else PAPER, line_width=0, layer="below")
        fig.add_annotation(x=24 * i + 12, y=4.75, text=label, showarrow=False, font=dict(size=11, color="#555"))
    fig.add_vrect(x0=0, x1=48, fillcolor="rgba(31,58,95,0.07)", line_width=0, layer="below")
    fig.add_vline(x=48, line=dict(color=INK, width=2))
    fig.add_annotation(x=48, y=4.3, text="forecast issued (end of I)", showarrow=False, xanchor="left",
                       xshift=4, font=dict(size=11, color=INK))
    rows = [  # (y, label, start hour, end hour, colour)
        (3.4, "Tmax(I) known · 06-18 UTC on I", 30, 42, "#7d8ea3"),
        (2.8, "Rain(I) not complete until 06 UTC on I+1", 30, 54, "#c9b48d"),
        (1.9, "Tmin(D) · 24 h ending 06 UTC on D", 54, 78, SAGE),
        (1.3, "Tmax(D) · 06-18 UTC on D", 78, 90, BRICK),
        (0.7, "Rain(D) · 06 UTC on D to 06 UTC on D+1", 78, 102, OCHRE),
    ]
    for y, label, start, end, color in rows:
        fig.add_shape(type="line", x0=start, x1=end, y0=y, y1=y, line=dict(color=color, width=9))
        fig.add_annotation(x=end, y=y, text=label, showarrow=False, xanchor="left", xshift=6, font=dict(size=11))
    fig.add_annotation(x=78, y=0.05, ax=48, ay=0.05, xref="x", yref="y", axref="x", ayref="y", text="",
                       showarrow=True, arrowhead=2, startarrowhead=2, arrowside="end+start", arrowcolor="#555")
    fig.add_annotation(x=63, y=0.05, text="lead L = D − I = 2 days", showarrow=False, yshift=10, font=dict(size=11))
    fig.update_xaxes(range=[0, 138], tickvals=list(range(0, 121, 6)),
                     ticktext=[f"{h % 24:02d}" for h in range(0, 121, 6)], title_text="hour (UTC)", showgrid=False)
    fig.update_yaxes(visible=False, range=[-0.3, 5.1])
    style(fig, 300)
    fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=40))
    return fig


# re-run
COMMANDS = {
    "Steps 3-7 (data already downloaded)": ["run_all.py", "3"],
    "Steps 1-7 (downloads everything first)": ["run_all.py"],
    "Step 5 only: retrain and predict": ["-m", "pipeline.step5_train_evaluate"],
    "Step 6 only: tables and figures": ["-m", "pipeline.step6_report"],
    "Step 7 only: SHAP": ["-m", "pipeline.step7_explain"],
    "Step 8: GRU comparison": ["-m", "pipeline.step8_gru_optional"],
    "Step 9: drought outlook": ["-m", "pipeline.step9_drought_optional"],
    "Tests": ["-m", "pytest", "-q"],
    "Rebuild the paper PDF": ["paper/make_paper.py"],
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
    yield log[-30000:] + f"\nexit code {process.returncode}"


# page
THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.slate, neutral_hue=gr.themes.colors.stone,
    font=[gr.themes.GoogleFont("IBM Plex Sans"), "Helvetica", "Arial", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "Consolas", "monospace"],
    radius_size=gr.themes.sizes.radius_sm,
).set(
    body_background_fill="#f6f3ec", block_background_fill=PAPER, block_border_color="#e2dccf",
    block_shadow="none", block_label_text_color="#6b6455", input_background_fill="#ffffff",
    button_primary_background_fill=INK, button_primary_background_fill_hover="#2b4c78",
    button_primary_text_color="#ffffff", button_secondary_background_fill="#ece6da",
    button_secondary_text_color="#2b2b2b", slider_color=INK, color_accent=INK, color_accent_soft="#dfe5ee",
)

CSS = """
.gradio-container {max-width: 1180px !important; margin: auto;}
.masthead h1 {font-weight: 600; letter-spacing: -0.01em; margin-bottom: 0;}
.masthead p {color: #5c5648; margin-top: 4px;}
.section h2 {font-size: 15px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em;
             color: #5c5648; border-top: 1px solid #d8d1c2; padding-top: 18px; margin-top: 26px;}
.caption p, .caption em {color: #5c5648; font-size: 13px; line-height: 1.45;}
.note {color: #8a8372; font-size: 12px;}
footer {display: none !important;}
"""

JS = """() => { const u = new URL(window.location); if (u.searchParams.get('__theme') !== 'light') {
  u.searchParams.set('__theme', 'light'); window.location.replace(u.href); } }"""

with gr.Blocks(title="TunCast") as demo:
    gr.Markdown(
        "# TunCast\nDaily Tmax, Tmin and rain forecasts from IFS, AIFS and GFS at 12 Tunisian SYNOP stations, "
        f"corrected with LightGBM. Test period {TEST_START} to {TEST_END}; every forecast shown here was made "
        "by a model that had not seen that month.", elem_classes="masthead")

    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            map_plot = gr.Plot(show_label=False)
        with gr.Column(scale=4):
            station = gr.Radio(list(IDS), value=NAMES[MAIN_STATION], label="Station")
            model = gr.Radio(list(MODELS), value="IFS", label="Global model")
            card = gr.Markdown()
            gr.Markdown("*Map colour: how much of the day-1 Tmax error the correction removes at each station.*",
                        elem_classes="caption")

    gr.Markdown("## One day, seven forecasts", elem_classes="section")
    with gr.Row():
        day = gr.Textbox("2026-08-23", label="Day (YYYY-MM-DD)", scale=2)
        hottest = gr.Button("Hottest day", variant="secondary", scale=1)
        wettest = gr.Button("Wettest day", variant="secondary", scale=1)
        worst = gr.Button("Worst raw forecast", variant="secondary", scale=1)
    dprog_plot = gr.Plot(show_label=False)
    dprog_caption = gr.Markdown(elem_classes="caption")

    gr.Markdown("## Error by lead time at this station", elem_classes="section")
    lead_plot = gr.Plot(show_label=False)
    gr.Markdown("*Mean absolute error over the test period. The grey dashed line is LightGBM over all 12 stations, "
                "for comparison. A 30-day running bias correction already gets most of the way.*",
                elem_classes="caption")

    gr.Markdown("## Move the inputs", elem_classes="section")
    gr.Markdown("Takes the forecast for the day above and lets you change it. The correction is recomputed when "
                "you let go of a slider.", elem_classes="caption")
    case = gr.State()
    with gr.Row():
        lead = gr.Slider(1, 7, value=1, step=1, label="Lead (days)")
        s_tmax = gr.Slider(0, 50, value=30, step=0.1, label="Forecast Tmax (°C)")
        s_wind = gr.Slider(0, 60, value=10, step=0.5, label="Forecast wind (km/h)")
        s_cloud = gr.Slider(0, 100, value=20, step=1, label="Forecast cloud cover (%)")
        s_bias = gr.Slider(-6, 6, value=0, step=0.05, label="Tmax error, last 30 days (°C)")
    with gr.Row(equal_height=False):
        with gr.Column(scale=3):
            line_plot = gr.Plot(show_label=False)
        with gr.Column(scale=2):
            why = gr.Markdown()

    gr.Markdown("## How a forecast is dated", elem_classes="section")
    gr.Plot(timing_figure(), show_label=False)
    gr.Markdown(
        "*A forecast for day D at lead L is treated as issued at the end of day I = D − L. It may use the Tmax "
        "and Tmin observed on I and the rain of I − 1 (the rain of I only closes at 06 UTC the next morning), "
        "and only model runs that existed by then. Forecast hours are added up over the same UTC windows as the "
        "station values. Models are retrained every month on data that ends 8 days before the month starts.*",
        elem_classes="caption")

    with gr.Accordion("Re-run the experiment", open=False):
        with gr.Row():
            command = gr.Dropdown(list(COMMANDS), value="Tests", show_label=False, scale=4)
            run_button = gr.Button("Run", variant="primary", scale=1)
        log = gr.Code(language="shell", lines=16, show_label=False)
        run_button.click(run, command, log)

    # wiring
    top = [model, station]
    for widget in top:
        widget.change(station_map, top, map_plot)
        widget.change(station_card, top, card)
        widget.change(error_by_lead, top, lead_plot)
        widget.change(dprog, [model, station, day], [dprog_plot, dprog_caption])
    day.submit(dprog, [model, station, day], [dprog_plot, dprog_caption])
    for button, kind in ((hottest, "hottest"), (wettest, "wettest"), (worst, "worst")):
        button.click(lambda m, s, k=kind: notable_day(m, s, k), top, day).then(
            dprog, [model, station, day], [dprog_plot, dprog_caption])

    case_inputs = [model, station, day, lead]
    case_outputs = [case, s_tmax, s_wind, s_cloud, s_bias, line_plot, why]
    dprog_plot.change(load_case, case_inputs, case_outputs)
    lead.release(load_case, case_inputs, case_outputs)
    for slider in (s_tmax, s_wind, s_cloud, s_bias):
        slider.release(explain, [case, s_tmax, s_wind, s_cloud, s_bias], [line_plot, why])

    demo.load(station_map, top, map_plot).then(station_card, top, card).then(
        error_by_lead, top, lead_plot).then(dprog, [model, station, day], [dprog_plot, dprog_caption])


if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS, js=JS)
