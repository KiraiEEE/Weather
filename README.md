# TunMOS

Station-level correction (MOS) of ECMWF IFS, ECMWF AIFS and NCEP GFS forecasts for 12 Tunisian
SYNOP stations, Sfax first. Daily Tmax, Tmin and rain, 1-7 days ahead, with LightGBM, conformal
intervals and SHAP.

Short write-up: [`paper/main.pdf`](paper/main.pdf). Full tables: [`results/results.md`](results/results.md).

## Results (test period 2025-09-01 to 2026-10-25, 12 stations)

Tmax MAE in °C, 1 day ahead:

| model | raw | LightGBM |
|---|---|---|
| IFS  | 1.74 | 0.89 |
| AIFS | 1.80 | 0.90 |
| GFS  | 1.53 | 0.93 |

Most of the improvement is bias removal. LightGBM beats linear MOS by 2-4% for IFS and GFS;
for AIFS the difference is not significant. Rain probabilities have a Brier skill of about 0.2
against climatology (raw forecasts are negative). Heavy rain is still poor.

## Setup

Python 3.11, tested on Windows.

```bash
uv venv .venv --python 3.11          # or python -m venv .venv
uv pip install -r requirements.txt --python .venv/Scripts/python.exe
```

## Run

```bash
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe run_all.py            # steps 1-7, about 1 h the first time (downloads)
.venv/Scripts/python.exe run_all.py 4          # start from step 4
.venv/Scripts/python.exe -m pipeline.step6_report

# optional
uv pip install -r requirements-gru.txt --python .venv/Scripts/python.exe
.venv/Scripts/python.exe -m pipeline.step8_gru_optional
.venv/Scripts/python.exe -m pipeline.step9_drought_optional

# paper (needs pdflatex)
.venv/Scripts/python.exe paper/make_paper.py
```

Downloads are cached in `data/raw/`. If Open-Meteo or Ogimet rate-limits you, run the step again
later and it picks up where it stopped. The Open-Meteo free tier allows ~10k calls a day, so step 2
may need two days on a first run.

## Layout

```
pipeline/config.py               settings
pipeline/utils.py                cached download, scores
pipeline/step1_observations.py   SYNOP -> daily Tmax/Tmin/rain (NOAA + Ogimet)
pipeline/step2_forecasts.py      archived IFS/AIFS/GFS forecasts
pipeline/step3_quality_control.py
pipeline/step4_features.py
pipeline/step5_train_evaluate.py monthly walk-forward
pipeline/step6_report.py         results/results.md, figures
pipeline/step7_explain.py        SHAP
pipeline/step8_gru_optional.py   GRU vs LightGBM
pipeline/step9_drought_optional.py  SPI-3 outlook
tests/test_pipeline.py
paper/                           LaTeX source + make_paper.py
```

## Data

Open-Meteo (CC BY 4.0), NOAA NCEI Global-Hourly, SYNOP via Ogimet, NAO index from NOAA CPC.
