# Plan

Correct IFS, AIFS and GFS forecasts for Sfax and 11 other Tunisian stations (Tmax, Tmin, rain,
1-7 days ahead), and check against simple baselines whether the correction is worth it.

## Changes from the first plan

Things that turned out wrong once I looked at the data:

- Meteostat rain for Tunisian stations is DWD MOSMIX, i.e. a forecast. Truth is now taken from
  SYNOP reports (NOAA Global-Hourly to 2025-08-24, Ogimet after).
- ERA5 is a reanalysis, not observations. Only used for the drought part (step 9).
- Archived forecasts start 2024-02 (IFS, GFS) and 2025-02 (AIFS), so a fixed 2023-24 test set
  is impossible. Using monthly walk-forward instead; test period 2025-09 to 2026-10.
- The archive API is hourly and only goes 7 days back. Daily values are built here, rain up to lead 6.
- Station days are not calendar days (Tmax 06-18 UTC, Tmin 24 h to 06 UTC, rain 06-06 UTC).
  Forecasts use the same windows.
- Only data up to the issue day can be used as input. There is a test for this.
- Added 30-day bias correction and linear MOS as baselines; beating climatology is too easy.
- Sfax has ~25 rain days a year, too few on its own, so all 12 stations go into one model.
- No make/conda on this laptop: plain venv + run_all.py.

Two things in the NOAA data:
- NOAA flags most rain reports as "suspect" (two 24 h values in one message). Dropping them loses ~70% of rain.
- A 24 h total reported at 06 UTC belongs to the previous day.

## Data

| what | source | period |
|---|---|---|
| Tmax, Tmin, rain | NOAA Global-Hourly + Ogimet SYNOP | 2014 - now |
| forecasts, leads 1-7 | Open-Meteo Previous-Runs API | 2024-02 / 2025-02 - now |
| drought (step 9) | ERA5 monthly rain + NAO (NOAA CPC) | 1991 - now |

## Methods

- Temperature: raw, climatology, persistence, 30-day bias correction, linear regression,
  quantile LightGBM on the forecast error, conformal 80% intervals.
- Rain: P(>= 1 mm) with LightGBM + isotonic calibration, amount on wet days. Baselines climatology,
  raw, logistic regression.
- SHAP summary plots, Sfax hottest/wettest day.
- Optional: GRU vs LightGBM (step 8), SPI-3 1-3 months ahead (step 9).

## Evaluation

- Monthly walk-forward, 8-day gap before each test month. No random splits.
- All methods on the same rows.
- Temperature: MAE, gain vs raw with weekly block-bootstrap 95% CI, interval coverage.
- Rain: Brier score/skill, ROC AUC, PR AUC, reliability, heavy rain (>= 10 mm) hits and false alarms.
- Parameters fixed in config.py before looking at test scores.

## Targets

1. `python run_all.py` rebuilds everything. - done
2. Tmax: LightGBM beats raw IFS and AIFS at >= 5 of 7 leads, with CIs. - all 7 leads, all 3 models
3. Rain Brier skill > 0 vs climatology. - 0.19-0.21
4. Conformal coverage 77-83%. - 80.2-83.5%, AIFS tmin slightly over
5. Tests pass. - done

## Risks

- Open-Meteo daily limit (~10k calls): cached, rerun step 2 the next day.
- Ogimet slow: 8 s pause, retries, cache.
- AIFS only ~20 months: pool stations, report CIs.
- Few heavy-rain days: pool stations, always give the count.

## Status (2026-11-10)

Steps 1-9 have run. Results in `results/`, write-up in `paper/main.pdf`.
