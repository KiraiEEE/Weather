"""Archived IFS / AIFS / GFS forecasts from the Open-Meteo Previous-Runs API.

The API is hourly only. `<var>_previous_dayK` holds, for each hour, the run started K days earlier.
For lead L and valid day D the run has to exist at the end of the issue day I = D - L,
so each hour uses K = (day of that hour) - I.

Daily windows match the observations:
  fc_tmax(D)  max 06-18 on D
  fc_tmin(D)  min of 07-23 on D-1 and 00-06 on D
  fc_prcp(D)  sum of 07-23 on D and 00-06 on D+1 (needs K = L+1, so rain stops at lead 6)
  others      mean 00-23 on D

Writes data/clean/forecasts.csv
"""
import io
import time

import numpy as np
import pandas as pd

from pipeline.config import CLEAN, FORECAST_END, MAX_LEAD_RAIN, MAX_LEAD_TEMP, MODELS, STATIONS
from pipeline.utils import download

API = "https://previous-runs-api.open-meteo.com/v1/forecast"
VARIABLES = ["temperature_2m", "precipitation", "dew_point_2m", "cloud_cover", "wind_speed_10m"]
SHORT_NAMES = {"dew_point_2m": "fc_dew", "cloud_cover": "fc_cloud", "wind_speed_10m": "fc_wind"}


def hourly_forecasts(lat, lon, model, start):
    """Download hourly forecasts in 60-day pieces and join them into one table.
    Open-Meteo's free limit is about 5000 "units" per hour and a piece costs ~15.5 units, so each real download
    takes at least 11.5 s in total (about 4800 units per hour). Bigger pieces made the server time out."""
    columns = ["temperature_2m"]          # day 0 (latest run): needed for the lead-1 evening of D-1
    for k in range(1, 8):
        columns += [f"{v}_previous_day{k}" for v in VARIABLES]
    pieces = []
    for piece_start in pd.date_range(start, FORECAST_END, freq="60D"):
        piece_end = min(piece_start + pd.Timedelta(days=59), pd.Timestamp(FORECAST_END))
        params = {"latitude": lat, "longitude": lon, "models": model, "hourly": ",".join(columns),
                  "start_date": piece_start.strftime("%Y-%m-%d"), "end_date": piece_end.strftime("%Y-%m-%d"),
                  "timezone": "GMT", "format": "csv"}
        started = time.time()
        content = download(API, params, check=lambda c: c.startswith(b"latitude"), tries=8)
        table = pd.read_csv(io.BytesIO(content), skiprows=3)      # first 3 lines describe the location
        table.columns = [c.split(" (")[0] for c in table.columns]  # "temperature_2m (°C)" -> "temperature_2m"
        pieces.append(table)
        elapsed = time.time() - started
        if elapsed > 0.5:                     # real download, not the cache
            time.sleep(max(0.0, 11.5 - elapsed))
    hourly = pd.concat(pieces)
    hourly.index = pd.to_datetime(hourly.pop("time"))
    return hourly


def column(hourly, variable, k):
    name = variable if k == 0 else f"{variable}_previous_day{k}"
    return hourly[name] if name in hourly else pd.Series(np.nan, index=hourly.index)


def daily(series, hours, how):
    """Aggregate the chosen UTC hours of each day; NaN if one of those hours is missing."""
    part = series[series.index.hour.isin(list(hours))]
    groups = part.groupby(part.index.normalize())
    return groups.agg(how).where(groups.count() == len(hours))


def daily_forecasts(hourly):
    one_day = pd.Timedelta(days=1)
    rows = []
    for lead in range(1, MAX_LEAD_TEMP + 1):
        temp_today = column(hourly, "temperature_2m", lead)
        temp_yesterday = column(hourly, "temperature_2m", lead - 1)
        evening_before = daily(temp_yesterday, range(7, 24), "min")
        evening_before.index = evening_before.index + one_day         # evening of D-1 belongs to day D
        night = daily(temp_today, range(0, 7), "min")
        table = pd.DataFrame({
            "fc_tmax": daily(temp_today, range(6, 19), "max"),
            "fc_tmin": pd.concat([evening_before, night], axis=1, sort=True).min(axis=1, skipna=False),
        })
        if lead <= MAX_LEAD_RAIN:
            day_rain = daily(column(hourly, "precipitation", lead), range(7, 24), "sum")
            night_rain = daily(column(hourly, "precipitation", lead + 1), range(0, 7), "sum")
            night_rain.index = night_rain.index - one_day             # early hours of D+1 belong to day D
            table["fc_prcp"] = (day_rain + night_rain).clip(lower=0)
        for variable, short in SHORT_NAMES.items():
            table[short] = daily(column(hourly, variable, lead), range(0, 24), "mean")
        table["lead"] = lead
        rows.append(table)
    result = pd.concat(rows)
    result.index.name = "date"
    return result.reset_index().dropna(subset=["fc_tmax", "fc_tmin"], how="all")


def main():
    tables = []
    for station, (name, lat, lon, _elev) in STATIONS.items():
        for label, (model, start) in MODELS.items():
            print(f"{name} {label}")
            hourly = hourly_forecasts(lat, lon, model, start)
            tables.append(daily_forecasts(hourly).assign(station=station, model=label))
    forecasts = pd.concat(tables)
    first = ["station", "model", "date", "lead"]
    forecasts = forecasts[first + [c for c in forecasts.columns if c not in first]]
    forecasts.to_csv(CLEAN / "forecasts.csv", index=False)
    print("Saved data/clean/forecasts.csv")


if __name__ == "__main__":
    main()
