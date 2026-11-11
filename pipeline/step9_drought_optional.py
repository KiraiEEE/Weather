"""SPI-3 forecast 1-3 months ahead (optional).

Station records start in 2014, too short for SPI, so this uses ERA5 monthly rain from 1991
via Open-Meteo. It describes large-scale drought, not station rain.
Inputs: SPI-1/3/6, month, NAO (NOAA CPC). LightGBM trained year by year, compared
with persistence and with climatology (SPI = 0).

Writes results/drought_spi.md
"""
import io

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy import stats

from pipeline.config import LGBM_PARAMS, RESULTS, STATIONS
from pipeline.utils import download, mae

ERA5_URL = "https://archive-api.open-meteo.com/v1/archive"
NAO_URL = "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii"
FEATURES = ["spi1", "spi3", "spi6", "month_sin", "month_cos", "nao", "nao_prev", "lat", "lon"]


def monthly_rain(lat, lon):
    end = (pd.Timestamp.now().replace(day=1) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    params = {"latitude": lat, "longitude": lon, "start_date": "1991-01-01", "end_date": end,
              "daily": "precipitation_sum", "timezone": "GMT", "format": "csv"}
    table = pd.read_csv(io.BytesIO(download(ERA5_URL, params, check=lambda c: c.startswith(b"latitude"))), skiprows=3)
    table.columns = ["date", "rain"]
    return table.set_index(pd.to_datetime(table.date)).rain.resample("MS").sum()


def spi(rain, months):
    """SPI: fit a gamma distribution per calendar month on 1991-2020, turn amounts into a normal score."""
    total = rain.rolling(months).sum()
    result = pd.Series(np.nan, index=total.index)
    for month in range(1, 13):
        values = total[total.index.month == month].dropna()
        reference = values[values.index.year <= 2020]
        dry_share = (reference == 0).mean()
        shape, _, scale = stats.gamma.fit(reference[reference > 0], floc=0)
        prob = dry_share + (1 - dry_share) * stats.gamma.cdf(values, shape, scale=scale)
        result[values.index] = stats.norm.ppf(np.clip(prob, 0.001, 0.999))
    return result


def nao_index():
    table = pd.read_csv(io.BytesIO(download(NAO_URL)), sep=r"\s+", header=None, names=["year", "month", "nao"])
    return pd.Series(table.nao.values, index=pd.to_datetime(dict(year=table.year, month=table.month, day=1)))


def main():
    nao = nao_index()
    rows = []
    for station, (name, lat, lon, _elev) in STATIONS.items():
        print(f"{name}: ERA5 monthly rain")
        rain = monthly_rain(lat, lon)
        table = pd.DataFrame({"spi1": spi(rain, 1), "spi3": spi(rain, 3), "spi6": spi(rain, 6)})
        table["nao"] = nao.reindex(table.index)
        table["nao_prev"] = table.nao.shift(1)
        table["month_sin"] = np.sin(2 * np.pi * table.index.month / 12)
        table["month_cos"] = np.cos(2 * np.pi * table.index.month / 12)
        for ahead in (1, 2, 3):
            table[f"target_{ahead}"] = table.spi3.shift(-ahead)      # SPI-3 `ahead` months later
        rows.append(table.assign(station=station, lat=lat, lon=lon))
    data = pd.concat(rows).dropna(subset=["spi6"])

    results = []
    for ahead in (1, 2, 3):
        target = f"target_{ahead}"
        rows_ok = data.dropna(subset=[target])
        predictions = []
        for year in range(2011, rows_ok.index.year.max() + 1):
            # the target of month t is known `ahead` months later, so stop training early enough
            train = rows_ok[rows_ok.index < pd.Timestamp(year, 1, 1) - pd.DateOffset(months=ahead)]
            test = rows_ok[rows_ok.index.year == year]
            model = LGBMRegressor(**{**LGBM_PARAMS, "n_estimators": 200}).fit(train[FEATURES], train[target])
            predictions.append(test.assign(pred=model.predict(test[FEATURES])))
        p = pd.concat(predictions)
        drought = p[target] <= -1
        results.append({"months ahead": ahead, "test months": len(p),
                        "MAE lightgbm": mae(p[target], p.pred), "MAE no-change": mae(p[target], p.spi3),
                        "MAE climatology": mae(p[target], 0 * p.pred),
                        "drought months": int(drought.sum()),
                        "droughts found % (lightgbm)": 100 * (drought & (p.pred <= -1)).sum() / max(drought.sum(), 1),
                        "droughts found % (no-change)": 100 * (drought & (p.spi3 <= -1)).sum() / max(drought.sum(), 1)})
    (RESULTS / "drought_spi.md").write_text(
        "# Drought (SPI-3) forecast, 12 stations, years 2011 onwards\n\n"
        + pd.DataFrame(results).to_markdown(index=False, floatfmt=".2f") + "\n", encoding="utf-8")
    print("Saved results/drought_spi.md")


if __name__ == "__main__":
    main()
