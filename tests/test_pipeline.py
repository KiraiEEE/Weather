"""Tests for the parts that are easy to get wrong: SYNOP decoding, time windows, leakage."""
import numpy as np
import pandas as pd
import pytest

from pipeline.step1_observations import parse_synop, to_daily
from pipeline.step2_forecasts import daily_forecasts
from pipeline.step4_features import TEMP_FEATURES, build_features


# step 1: reading SYNOP
def test_synop_temperatures_and_rain():
    evening = parse_synop("AAXX 01184 60750 12960 00505 10324 20219 30134 40168 53004 60002 333 10367=")
    assert evening["tmax"] == pytest.approx(36.7)
    assert evening["rain"] == [(12, 0.0)]
    morning = parse_synop("AAXX 15064 60725 11560 99905 10104 20048 30134 40168 58004 69904 333 21012 70135=")
    assert morning["tmin"] == pytest.approx(-1.2)
    assert (24, 13.5) in morning["rain"]           # 7RRRR group, tenths of mm


def test_radiation_group_is_not_read_as_tmin():
    # '22135' comes after the sunshine group 55103, so it is a radiation group, not a minimum temperature
    report = parse_synop("AAXX 01004 60750 12560 30706 10311 20264 30128 40162 56008 60003 83500 333 55103 22135 83630=")
    assert "tmin" not in report


def test_daily_values_use_the_right_days():
    temps = pd.DataFrame({"time": ["2025-03-15 18:00", "2025-03-15 06:00"], "kind": ["tmax", "tmin"],
                          "value": [25.0, 10.0]})
    rain = pd.DataFrame({"time": ["2025-03-16 06:00", "2025-03-16 06:00"], "hours": [24, 24], "mm": [9.0, 8.6]})
    day = to_daily(temps, rain).loc["2025-03-15"]
    assert day.tmax == 25.0 and day.tmin == 10.0
    assert day.prcp == pytest.approx(8.6)          # rain reported on the 16th at 06 UTC belongs to the 15th


# step 2: no future model runs
def test_forecasts_only_use_runs_from_the_issue_day():
    hours = pd.date_range("2025-01-01", periods=24 * 20, freq="h")
    hourly = pd.DataFrame(index=hours)
    for k in range(0, 8):
        run_day = (hours.normalize() - pd.Timedelta(days=k)).map(pd.Timestamp.toordinal).astype(float)
        hourly["temperature_2m" if k == 0 else f"temperature_2m_previous_day{k}"] = run_day
        if k:
            hourly[f"precipitation_previous_day{k}"] = run_day
    daily = daily_forecasts(hourly).dropna(subset=["fc_tmax", "fc_tmin"])
    issue_day = (daily.date - pd.to_timedelta(daily.lead, unit="D")).map(pd.Timestamp.toordinal)
    assert np.allclose(daily.fc_tmax, issue_day)
    assert np.allclose(daily.fc_tmin, issue_day)
    rain = daily.dropna(subset=["fc_prcp"])
    assert np.allclose(rain.fc_prcp, 24 * issue_day[rain.index])    # 24 hours, all from the same run


# step 4: no future observations
def test_features_do_not_use_future_observations():
    rng = np.random.default_rng(0)
    days = pd.date_range("2014-01-01", "2025-03-31")
    obs = pd.DataFrame({"station": "60750", "date": days, "tmax": 25 + rng.normal(0, 3, len(days)),
                        "tmin": 12 + rng.normal(0, 3, len(days)), "prcp": rng.gamma(0.3, 5, len(days))})
    fdays = pd.date_range("2024-06-01", "2025-03-31")
    forecasts = pd.concat([pd.DataFrame({"station": "60750", "model": "IFS", "date": fdays, "lead": lead,
                                         "fc_tmax": 25 + rng.normal(0, 3, len(fdays)),
                                         "fc_tmin": 12 + rng.normal(0, 3, len(fdays)),
                                         "fc_prcp": rng.gamma(0.3, 5, len(fdays)), "fc_dew": 10.0,
                                         "fc_cloud": 30.0, "fc_wind": 12.0}) for lead in (1, 3, 7)])
    before = build_features(forecasts, obs)
    for lead in (1, 3, 7):
        target_day = pd.Timestamp("2025-02-15")
        issue = target_day - pd.Timedelta(days=lead)
        changed = obs.copy()
        changed.loc[changed.date > issue, ["tmax", "tmin"]] += 50     # not known yet at the issue day
        changed.loc[changed.date >= issue, "prcp"] += 50              # rain of the issue day ends next morning
        after = build_features(forecasts, changed)
        row = (before.date == target_day) & (before.lead == lead)
        columns = TEMP_FEATURES + ["last_prcp", "rain_30d"]
        pd.testing.assert_frame_equal(before.loc[row, columns].reset_index(drop=True),
                                      after.loc[row, columns].reset_index(drop=True))
