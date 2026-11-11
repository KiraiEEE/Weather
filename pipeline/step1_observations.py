"""Daily Tmax, Tmin and rain per station, decoded from SYNOP reports.

NOAA Global-Hourly up to 2025-08-24 (NOAA has no Tunisian reports after that), Ogimet raw SYNOP after.
Meteostat isn't used: its rain for Tunisian stations is DWD MOSMIX output, i.e. a forecast.

Daily windows (UTC):
  tmax(D)  max over 06-18 on D, from the 18 UTC report
  tmin(D)  min over the 24 h ending 06 on D, from the 06 UTC report
  prcp(D)  06 on D to 06 on D+1, from the 24 h total in the 06 UTC report of D+1

Writes data/clean/observations_raw.csv (station, date, tmax, tmin, prcp, source)
"""
import io
import time

import pandas as pd

from pipeline.config import CLEAN, NOAA_FIRST_YEAR, NOAA_LAST_DAY, OGIMET_START, STATIONS
from pipeline.utils import download

NOAA_URL = "https://noaa-global-hourly-pds.s3.amazonaws.com/{year}/{station}099999.csv"
OGIMET_URL = "https://www.ogimet.com/cgi-bin/getsynop"

RAIN_PERIOD_HOURS = {"1": 6, "2": 12, "3": 18, "4": 24, "5": 1, "6": 2, "7": 3, "8": 9, "9": 15}
# When a report says "no rain" (iR = 3) without a number, this is the period it covers in Tunisia.
NO_RAIN_PERIOD = {0: 18, 6: 24, 12: 6, 18: 12}


# reading SYNOP messages
def signed_temp(group):
    """'10367' -> 36.7 ; '11012' -> -1.2 ; None if the group is incomplete."""
    if "/" in group or group[1] not in "01":
        return None
    value = int(group[2:]) / 10
    return -value if group[1] == "1" else value


def rain_mm(group):
    """'6RRRt' rain group -> (hours, mm). 990 means a trace (0 mm), 991-999 mean 0.1-0.9 mm."""
    rrr, period = group[1:4], group[4]
    if "/" in rrr or period not in RAIN_PERIOD_HOURS:
        return None
    rrr = int(rrr)
    mm = 0.0 if rrr == 990 else (rrr - 990) / 10 if rrr > 990 else float(rrr)
    return RAIN_PERIOD_HOURS[period], mm


def parse_synop(message):
    """Read one SYNOP message. Returns {'tmax': .., 'tmin': .., 'rain': [(hours, mm), ...]}.
    Inside each section the groups must come in increasing order of their first digit (WMO rule);
    groups out of order (e.g. radiation groups after the sunshine group 55SSS) are skipped."""
    result = {"rain": []}
    groups = message.replace("=", " ").split()
    if "AAXX" in groups:
        groups = groups[groups.index("AAXX") + 1:]
    if len(groups) < 5 or "NIL" in groups:
        return result
    hour = int(groups[0][2:4])
    rain_flag = groups[2][0]          # iR: 0/1/2 = rain groups present, 3 = no rain, 4 = not measured
    section, last_digit = 1, -1
    for g in groups[4:]:
        if g == "333":
            section, last_digit = 3, -1
            continue
        if g in ("222", "444", "555"):
            section = 0               # sections we don't use
            continue
        if section == 0 or len(g) != 5 or not g[0].isdigit() or int(g[0]) <= last_digit:
            continue
        last_digit = int(g[0])
        if g[0] == "6" and rain_flag in "012":
            amount = rain_mm(g)
            if amount:
                result["rain"].append(amount)
        elif section == 3 and g[0] == "1":
            result["tmax"] = signed_temp(g)
        elif section == 3 and g[0] == "2":
            result["tmin"] = signed_temp(g)
        elif section == 3 and g[0] == "7" and "/" not in g:
            value = int(g[1:])
            result["rain"].append((24, 0.0 if value == 9999 else value / 10))
    if rain_flag == "3" and hour in NO_RAIN_PERIOD:
        result["rain"].append((NO_RAIN_PERIOD[hour], 0.0))
    return result


# from reports to daily values
def to_daily(temps, rain):
    """temps: DataFrame(time, kind, value) ; rain: DataFrame(time, hours, mm) -> daily table."""
    temps = temps.assign(time=pd.to_datetime(temps["time"]))
    rain = rain.assign(time=pd.to_datetime(rain["time"]))
    tmax = temps[(temps.kind == "tmax") & (temps.time.dt.hour == 18)]
    tmin = temps[(temps.kind == "tmin") & (temps.time.dt.hour == 6)]
    tmax = tmax.groupby(tmax.time.dt.normalize())["value"].last().rename("tmax")
    tmin = tmin.groupby(tmin.time.dt.normalize())["value"].last().rename("tmin")

    # Some reports contain the 24 h rain twice (whole mm and tenths): keep the more precise one.
    def most_precise(values):
        precise = values[values % 1 != 0]
        return precise.iloc[0] if len(precise) else values.iloc[0]
    rain = rain.groupby(["time", "hours"])["mm"].agg(most_precise).reset_index()

    one_day = pd.Timedelta(days=1)
    r24 = rain[(rain.hours == 24) & (rain.time.dt.hour == 6)]
    r24 = pd.Series(r24.mm.values, index=r24.time.dt.normalize() - one_day)       # belongs to day before
    # backup when the 24 h total is missing: 12 h at 18 UTC (day D) + 12 h at 06 UTC (day D+1)
    r12 = rain[rain.hours == 12]
    evening = r12[r12.time.dt.hour == 18]
    night = r12[r12.time.dt.hour == 6]
    evening = pd.Series(evening.mm.values, index=evening.time.dt.normalize())
    night = pd.Series(night.mm.values, index=night.time.dt.normalize() - one_day)
    prcp = r24.combine_first(evening.add(night)).rename("prcp")       # add() gives NaN if one half is missing

    daily = pd.concat([tmax, tmin, prcp[~prcp.index.duplicated()]], axis=1).sort_index()
    daily.index.name = "date"
    return daily


# NOAA
def noaa_station(station):
    temps, rain = [], []
    last_year = pd.Timestamp(NOAA_LAST_DAY).year
    for year in range(NOAA_FIRST_YEAR, last_year + 1):
        content = download(NOAA_URL.format(year=year, station=station))
        if content is None:
            continue
        df = pd.read_csv(io.BytesIO(content), low_memory=False)
        times = pd.to_datetime(df["DATE"])
        # KA = extreme temperature: "period,code,value,quality" (code M = max, N = min)
        for col in ("KA1", "KA2"):
            if col not in df:
                continue
            for t, text in zip(times, df[col], strict=True):
                if isinstance(text, str):
                    _, code, value, quality = text.split(",")
                    if code in "MN" and value not in ("+9999", "-9999") and quality not in ("3", "7"):
                        temps.append((t, "tmax" if code == "M" else "tmin", int(value) / 10))
        # AA = rain: "hours,depth,condition,quality". Quality 2 ("suspect") is kept on purpose:
        # NOAA uses it whenever a report has two 24 h values (9.0 and 8.6), i.e. on most rain days.
        for col in ("AA1", "AA2", "AA3"):
            if col not in df:
                continue
            for t, text in zip(times, df[col], strict=True):
                if isinstance(text, str):
                    hours, depth, _, quality = text.split(",")
                    if depth != "9999" and quality not in ("3", "7"):
                        rain.append((t, int(hours), int(depth) / 10))
    return to_daily(pd.DataFrame(temps, columns=["time", "kind", "value"]),
                    pd.DataFrame(rain, columns=["time", "hours", "mm"]))


# Ogimet
def ogimet_all_stations():
    """All Tunisian SYNOP messages (station ids start with 607), 10 days per request."""
    rows = []
    start = pd.Timestamp(OGIMET_START)
    yesterday = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
    while start <= yesterday:
        end = min(start + pd.Timedelta(days=10), yesterday + pd.Timedelta(days=1)) - pd.Timedelta(hours=1)
        params = {"block": "607", "begin": start.strftime("%Y%m%d%H%M"), "end": end.strftime("%Y%m%d%H%M")}
        print(f"  Ogimet {start.date()} -> {end.date()}")
        started = time.time()
        content = download(OGIMET_URL, params, check=lambda c: len(c) > 0
                           and not c[:200].lower().startswith((b"status", b"<")) and b"sorry" not in c[:200].lower())
        for line in content.decode("latin-1").splitlines():
            parts = line.split(",", 6)        # station,year,month,day,hour,minute,message
            if len(parts) == 7 and parts[0].isdigit():
                t = pd.Timestamp(int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]))
                rows.append((parts[0], t, parts[6]))
        if time.time() - started > 0.5:       # real download (not cached): pause so we do not overload Ogimet
            time.sleep(8)
        start = end + pd.Timedelta(hours=1)
    return pd.DataFrame(rows, columns=["station", "time", "message"]).drop_duplicates(["station", "time"], keep="last")


def ogimet_station(messages):
    temps, rain = [], []
    for t, msg in zip(messages.time, messages.message, strict=True):
        report = parse_synop(msg)
        for kind in ("tmax", "tmin"):
            if report.get(kind) is not None:
                temps.append((t, kind, report[kind]))
        for hours, mm in report["rain"]:
            rain.append((t, hours, mm))
    return to_daily(pd.DataFrame(temps, columns=["time", "kind", "value"]),
                    pd.DataFrame(rain, columns=["time", "hours", "mm"]))


def main():
    print("Downloading Ogimet SYNOP messages (about 20 minutes the first time)...")
    messages = ogimet_all_stations()
    tables, checks = [], []
    for station, (name, *_rest) in STATIONS.items():
        print(f"{name}: NOAA history")
        noaa = noaa_station(station)
        ogimet = ogimet_station(messages[messages.station == station])
        both = noaa.join(ogimet, lsuffix="_noaa", rsuffix="_ogimet", how="inner")
        for var in ("tmax", "tmin", "prcp"):
            diff = (both[f"{var}_noaa"] - both[f"{var}_ogimet"]).abs().dropna()
            checks.append({"station": name, "variable": var, "days": len(diff), "same_value_%": 100 * (diff < 0.05).mean()})
        # NOAA until its last day, Ogimet afterwards
        last_noaa = noaa.index.max() if len(noaa) else pd.Timestamp(OGIMET_START)
        table = pd.concat([noaa.assign(source="noaa"), ogimet[ogimet.index > last_noaa].assign(source="ogimet")])
        tables.append(table.reset_index().assign(station=station))
        print(f"  {len(table)} days, tmax {table.tmax.count()}, tmin {table.tmin.count()}, rain {table.prcp.count()}")

    obs = pd.concat(tables)[["station", "date", "tmax", "tmin", "prcp", "source"]]
    obs.to_csv(CLEAN / "observations_raw.csv", index=False)
    pd.DataFrame(checks).to_csv(CLEAN / "noaa_vs_ogimet_check.csv", index=False)
    print("Saved data/clean/observations_raw.csv")


if __name__ == "__main__":
    main()
